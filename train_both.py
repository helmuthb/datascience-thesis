from contextlib import redirect_stdout
import os
import argparse
import timeit

import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope, plot_model
import cv2

from lib.deeplab import add_deeplab_features
from lib.preprocess import preprocess
from lib.ssdlite import (
    add_ssdlite_features, detection_head, get_anchor_boxes_cwh,
    ssdlite_base_layers)
from lib.tfr_utils import read_tfrecords
from lib.mobilenet import mobilenetv2
from lib.losses import SSDLoss


def ssd_deeplab_model(size, n_seg, n_det, out_folder):
    """
    """
    input_layer = tf.keras.layers.Input(
        shape=(size[0], size[1], 3))
    # base model
    base = mobilenetv2(input_layer)
    plot_model(base, to_file=os.path.join(out_folder, 'model-mobilenet.png'))
    with open(os.path.join(out_folder, 'model-mobilenet.txt'), mode='w') as f:
        with redirect_stdout(f):
            base.summary()
    # Keras implementation
    mnet_keras = tf.keras.applications.MobileNetV2(
        input_shape=(300, 300, 3),
        include_top=False,
        weights=None,
        classes=11
    )
    plot_model(mnet_keras,
            to_file=os.path.join(out_folder, 'model-mobilenet-keras.png'))
    with open(os.path.join(out_folder, 'model-mobilenet-keras.txt'),
            mode='w') as f:
        with redirect_stdout(f):
            mnet_keras.summary()
    # add deeplab layers
    deeplab_output = add_deeplab_features(base, n_seg)
    # add SSDlite layers
    ext_base = add_ssdlite_features(base)
    l1, l2, l3, l4, l5, l6 = ssdlite_base_layers(ext_base)
    # add class and location predictions
    ssd_output = detection_head(n_det, l1, l2, l3, l4, l5, l6)
    # create models
    ssd_model = tf.keras.Model(
        inputs=input_layer,
        outputs=ssd_output
    )
    plot_model(ssd_model, to_file=os.path.join(out_folder, 'model-ssd.png'))
    print(ssd_model.summary())
    deeplab_model = tf.keras.Model(
        inputs=input_layer,
        outputs=deeplab_output
    )
    plot_model(deeplab_model,
            to_file=os.path.join(out_folder, 'model-deeplab.png'))
    print(deeplab_model.summary())
    combined_model = tf.keras.Model(
        inputs=input_layer,
        outputs=[deeplab_output, ssd_output])
    # calculate anchor boxes
    anchor_boxes_cwh = get_anchor_boxes_cwh(l1, l2, l3, l4, l5, l6)
    print(combined_model.summary())
    plot_model(combined_model,
            to_file=os.path.join(out_folder, 'model-both.png'))
    # return combined model and the anchors
    return combined_model, anchor_boxes_cwh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tfrecords',
        type=str,
        help='Directory with TFrecords.',
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Directory for models.',
        required=True
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs.',
        required=True
    )
    args = parser.parse_args()

    # build model
    model, anchor_boxes_cwh = ssd_deeplab_model((300, 300), 11, 11, args.model)

    # Loss functions
    losses = {
        "deeplab_output":
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "ssd_output": SSDLoss(11),
    }
    # loss weights
    lossWeights = {
        "deeplab_output": 1.0,
        "ssd_output": 1.0,
    }
    # compile combined model
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses,
        loss_weights=lossWeights,
        run_eagerly=False,
    )

    # Load training & validation data
    train_ds_orig = read_tfrecords(
            os.path.join(args.tfrecords, 'det_train.tfrec'))
    val_ds_orig = read_tfrecords(
            os.path.join(args.tfrecords, 'det_val.tfrec'))

    # Preprocess data
    train_ds = train_ds_orig.map(
        preprocess((300, 300), anchor_boxes_cwh, 11, 11))
    val_ds = val_ds_orig.map(
        preprocess((300, 300), anchor_boxes_cwh, 11, 11))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=8)
    val_ds_batch = val_ds.batch(batch_size=8)

    # load model or train
    if os.path.isdir(os.path.join(args.model, 'model-both')):
        with custom_object_scope({'SSDLoss': SSDLoss}):
            model = tf.keras.models.load_model(
                    os.path.join(args.model, "model-both"))
    else:
        # save empty model
        model.save(os.path.join(args.model, "model-both-empty"))
        # Perform training
        model.fit(
                train_ds_batch,
                epochs=args.epochs,
                validation_data=val_ds_batch)
        # save model
        model.save(os.path.join(args.model, "model-both"))

    # perform inference on validation set
    preds = model.predict(val_ds_batch)
    print(preds[0].shape)  # deeplab
    print(preds[1].shape)  # ssd

    # combine ...
    i_origs = val_ds_orig.as_numpy_iterator()
    for p, o in zip(preds[1], i_origs):
        image, boxes_xy, boxes_cl, mask, name = o
        name = name.decode('utf-8')
        boxes_xy[:, 0] *= 1920
        boxes_xy[:, 1] *= 1080
        boxes_xy[:, 2] *= 1920
        boxes_xy[:, 3] *= 1080
        img = image.copy()
        for box_xy in boxes_xy:
            top_left = (int(round(box_xy[0])), int(round(box_xy[1])))
            bot_right = (int(round(box_xy[2])), int(round(box_xy[3])))
            color = (0, 255, 0)
            cv2.rectangle(img, top_left, bot_right, color, 2)
        cv2.imwrite(f"{args.model}/orig-annotated/{name}.jpg", img)
        print(p.shape, boxes_xy.shape, name)
        break

    # runtime for inference
    print(timeit.timeit(lambda: model.predict(val_ds_batch), number=1))

if __name__ == '__main__':
    main()
