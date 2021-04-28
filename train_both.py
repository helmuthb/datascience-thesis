import os
import argparse

import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope, plot_model

from lib.deeplab import add_deeplab_features
from lib.preprocess import preprocess
from lib.ssdlite import (
    add_ssdlite_features, detection_head, get_anchor_boxes_cwh,
    ssdlite_base_layers)
from lib.tfr_utils import read_tfrecords
from lib.mobilenet import mobilenetv2
from lib.losses import SSDLoss


def ssd_deeplab_model(size, n_seg, n_det):
    """
    """
    input_layer = tf.keras.layers.Input(
        shape=(size[0], size[1], 3))
    # base model
    base = mobilenetv2(input_layer)
    # add deeplab layers
    deeplab_output = add_deeplab_features(base, n_seg)
    # add SSDlite layers
    ext_base = add_ssdlite_features(base)
    l1, l2, l3, l4, l5, l6 = ssdlite_base_layers(ext_base)
    # add class and location predictions
    ssd_output = detection_head(n_det, l1, l2, l3, l4, l5, l6)
    # create model
    combined_model = tf.keras.Model(
        inputs=input_layer,
        outputs=[deeplab_output, ssd_output])
    # calculate anchor boxes
    anchor_boxes_cwh = get_anchor_boxes_cwh(l1, l2, l3, l4, l5, l6)
    print(combined_model.summary())
    plot_model(combined_model, to_file='model-both.png')
    # return combined model and the anchors
    return combined_model, anchor_boxes_cwh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        help='Directory with TFrecords.',
        required=True
    )
    args = parser.parse_args()

    # build model
    model, anchor_boxes_cwh = ssd_deeplab_model((300, 300), 11, 11)

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
    train_ds_orig = read_tfrecords(os.path.join(args.data, 'det_train.tfrec'))
    val_ds_orig = read_tfrecords(os.path.join(args.data, 'det_val.tfrec'))

    # Preprocess data
    train_ds = train_ds_orig.map(
        preprocess((300, 300), anchor_boxes_cwh, 11, 11))
    val_ds = val_ds_orig.map(
        preprocess((300, 300), anchor_boxes_cwh, 11, 11))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=8)
    val_ds_batch = val_ds.batch(batch_size=8)

    # load model or train
    if os.path.isdir("model-both"):
        with custom_object_scope({'SSDLoss': SSDLoss}):
            model = tf.keras.models.load_model("model-both")
    else:
        # save empty model
        model.save("model-both-empty")
        # Perform training
        model.fit(train_ds_batch, epochs=1, validation_data=val_ds_batch)
        # save model
        model.save("model-both")

    # perform inference on validation set
    preds = model.predict(val_ds_batch)
    print(preds.shape)


if __name__ == '__main__':
    main()
