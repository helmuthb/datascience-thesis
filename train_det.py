import os
import argparse
import cv2
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from lib.np_bbox_utils import pred_to_boxes
from lib.ssdlite import ssdlite
from lib.losses import SSDLoss
from lib.tfr_utils import read_tfrecords
from lib.preprocess import preprocess_det


def main():
    min_confidence = .1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        help='Directory with TFrecords.',
        required=True
    )
    args = parser.parse_args()

    # build model
    model, anchors = ssdlite((300, 300), 11)

    # Compile
    model.compile(
        optimizer=optimizers.Adam(),
        loss=SSDLoss(11),
        run_eagerly=False,
    )

    # Load training & validation data
    train_ds_orig = read_tfrecords(os.path.join(args.data, 'det_train.tfrec'))
    val_ds_orig = read_tfrecords(os.path.join(args.data, 'det_val.tfrec'))

    # Preprocess data
    train_ds = train_ds_orig.map(preprocess_det((300, 300), anchors, 11))
    val_ds = val_ds_orig.map(preprocess_det((300, 300), anchors, 11))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=8)
    val_ds_batch = val_ds.batch(batch_size=8)

    # load model or train
    if os.path.isdir("model"):
        with custom_object_scope({'SSDLoss': SSDLoss}):
            model = tf.keras.models.load_model("model")
    else:
        # Perform training
        model.fit(train_ds_batch, epochs=1, validation_data=val_ds_batch)
        # save model
        model.save("model")

    # perform inference on validation set
    preds = model.predict(val_ds_batch)

    # iterate through the validation images
    idx = 0
    for pred, orig in zip(preds, val_ds.as_numpy_iterator()):
        idx += 1
        orig_img, orig_ann = orig
        img = orig_img*256.
        # get boxes from originals and predictions
        # o_boxes_xy, o_boxes_cl, o_boxes_sc = pred_to_boxes(orig_ann, anchors)
        p_boxes_xy, p_boxes_cl, p_boxes_sc = pred_to_boxes(
            pred, anchors, min_confidence)
        # draw bounding boxes
        for one_xy, one_cl, one_sc in zip(p_boxes_xy, p_boxes_cl, p_boxes_sc):
            one_xy *= 300
            top_left = (int(one_xy[0]), int(one_xy[1]))
            bot_right = (int(one_xy[2]), int(one_xy[3]))
            color = (0, 255, 0)
            cv2.rectangle(img, top_left, bot_right, color, 2)
        # save image
        cv2.imwrite(f"img_val{idx:03}.jpg", img)

    # perform inference on training set
    preds = model.predict(train_ds_batch)

    idx = 0
    for pred, orig in zip(preds, train_ds.as_numpy_iterator()):
        idx += 1
        orig_img, orig_ann = orig
        img = orig_img*256.
        # get boxes from originals and predictions
        # o_boxes_xy, o_boxes_cl, o_boxes_sc = pred_to_boxes(orig_ann, anchors)
        p_boxes_xy, p_boxes_cl, p_boxes_sc = pred_to_boxes(
            pred, anchors, min_confidence)
        # draw bounding boxes
        for one_xy, one_cl, one_sc in zip(p_boxes_xy, p_boxes_cl, p_boxes_sc):
            one_xy *= 300
            top_left = (int(one_xy[0]), int(one_xy[1]))
            bot_right = (int(one_xy[2]), int(one_xy[3]))
            color = (0, 255, 0)
            cv2.rectangle(img, top_left, bot_right, color, 2)
        # save image
        cv2.imwrite(f"img_train{idx:03}.jpg", img)

    # result = model.evaluate(val_ds)
    # print(result)


if __name__ == '__main__':
    main()
