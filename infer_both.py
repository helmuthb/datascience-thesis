from collections import defaultdict
import os
import argparse
import timeit

import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
import cv2
import numpy as np

from lib.preprocess import preprocess
from lib.np_bbox_utils import BBoxUtils
from lib.ssdlite import (
    add_ssdlite_features, get_anchor_boxes_cwh,
    ssdlite_base_layers)
from lib.tfr_utils import read_tfrecords
from lib.mobilenet import mobilenetv2
from lib.losses import SSDLoss
from lib.visualize import annotate_detection


# Names of classes used in object detection
det_classes = ["background", "buffer-stop", "crossing", "switch-indicator",
               "switch-left", "switch-right", "switch-static",
               "switch-unknown", "track-signal-back", "track-signal-front",
               "track-sign-front"]


def ssd_anchors(size):
    """
    """
    input_layer = tf.keras.layers.Input(
        shape=(size[0], size[1], 3))
    # base model
    base = mobilenetv2(input_layer)
    # add SSDlite layers
    ext_base = add_ssdlite_features(base)
    l1, l2, l3, l4, l5, l6 = ssdlite_base_layers(ext_base)
    # calculate anchor boxes
    anchor_boxes_cwh = get_anchor_boxes_cwh(l1, l2, l3, l4, l5, l6)
    return anchor_boxes_cwh


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
        '--out-samples',
        type=str,
        help='Directory for output samples.',
        required=True
    )
    args = parser.parse_args()
    outdir = args.out_samples
    # create output directory if missing
    os.makedirs(f"{outdir}/orig-annotated", exist_ok=True)
    os.makedirs(f"{outdir}/pred-annotated", exist_ok=True)

    # get anchor boxes
    anchor_boxes_cwh = ssd_anchors((300, 300))

    # Bounding box utility object
    bbox_util = BBoxUtils(11, anchor_boxes_cwh, min_confidence=0.01)

    # Load validation data
    val_ds_orig = read_tfrecords(
            os.path.join(args.tfrecords, 'val.tfrec'))

    # Preprocess data
    val_ds = val_ds_orig.map(
        preprocess((300, 300), bbox_util, 11))

    # find for each validation sample the highest class
    for sample in val_ds:
        boxes = sample[1][1]
        num_boxes = boxes.shape[1]
        max_c = defaultdict(int)
        for b in range(num_boxes):
            c = np.argmax(boxes[b, :-4])
            # print(boxes[b, :-4])
            max_c[c] += 1
        # print(max_c)

    # Create batches
    val_ds_batch = val_ds.batch(batch_size=8)

    # load model
    with custom_object_scope({'SSDLoss': SSDLoss}):
        model = tf.keras.models.load_model(args.model)

    # perform inference on validation set
    preds = model.predict(val_ds_batch)
    print(preds[0].shape)  # deeplab
    print(preds[1].shape)  # ssd

    # find for each prediction the highest class
    num_samples = preds[1].shape[0]
    num_boxes = preds[1].shape[1]
    for i in range(num_samples):
        max_c = defaultdict(int)
        for b in range(num_boxes):
            c = np.argmax(preds[1][i, b, :-4])
            max_c[c] += 1
        # print(i, max_c)
    # combine ...
    i_origs = val_ds_orig.as_numpy_iterator()
    for p, o in zip(preds[1], i_origs):
        image, boxes_xy, boxes_cl, mask, name = o
        name = name.decode('utf-8')
        boxes_xy = boxes_xy.copy()
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
        cv2.imwrite(f"{outdir}/orig-annotated/{name}.jpg", img)
        p_boxes_xy, p_boxes_cl, p_boxes_sc = bbox_util.pred_to_boxes(p)
        file_name = f"{outdir}/pred-annotated/{name}.jpg"
        annotate_detection(image, p_boxes_xy, p_boxes_cl, None,
                           det_classes, file_name)
    # runtime for inference
    runtime = timeit.timeit(lambda: model.predict(val_ds_batch), number=1)
    ds_size = 0
    for img in val_ds_batch:
        ds_size += 1
    print(f"Inference time: {runtime/ds_size} sec.")


if __name__ == '__main__':
    main()
