import os
import argparse
import timeit

import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
import cv2

from lib.preprocess import (
    preprocess, filter_classes_bbox, filter_classes_mask, subset_names)
from lib.np_bbox_utils import BBoxUtils
from lib.ssdlite import (
    add_ssdlite_features, get_default_boxes_cwh,
    ssdlite_base_layers)
from lib.tfr_utils import read_tfrecords
from lib.mobilenet import mobilenetv2
from lib.losses import SSDLoss, DeeplabLoss
from lib.visualize import annotate_boxes, annotate_segmentation
import lib.rs19_classes as rs19


def ssd_defaults(size):
    """
    """
    input_layer = tf.keras.layers.Input(
        shape=(size[0], size[1], 3))
    # base model
    base = mobilenetv2(input_layer)
    # add SSDlite layers
    ext_base = add_ssdlite_features(base)
    l1, l2, l3, l4, l5, l6 = ssdlite_base_layers(ext_base)
    # calculate default boxes
    default_boxes_cwh = get_default_boxes_cwh(l1, l2, l3, l4, l5, l6)
    return default_boxes_cwh


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
    os.makedirs(f"{outdir}/seg-annotated", exist_ok=True)

    # number of classes
    n_seg = len(rs19.seg_subset)
    n_det = len(rs19.det_subset)

    # names of detection subsets
    det_names = subset_names(rs19.det_subset)

    # get default boxes
    default_boxes_cwh = ssd_defaults((300, 300))

    # Bounding box utility object
    bbox_util = BBoxUtils(n_det, default_boxes_cwh, min_confidence=0.01)

    # Load validation data
    val_ds_orig = read_tfrecords(
            os.path.join(args.tfrecords, 'val.tfrec'))

    # Filter for classes of interest
    val_ds_filtered_det = val_ds_orig.map(
        filter_classes_bbox(rs19.det_classes, rs19.det_subset)
    )
    val_ds_filtered = val_ds_filtered_det.map(
        filter_classes_mask(rs19.seg_classes, rs19.seg_subset)
    )

    # Preprocess data
    val_ds = val_ds_filtered.map(
        preprocess((300, 300), bbox_util, n_seg))

    # Create batches
    val_ds_batch = val_ds.batch(batch_size=128)

    # load model
    with custom_object_scope({
                'SSDLoss': SSDLoss,
                'DeeplabLoss': DeeplabLoss
            }):
        model = tf.keras.models.load_model(args.model)

    # perform inference on validation set
    preds = model.predict(val_ds_batch)
    print(preds[0].shape)  # deeplab
    print(preds[1].shape)  # ssd

    # combine ...
    i_origs = val_ds_filtered.as_numpy_iterator()
    for s, p, o in zip(preds[0], preds[1], i_origs):
        image, boxes_xy, boxes_cl, mask, name = o
        mask = mask[0, :, :]
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
        annotate_boxes(image, p_boxes_xy, p_boxes_cl, p_boxes_sc,
                       det_names, file_name)
        # annotate segmentation
        file_prefix = f"{outdir}/seg-annotated/{name}"
        annotate_segmentation(image, mask, s, file_prefix)
    # runtime for inference
    runtime = timeit.timeit(lambda: model.predict(val_ds_batch), number=1)
    ds_size = 0
    for img in val_ds_batch:
        ds_size += 1
    print(f"Inference time: {runtime/ds_size} sec.")


if __name__ == '__main__':
    main()
