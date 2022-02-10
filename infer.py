import os
import argparse
import timeit

import tensorflow as tf
import cv2
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

from lib.combined import ssd_deeplab_model
from lib.preprocess import (
    filter_empty_samples, preprocess, filter_classes_bbox,
    filter_classes_mask, subset_names)
from lib.tf_bbox_utils import BBoxUtils, to_cw
from lib.evaluate import DetEval, SegEval
from lib.tfr_utils import read_tfrecords
from lib.visualize import annotate_boxes, annotate_segmentation
import lib.rs19_classes as rs19


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
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Number of samples per batch (default=8).'
    )
    parser.add_argument(
        '--subset-only',
        action='store_true',
        help='Use only subset of classes instead of all.'
    )
    parser.add_argument(
        '--ssd-only',
        action='store_true',
        help='Only train SSDlite model.'
    )
    parser.add_argument(
        '--deeplab-only',
        action='store_true',
        help='Ony train DeepLab model.'
    )
    parser.add_argument(
        '--model-width',
        type=int,
        default=224,
        help='Specify image width for model'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=1920,
        help='Specify original image width'
    )
    parser.add_argument(
        '--image-height',
        type=int,
        default=1080,
        help='Specify original image height'
    )
    args = parser.parse_args()
    outdir = args.out_samples
    batch_size = args.batch_size
    subset_only = args.subset_only
    ssd_only = args.ssd_only
    deeplab_only = args.deeplab_only
    model_width = args.model_width
    image_width = args.image_width
    image_height = args.image_height
    # create output directory if missing
    os.makedirs(f"{outdir}/orig-annotated", exist_ok=True)
    os.makedirs(f"{outdir}/pred-annotated", exist_ok=True)
    os.makedirs(f"{outdir}/pred-data", exist_ok=True)
    os.makedirs(f"{outdir}/seg-annotated", exist_ok=True)

    # number & names of classes
    if subset_only:
        n_seg = len(rs19.seg_subset)
        n_det = len(rs19.det_subset)
        seg_names = rs19.seg_classes
        det_names = subset_names(rs19.det_subset)
    else:
        n_seg = len(rs19.seg_classes)
        n_det = len(rs19.det_classes)
        seg_names = rs19.seg_classes
        det_names = rs19.det_classes

    # build model
    models = ssd_deeplab_model((model_width, model_width), n_det, n_seg)
    model, default_boxes_cw, base, deeplab, ssd = models

    # Bounding box utility object
    bbox_util = None if deeplab_only else BBoxUtils(
        n_det, default_boxes_cw, min_confidence=0.01)

    # Number of classes for segmentation = 0 if only object detection
    if ssd_only:
        n_seg = 0

    # Load validation data
    val_ds = read_tfrecords(
            os.path.join(args.tfrecords, 'val.tfrec'))

    # Filter for classes of interest
    if subset_only:
        val_ds = val_ds.map(
            filter_classes_bbox(rs19.det_classes, rs19.det_subset)
        )
        val_ds = val_ds.map(
            filter_classes_mask(rs19.seg_classes, rs19.seg_subset)
        )

    # Filter out empty samples - if SSD requested
    if not deeplab_only:
        val_ds = val_ds.filter(filter_empty_samples)

    # Preprocess data
    val_ds_preprocessed = val_ds.map(
        preprocess((model_width, model_width), bbox_util, n_seg))

    # Create batches
    val_ds_batch = val_ds_preprocessed.batch(batch_size=batch_size)

    # load model
    model = tf.keras.models.load_model(args.model)
    plot_model(model, to_file='infer-model.png', show_shapes=True)

    # perform inference on validation set
    pr = model.predict(val_ds_batch)
    if deeplab_only:
        pr = (pr,)

    # evaluation & plots
    seg_eval = SegEval(n_seg)
    det_eval = DetEval(n_det)
    i_origs = val_ds.as_numpy_iterator()
    ds_size = 0
    for x in tqdm(zip(*pr, i_origs)):
        ds_size += 1
        if ssd_only:
            p_conf, p_locs, (image, g_cl, g_xy, g_segs, name) = x
        elif deeplab_only:
            p_segs, (image, g_cl, g_xy, g_segs, name) = x
        else:
            p_conf, p_locs, p_segs, (image, g_cl, g_xy, g_segs, name) = x
        name = name.decode('utf-8')
        if not deeplab_only:
            p_cl, p_sc, p_xy = bbox_util.pred_to_boxes_np(p_conf, p_locs)
            det_eval.evaluate_sample(g_cl, g_xy, p_cl, p_sc, p_xy)
            g_xy = g_xy.copy()
            g_xy[:, 0] *= image_width
            g_xy[:, 1] *= image_height
            g_xy[:, 2] *= image_width
            g_xy[:, 3] *= image_height
            img = image.copy()
            for box_xy in g_xy:
                top_left = (int(round(box_xy[0])), int(round(box_xy[1])))
                bot_right = (int(round(box_xy[2])), int(round(box_xy[3])))
                color = (0, 255, 0)
                cv2.rectangle(img, top_left, bot_right, color, 2)
            cv2.imwrite(f"{outdir}/orig-annotated/{name}.jpg", img)
            file_name = f"{outdir}/pred-annotated/{name}.jpg"
            annotate_boxes(image, p_cl, p_sc, p_xy, det_names, file_name)
            # create output file for evaluation
            p_cw = to_cw(p_xy)
            with open(f"{outdir}/pred-data/{name}.txt", "w") as f:
                for i, cw in enumerate(p_cw):
                    xy = p_xy[i]
                    cl = p_cl[i].item()
                    sc = p_sc[i].item()
                    b_str = " ".join([str(b) for b in cw])
                    b2_str = " ".join([str(b) for b in xy])
                    f.write(f"{cl} {sc} {b_str}\n")
                    f.write(f"# XY: {cl} {sc} {b2_str}\n")
        if not ssd_only:
            # evaluation of segmentation
            seg_eval.evaluate_sample(g_segs, p_segs)
            # annotate segmentation
            file_prefix = f"{outdir}/seg-annotated/{name}"
            annotate_segmentation(image, g_segs, p_segs, file_prefix)
    # runtime for inference
    print("Calculating running time ...")
    runtime = timeit.timeit(lambda: model.predict(val_ds_batch), number=1)
    # runtime for preprocessing
    print("Calculating preprocessing time ...")
    runtime_pre = timeit.timeit(lambda: sum(1 for i in val_ds), number=1)
    print(f"Inference time: {runtime/ds_size} sec.")
    print(f"Preprocessing time: {runtime_pre/ds_size} sec.")
    print(f"Network time: {(runtime-runtime_pre)/ds_size} sec.")
    if not ssd_only:
        print("Segmentation metrics")
        print(f"Mean IoU: {seg_eval.mean_iou():.2%}")
        print(f"Pixel accuracy: {seg_eval.pixel_accuracy():.2%}")
        print(f"Mean accuracy: {seg_eval.mean_accuracy():.2%}")
        print(f"Mean dice coefficient: {seg_eval.mean_dice_coefficient():.2%}")
        print(f"Frequency-weighted IoU: {seg_eval.fw_iou():.2%}")
        # create miou / confusion matrix plots
        seg_eval.plot_iou(seg_names, f"{outdir}/iou-plot.png")
        seg_eval.plot_cm(seg_names, f"{outdir}/cm-plot.png")
    if not deeplab_only:
        print("Object detection metrics")
        # calculate mean average precision / recall
        prec_rec = det_eval.mean_average_precision_recall()
        print(f"Mean Average Precision & Recall: {prec_rec}")
        prec_rec2 = det_eval.mean_average_precision_recall(min_iou=0.1)
        print(f"Mean Average Precision & Recall (min-IoU=0.1): {prec_rec2}")
        for i in range(0, n_det):
            det_eval.plot_precision_recall_curves(
                i,
                det_names[i],
                [x/10. for x in range(10)],
                f"{outdir}/precrecs-{i}.png"
            )
        det_eval.plot_precision(det_names, f"{outdir}/det-precision.png")
        det_eval.plot_recall(det_names, f"{outdir}/det-recall.png")


if __name__ == '__main__':
    main()
