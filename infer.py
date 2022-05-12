from inspect import getsourcefile
import math
import os
import argparse
import timeit

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

from lib.combined import ssd_deeplab_model
from lib.preprocess import (
    filter_empty_samples, filter_no_mask, preprocess)
from lib.bbox_utils import BBoxUtils, to_cw
from lib.evaluate import DetEval, SegEval
from lib.tfr_utils import read_tfrecords
from lib.visualize import annotate_boxes, annotate_segmentation
from lib.config import Config


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
        help='Folder or H5-file for trained model.',
        required=True
    )
    parser.add_argument(
        '--load-weights',
        action='store_true',
        help='Use load_weights() to load compatible model.',
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
        '--det-classes',
        type=str,
        help='File with class names for object detection.'
    )
    parser.add_argument(
        '--seg-classes',
        type=str,
        help='File with class names for segmentation.'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        help='Specify configuration yaml file for model.',
        required=True
    )
    args = parser.parse_args()
    tfrecdir = args.tfrecords
    model_path = args.model
    load_weights = args.load_weights
    outdir = args.out_samples
    batch_size = args.batch_size
    det_classes = args.det_classes
    seg_classes = args.seg_classes
    model_config = args.model_config
    # create output directories if missing
    os.makedirs(f"{outdir}/orig-annotated", exist_ok=True)
    os.makedirs(f"{outdir}/pred-annotated", exist_ok=True)
    os.makedirs(f"{outdir}/pred-data", exist_ok=True)
    os.makedirs(f"{outdir}/seg-annotated", exist_ok=True)

    # number & names of classes
    if det_classes is None:
        det_names = []
        n_det = 0
    else:
        with open(det_classes, 'r') as f:
            det_names = f.read().splitlines()
        n_det = len(det_names)
    if seg_classes is None:
        seg_names = []
        n_seg = 0
    else:
        with open(seg_classes, 'r') as f:
            seg_names = f.read().splitlines()
        n_seg = len(seg_names)

    # read model config
    if not os.path.exists(model_config):
        # current script folder ...
        folder = os.path.dirname(getsourcefile(main))
        model_config = f"{folder}/config/{model_config}.cfg"
    config = Config.load_file(model_config)

    # load model width from config
    model_width = config.width

    # build model (we only need the default boxes)
    models = ssd_deeplab_model(n_det, n_seg, config)
    model, _, deeplab, ssd, default_boxes_cw, prep = models

    # No object detection or segmentation?
    if n_seg == 0:
        model = ssd
    elif n_det == 0:
        model = deeplab

    print(model.summary())
    # Bounding box utility object
    bbox_util = None if n_det == 0 else BBoxUtils(
        n_det, default_boxes_cw)

    # Load validation data
    val_ds = read_tfrecords(f"{tfrecdir}/val.tfrec", shuffle=False)
    # val_ds = read_tfrecords(f"{tfrecdir}/train.tfrec", shuffle=False)

    # Filter out empty samples - if object detection requested
    if n_det > 0:
        val_ds = val_ds.filter(filter_empty_samples)

    # Filter out missing masks - if segmentation requested
    if n_seg > 0:
        val_ds = val_ds.filter(filter_no_mask)

    # Count elements
    val_ds_size = sum(1 for _ in val_ds)
    num_batches = math.ceil(val_ds_size / batch_size)

    # Preprocess data
    val_ds_preprocessed = val_ds.map(
        preprocess(prep, (model_width, model_width), bbox_util, n_seg)
    )

    # Create batches
    val_ds_batch = val_ds_preprocessed.batch(
        batch_size=batch_size
    )

    # load model
    if load_weights:
        model.load_weights(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
    plot_model(model, to_file='infer-model.png', show_shapes=True)

    # evaluation & plots
    seg_eval = SegEval(n_seg)
    det_eval = DetEval(n_det)
    for batch in tqdm(iterable=val_ds_batch, unit='bt', total=num_batches):
        if n_seg == 0:
            img_prep, (g_cl, g_yx), img, nm = batch
            p_conf, p_locs = model(img_prep, training=False)
        elif n_det == 0:
            img_prep, g_sg, img, nm = batch
            p_segs = model(img_prep, training=False)
        else:
            img_prep, (g_cl, g_yx, g_sg), img, nm = batch
            p_conf, p_locs, p_segs = model(img_prep, training=False)
        for i in range(len(batch)):
            name = nm[i].numpy().decode('utf-8')
            if n_det > 0:
                p_cl, p_sc, p_yx = bbox_util.pred_to_boxes(
                    p_conf[i], p_locs[i])
                det_eval.evaluate_sample(
                    g_cl[i].numpy(),
                    g_yx[i].numpy(),
                    p_cl,
                    p_sc,
                    p_yx)
                file_name = f"{outdir}/orig-annotated/{name}.jpg"
                annotate_boxes(
                    img[i],
                    g_cl[i].numpy(),
                    None,
                    g_yx[i].numpy(),
                    det_names,
                    file_name)
                file_name = f"{outdir}/pred-annotated/{name}.jpg"
                annotate_boxes(img[i], p_cl, p_sc, p_yx, det_names, file_name)
                # create output file for evaluation
                p_cw = to_cw(p_yx)
                with open(f"{outdir}/pred-data/{name}.txt", "w") as f:
                    for j, cw in enumerate(p_cw):
                        yx = p_yx[j]
                        cl = p_cl[j].item()
                        sc = p_sc[j].item()
                        b_str = " ".join([str(b) for b in cw])
                        b2_str = " ".join([str(b) for b in yx])
                        f.write(f"{cl} {sc} {b_str}\n")
                        f.write(f"# yx: {cl} {sc} {b2_str}\n")
            if n_seg > 0:
                # evaluation of segmentation
                seg_eval.evaluate_sample(g_sg[i], p_segs[i])
                # annotate segmentation
                file_prefix = f"{outdir}/seg-annotated/{name}"
                annotate_segmentation(img[i], g_sg[i], p_segs[i], file_prefix)
    # runtime for inference
    print("Calculating running time ...")
    runtime = timeit.timeit(lambda: model.predict(val_ds_batch), number=1)
    # runtime for preprocessing
    print("Calculating preprocessing time ...")
    runtime_pre = timeit.timeit(lambda: sum(1 for i in val_ds), number=1)
    print(f"Inference time: {runtime/val_ds_size} sec.")
    print(f"Preprocessing time: {runtime_pre/val_ds_size} sec.")
    print(f"Network time: {(runtime-runtime_pre)/val_ds_size} sec.")
    if n_seg > 0:
        print("Segmentation metrics")
        print(f"Mean IoU: {seg_eval.mean_iou():.2%}")
        print(f"Pixel accuracy: {seg_eval.pixel_accuracy():.2%}")
        print(f"Mean accuracy: {seg_eval.mean_accuracy():.2%}")
        print(f"Mean dice coefficient: {seg_eval.mean_dice_coefficient():.2%}")
        print(f"Frequency-weighted IoU: {seg_eval.fw_iou():.2%}")
        # create miou / confusion matrix plots
        seg_eval.plot_iou(seg_names, f"{outdir}/iou-plot.png")
        seg_eval.plot_cm(seg_names, f"{outdir}/cm-plot.png")
    if n_det > 0:
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
