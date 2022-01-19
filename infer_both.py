import os
import argparse
import timeit

import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import cv2

from lib.preprocess import (
    preprocess, filter_classes_bbox, filter_classes_mask, subset_names)
from lib.np_bbox_utils import BBoxUtils, to_cwh
from lib.evaluate import DetEval, SegEval
from lib.deeplab import add_deeplab_features
from lib.ssdlite import (
    add_ssdlite_features, get_default_boxes_cwh,
    ssdlite_base_layers, detection_head)
from lib.tfr_utils import read_tfrecords
from lib.losses import SSDLoss, DeeplabLoss
from lib.metrics import MeanAveragePrecisionMetric, MeanIoUMetric
from lib.visualize import annotate_boxes, annotate_segmentation
import lib.rs19_classes as rs19


class CustomSchedule(LearningRateSchedule):
    """Custom schedule, based on the code in Transformer tutorial.
    https://www.tensorflow.org/text/tutorials/transformer#optimizer
    """
    def __init__(self, peak_rate, warmup_steps):
        super().__init__()
        self.peak_rate = peak_rate
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_steps ** -1.5

    def get_config(self):
        config = {}
        config['peak_rate'] = self.peak_rate
        config['warmup_steps'] = self.warmup_steps
        return config

    def __call__(self, step):
        a1 = tf.math.rsqrt(step)
        a2 = step * self.warmup_factor
        return self.peak_rate * tf.math.minimum(a1, a2)


def ssd_deeplab_model(size, n_det, n_seg):
    """
    """
    input_layer = tf.keras.layers.Input(
        shape=(size[0], size[1], 3))
    # base model
    base = MobileNetV2(
        input_tensor=input_layer,
        include_top=False)
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
    deeplab_model = tf.keras.Model(
        inputs=input_layer,
        outputs=deeplab_output
    )
    combined_model = tf.keras.Model(
        inputs=input_layer,
        outputs=[deeplab_output, ssd_output])
    # calculate default boxes
    default_boxes_cwh = get_default_boxes_cwh(l1, l2, l3, l4, l5, l6)
    # return combined model and the defaults
    return combined_model, default_boxes_cwh, base, deeplab_model, ssd_model


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
        '--all-classes',
        action='store_true',
        help='Use all classes instead of a subset'
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
    all_classes = args.all_classes
    model_width = args.model_width
    image_width = args.image_width
    image_height = args.image_height
    # create output directory if missing
    os.makedirs(f"{outdir}/orig-annotated", exist_ok=True)
    os.makedirs(f"{outdir}/pred-annotated", exist_ok=True)
    os.makedirs(f"{outdir}/pred-data", exist_ok=True)
    os.makedirs(f"{outdir}/seg-annotated", exist_ok=True)

    # number & names of classes
    if all_classes:
        n_seg = len(rs19.seg_classes)
        n_det = len(rs19.det_classes)
        seg_names = rs19.seg_classes
        det_names = rs19.det_classes
    else:
        n_seg = len(rs19.seg_subset)
        n_det = len(rs19.det_subset)
        seg_names = rs19.seg_classes
        det_names = subset_names(rs19.det_subset)

    # build model
    models = ssd_deeplab_model((model_width, model_width), n_det, n_seg)
    model, default_boxes_cwh, base, deeplab, ssd = models

    # Bounding box utility object
    bbox_util = BBoxUtils(n_det, default_boxes_cwh, min_confidence=0.01)

    # Load validation data
    val_ds_orig = read_tfrecords(
            os.path.join(args.tfrecords, 'val.tfrec'))

    # Filter for classes of interest
    if all_classes:
        val_ds_filtered = val_ds_orig
    else:
        val_ds_filtered_det = val_ds_orig.map(
            filter_classes_bbox(rs19.det_classes, rs19.det_subset)
        )
        val_ds_filtered = val_ds_filtered_det.map(
            filter_classes_mask(rs19.seg_classes, rs19.seg_subset)
        )

    # Preprocess data
    val_ds = val_ds_filtered.map(
        preprocess((model_width, model_width), bbox_util, n_seg))

    # Create batches
    val_ds_batch = val_ds.batch(batch_size=batch_size)

    # load model
    with custom_object_scope({
                'SSDLoss': SSDLoss,
                'DeeplabLoss': DeeplabLoss,
                'MeanIoUMetric': MeanIoUMetric,
                'MeanAveragePrecisionMetric': MeanAveragePrecisionMetric,
                'CustomSchedule': CustomSchedule,
            }):
        model = tf.keras.models.load_model(args.model)

    # perform inference on validation set
    preds = model.predict(val_ds_batch)
    print(preds[0].shape)  # deeplab
    print(preds[1].shape)  # ssd

    # evaluation & plots
    seg_eval = SegEval(n_seg)
    det_eval = DetEval(n_det)
    i_origs = val_ds_filtered.as_numpy_iterator()
    for s, p, o in zip(preds[0], preds[1], i_origs):
        image, boxes_xy, boxes_cl, mask, name = o
        name = name.decode('utf-8')
        p_boxes_xy, p_boxes_cl, p_boxes_sc = bbox_util.pred_to_boxes(p)
        det_eval.evaluate_sample(
            boxes_xy, boxes_cl, p_boxes_xy, p_boxes_cl, p_boxes_sc
        )
        boxes_xy = boxes_xy.copy()
        boxes_xy[:, 0] *= image_width
        boxes_xy[:, 1] *= image_height
        boxes_xy[:, 2] *= image_width
        boxes_xy[:, 3] *= image_height
        img = image.copy()
        for box_xy in boxes_xy:
            top_left = (int(round(box_xy[0])), int(round(box_xy[1])))
            bot_right = (int(round(box_xy[2])), int(round(box_xy[3])))
            color = (0, 255, 0)
            cv2.rectangle(img, top_left, bot_right, color, 2)
        cv2.imwrite(f"{outdir}/orig-annotated/{name}.jpg", img)
        file_name = f"{outdir}/pred-annotated/{name}.jpg"
        annotate_boxes(image, p_boxes_xy, p_boxes_cl, p_boxes_sc,
                       det_names, file_name)
        # create output file for evaluation
        p_boxes_cwh = to_cwh(p_boxes_xy)
        with open(f"{outdir}/pred-data/{name}.txt", "w") as f:
            for i, box_cwh in enumerate(p_boxes_cwh):
                box_xy = p_boxes_xy[i]
                cl = p_boxes_cl[i].item()
                sc = p_boxes_sc[i].item()
                b_str = " ".join([str(b) for b in box_cwh])
                b2_str = " ".join([str(b) for b in box_xy])
                f.write(f"{cl} {sc} {b_str}\n")
                f.write(f"# XY: {cl} {sc} {b2_str}\n")
        # evaluation of segmentation
        seg_eval.evaluate_sample(mask, s)
        # annotate segmentation
        file_prefix = f"{outdir}/seg-annotated/{name}"
        annotate_segmentation(image, mask, s, file_prefix)
    # runtime for inference
    runtime = timeit.timeit(lambda: model.predict(val_ds_batch), number=1)
    ds_size = 0
    for img in val_ds_batch:
        ds_size += 1
    print(f"Inference time: {runtime/ds_size} sec.")
    print(f"Mean IoU: {seg_eval.mean_iou():.2%}")
    print(f"Pixel accuracy: {seg_eval.pixel_accuracy():.2%}")
    print(f"Mean accuracy: {seg_eval.mean_accuracy():.2%}")
    print(f"Mean dice coefficient: {seg_eval.mean_dice_coefficient():.2%}")
    print(f"Frequency-weighted IoU: {seg_eval.fw_iou():.2%}")
    # create miou / confusion matrix plots
    seg_eval.plot_iou(seg_names, f"{outdir}/iou-plot.png")
    seg_eval.plot_cm(seg_names, f"{outdir}/cm-plot.png")
    # calculate mean average precision / recall
    prec_rec = det_eval.mean_average_precision_recall()
    print(f"Mean Average Precision & Recall: {prec_rec}")
    prec_rec2 = det_eval.mean_average_precision_recall(min_iou=0.1)
    print(f"Mean Average Precision & Recall (min-IoU=0.1): {prec_rec2}")
    det_eval.plot_precision_recall_curves(
        1, det_names[1], [x/10. for x in range(10)], f"{outdir}/precrecs.png"
    )
    det_eval.plot_precision(det_names, f"{outdir}/det-precision.png")
    det_eval.plot_recall(det_names, f"{outdir}/det-recall.png")


if __name__ == '__main__':
    main()
