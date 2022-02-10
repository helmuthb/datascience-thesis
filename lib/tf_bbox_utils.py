# -*- coding: utf-8 -*-

"""Collection of utility functions for bounding boxes.
This includes conversion of coordinates, calculation of IoU value,
and preparation of data for detector training.
The functions use numpy for manipulations.
"""

import tensorflow as tf
import numpy as np

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2022, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def to_cw(box_xy):
    """Convert from x0/y0/x1/y1 to cx/cy/w/h.
    The first format is used in the ground truth, the
    second format is used in the neural network.
    It is optimized for batches of boxes where the first
    dimensions are used for the batches, but works for single
    boxes as well.

    Args:
        box_xy (tensor [..., 4]): x0/y0/x1/y1 bounding box.
    Returns:
        box_cw (tensor [..., 4]): cx/cy/w/h bounding box.
    """
    box_cw = tf.concat([
        (box_xy[..., :2] + box_xy[..., 2:]) / 2,
        box_xy[..., 2:] - box_xy[..., :2]
    ], axis=-1)
    return box_cw


def to_xy(box_cw):
    """Convert from cx/cy/w/h to x0/y0/x1/y1.
    The first format is used in the neural network, the
    second format is used in the ground truth.

    Args:
        box_cw (tensor [..., 4]): cx/cy/w/h bounding box.
    Returns:
        box_xy (tensor [..., 4]): x0/y0/x1/y1 bounding box.
    """
    box_xy = tf.concat([
        box_cw[..., :2] - box_cw[..., 2:] / 2,
        box_cw[..., :2] + box_cw[..., 2:] / 2
    ], axis=-1)
    return box_xy


def iou_xy(b1_xy, b2_xy):
    """Calculate the Intersection over Union between two lists of boxes.
    One can also specify just two boxes.

    Args:
        b1_xy (tensor [n1, 4]): List 1 of boxes in x0/y0/x1/y1.
        b2_xy (tensor [n2, 4]): List 2 of boxes in x0/y0/x1/y1.
    Returns:
        IoU (tensor, [n1, n2]): IoU measure.
    """
    b1_xy = tf.expand_dims(b1_xy, 1)
    b2_xy = tf.expand_dims(b2_xy, 0)
    i_tl = tf.math.maximum(b1_xy[..., :2], b2_xy[..., :2])
    i_br = tf.math.minimum(b1_xy[..., 2:], b2_xy[..., 2:])
    # area of intersection
    i_hw = i_tl - i_br
    i_ar = i_hw[..., 0] * i_hw[..., 1]
    # area of b1 & b2 boxes
    b1_hw = b1_xy[..., :2] - b1_xy[..., 2:]
    b2_hw = b2_xy[..., :2] - b2_xy[..., 2:]
    b1_ar = b1_hw[..., 0] * b1_hw[..., 1]
    b2_ar = b2_hw[..., 0] * b2_hw[..., 1]
    # union area
    u_ar = b1_ar + b2_ar - i_ar
    return i_ar / u_ar


class BBoxUtils(object):
    """Utility class to perform bounding box operations.

    One BBoxUtils class stores parameters used throughout the
    object detection process, including the variances, the number of
    classes, the minimum area of a box, the minimum positive and
    maximum negative IoU value, and the IoU threshold for the non-max
    suppression. Also the number of bounding boxes to be kept after
    NMS step is stored in the object.
    """
    def __init__(self, n_classes, default_boxes_cw,
                 variances=(.1, .2), min_pos_iou=.05,
                 max_neg_iou=.03, min_area=0.00001, min_confidence=0.2,
                 iou_threshold=0.45, top_k=400):
        """Create BBoxUtils object.

        The parameters are used thoughout the object detection process.

        Args:
            n_classes (int): Number of distinct classes.
            variances (tuple[2]): Factors to scale the distortion (horiz/vert).
            min_area (float): Minimum area of a box.
            default_boxes_cw (tensor [n1, 4]): List of default boxes.
            min_pos_iou (float): Minimum IoU value for mapping default boxes
                not assigned as optimal.
            max_neg_iou (float): Maximum IoU value for mapping to background.
            min_area (float): Minimum area of a ground truth box to be used.
            min_confidence (float): Minimum confidence (score) required for
                using a prediction.
            iou_threshold (float): Maximum IoU value for not discarding
                default boxes similar to the one with highest confidence.
            top_k (float): Maximum number of default boxes to keep.
        """
        self.n_classes = n_classes
        self.default_boxes_cw = default_boxes_cw
        self.default_boxes_xy = to_xy(default_boxes_cw)
        self.n_default_boxes = default_boxes_cw.shape[0]
        self.variances = variances
        self.min_pos_iou = min_pos_iou
        self.max_neg_iou = max_neg_iou
        self.min_area = min_area
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.top_k = top_k

    def _encode_cw(self, boxes_cw):
        """Calculate distortion needed for adjusting default to bounding box.
        This calculates the delta in x/y of the center, and the
        delta factor in extending/shrinking the width/height of
        the default box to match the bounding box.
        The delta in x/y is divided by the width/height of the default box,
        and the natural logarithm is taken from the factors.
        In addition they are all divided by the variances for scaling.

        Args:
            boxes_cw (tensor [..., n_defaults, 4]): Boxes in cx/cy/w/h.
        Returns:
            locs (tensor [..., n_defaults, 4]): Locations for network.
        """
        locs = tf.concat([
            ((boxes_cw[..., :2] - self.default_boxes_cw[:, :2]) /
                (self.default_boxes_cw[:, 2:] * self.variances[0])),
            (tf.math.log(boxes_cw[..., 2:] / self.default_boxes_cw[:, 2:]) /
                self.variances[1])
        ], axis=-1)
        return locs

    def _decode_cw(self, locs):
        """Calculate bounding box back from distortion of default box.
        This calculates the bounding box coordinates back from the distortion
        needed to the default boz, by inverting the operations done in
        `_encode_cw`.

        Args:
            locs (tensor [..., n_defaults, 4]): Locations from network.
        Returns:
            boxes_cw (tensor [..., n_defaults, 4]): Boxes in cx/cy/w/h.
        """
        boxes = tf.concat([
            (locs[..., :2] * self.variances[0] *
                self.default_boxes_cw[:, 2:] + self.default_boxes_cw[:, :2]),
            (tf.math.exp(locs[..., 2:] * self.variances[1]) *
                self.default_boxes_cw[:, 2:])
        ], axis=-1)
        return boxes

    def map_defaults_xy(self, boxes_cl, boxes_xy):
        """Assign default boxes for provided bounding boxes.
        First each bounding box is mapped to the default box of highest match,
        such that each default box is used only once.
        The remaining default boxes are mapped to the bounding box with
        highest IoU, as long as the IoU is at least min_pos_iou.
        The remaing default boxes are mapped to background if the largest IoU
        is less than max_neg_iou; otherwise they are marked as neutral.

        Args:
            boxes_cl (tensor [n1, 1]): Array of classes per box.
            boxes_xy (tensor [n1, 4]): Array of bounding boxes corners.
        Returns:
            gt_clss (tensor [n_defaults]): Ground truth classes.
            gt_locs (tensor [n_defaults, 4]): Location ground truth.
        """
        iou = iou_xy(boxes_xy, self.default_boxes_xy)
        # for each default find the best IoU
        best_gt_iou = tf.math.reduce_max(iou, 0)
        # for each default find the best gt
        best_gt_idx = tf.math.argmax(iou, 0)
        # for each gt find the best default
        best_default_idx = tf.math.argmax(iou, 1)

        # best default for each ground truth has preference ...
        best_gt_idx = tf.tensor_scatter_nd_update(
            best_gt_idx,
            tf.expand_dims(best_default_idx, 1),
            tf.range(tf.shape(best_default_idx)[0], dtype=tf.int64)
        )
        # ... and is set to IoU=1 to defy minimum IoU criteria
        best_gt_iou = tf.tensor_scatter_nd_update(
            best_gt_iou,
            tf.expand_dims(best_default_idx, 1),
            tf.ones_like(best_default_idx, dtype=tf.float32)
        )

        # find class for each default ...
        gt_clss = tf.gather(boxes_cl, best_gt_idx)
        # ... and set to 0 (background) if IoU too small
        gt_clss = tf.where(
            tf.less(best_gt_iou, self.iou_threshold),
            tf.zeros_like(gt_clss),
            gt_clss
        )

        # find box for each default and convert to distortion
        boxes_xy = tf.gather(boxes_xy, best_gt_idx)
        boxes_cw = to_cw(boxes_xy)
        gt_locs = self._encode_cw(boxes_cw)

        return gt_clss, gt_locs

    def _nms_xy(self, boxes_sc, boxes_xy):
        """Perform non-maximum suppression of candidate boxes for one class.

        This is a greedy algorithm, starting with the highest scoring boxes,
        and then removing all remaining boxes having a too high overlap.

        Args:
            boxes_sc (np.ndarray [n]): Array of scores per box.
            boxes_xy (np.ndarray [n, 4]): Array of candidate boxes (xy format).
        Returns:
            boxes_sc (np.ndarray [n2]): Remaining class scores.
            boxes_xy (np.ndarray [n2, 4]): Remaining bounding boxes.
        """
        if boxes_xy.shape[0] == 0:
            return tf.constant([], dtype=tf.int32)
        selected = [0]
        idx = tf.argsort(boxes_sc, direction='DESCENDING')
        idx = idx[:self.top_k]
        boxes_xy = tf.gather(boxes_xy, idx)

        iou = iou_xy(boxes_xy, boxes_xy)

        while True:
            row = iou[selected[-1]]
            next_indices = (row <= self.min_confidence)
            iou = tf.where(
                tf.expand_dims(tf.math.logical_not(next_indices), 0),
                tf.ones_like(iou, dtype=tf.float32),
                iou
            )
            if not tf.math.reduce_any(next_indices):
                break
            selected.append(tf.argsort(
                tf.dtypes.cast(next_indices, tf.int32),
                direction='DESCENDING')[0].numpy())

        return tf.gather(idx, selected)

    def pred_to_boxes_np(self, pr_conf, pr_locs):
        """Convert predictions into boxes and class scores.

        Args:
            pr_conf (tensor(n, n_classes)): confidence predictions.
            pr_locs (tensor(n, 4)): locations predictions.
        Returns:
            boxes_cl (np.array(n2)): integer classes.
            boxes_sc (np.array(n2)): float scores.
            boxes_xy (np.array(n2, 4)): boxes predicted from network.
        """
        pr_conf = tf.math.softmax(pr_conf, axis=-1)
        pr_boxes = self._decode_cw(pr_locs)

        boxes_cl = []
        boxes_sc = []
        boxes_xy = []

        for c in range(1, self.n_classes):
            cls_scores = pr_conf[:, c]

            score_idx = cls_scores > 0.6
            cls_boxes = pr_boxes[score_idx]
            cls_scores = cls_scores[score_idx]

            nms_idx = self._nms_xy(cls_scores, cls_boxes)
            cls_boxes = tf.gather(cls_boxes, nms_idx)
            cls_scores = tf.gather(cls_scores, nms_idx)
            cls_labels = [c] * cls_boxes.shape[0]

            boxes_cl.extend(cls_labels)
            boxes_sc.append(cls_scores)
            boxes_xy.append(cls_boxes)

        boxes_cl = np.array(boxes_cl)
        boxes_sc = tf.concat(boxes_sc, axis=0)
        boxes_xy = tf.concat(boxes_xy, axis=0)

        boxes_xy = tf.clip_by_value(boxes_xy, 0.0, 1.0).numpy()
        boxes_sc = boxes_sc.numpy()

        return boxes_cl, boxes_sc, boxes_xy
