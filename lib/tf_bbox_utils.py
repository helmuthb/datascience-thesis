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
    # top-left / bottom-right points of the intersection
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
        self.variances = tf.convert_to_tensor(variances, dtype=tf.float32)
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
                self.variances[1]),
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
        max_iou_default = tf.math.reduce_max(iou, axis=0)
        # for each default find the best gt index
        max_gt_idx_default = tf.math.argmax(iou, axis=0)
        # for each gt find the best default index
        max_default_gt_idx = tf.math.argmax(iou, axis=1)
        max_default_gt_idx_2 = tf.expand_dims(max_default_gt_idx, axis=1)

        # best default for each ground truth has priority ...
        max_gt_idx_default = tf.tensor_scatter_nd_update(
            max_gt_idx_default,
            max_default_gt_idx_2,
            tf.range(tf.shape(max_default_gt_idx)[0], dtype=tf.int64)
        )
        # ... and is set to IoU=1 to defy minimum IoU criteria
        max_iou_default = tf.tensor_scatter_nd_update(
            max_iou_default,
            max_default_gt_idx_2,
            tf.ones_like(max_default_gt_idx, dtype=tf.float32)
        )

        # find class for each default
        gt_clss = tf.gather(boxes_cl, max_gt_idx_default)

        # find box for each default and convert to distortion
        boxes_xy = tf.gather(boxes_xy, max_gt_idx_default)
        boxes_cw = to_cw(boxes_xy)
        gt_locs = self._encode_cw(boxes_cw)

        # set class to 0 (background) if IoU too small
        # not sure what this shall do ...
        below_threshold = tf.less(max_iou_default, self.min_pos_iou)
        gt_clss = tf.where(
            below_threshold,
            tf.zeros_like(gt_clss),
            gt_clss
        )

        # set class to -1 (neutral) if IoU very small
        neutral_threshold = tf.less(max_iou_default, self.max_neg_iou)
        gt_clss = tf.where(
            neutral_threshold,
            -tf.ones_like(gt_clss),
            gt_clss
        )

        return gt_clss, gt_locs

    def map_defaults_xy_np(self, boxes_cl, bboxes_xy):
        """Assign default boxes for provided bounding boxes.
        First each bounding box is mapped to the default box of highest match,
        such that each default box is used only once.
        The remaining default boxes are mapped to the bounding box with
        highest IoU, as long as the IoU is at least min_pos_iou.
        The remaing default boxes are mapped to background if the largest IoU
        is less than max_neg_iou; otherwise they are marked as neutral.

        Args:
            boxes_cl (np.ndarray [n2, 1]): Array of classes per box.
            bboxes_xy (np.ndarray [n2, 4]): Array of bounding boxes corners.
        Returns:
            gt_clss (np.ndarray [n1]): Ground truth classes.
            gt_locs (np.ndarray [n1, 4]): Location ground truth.
        """
        # only one box provided?
        if tf.rank(bboxes_xy) == 1:
            bboxes_xy = tf.expand_dims(bboxes_xy, 0)
        bboxes_cw = to_cw(bboxes_xy)
        # initialize n1, n2
        n1 = self.n_default_boxes
        # ground truth tensor for training
        gt_clss = tf.zeros((n1,), dtype=tf.int32)
        gt_cw = tf.zeros((n1, 4), dtype=tf.float32)
        # no box left after skipping small ones? return as-is
        if bboxes_xy.shape[0] == 0:
            return gt_clss, gt_cw
        # calculate IoU values for all defaults and all boxes
        iou_vals = iou_xy(self.default_boxes_xy, bboxes_xy)
        # step 1: map each box to the default with highest IoU
        main_box_default_box = tf.argmax(iou_vals, axis=0)
        # removing the selected defaults (setting their IoU to 0)
        # from further mapping
        iou_vals = tf.tensor_scatter_nd_update(
            iou_vals,
            main_box_default_box,
            tf.zeros_like(iou_vals)
        )
        # add defaults to ground truth
        bbox = tf.range(tf.shape(main_box_default_box)[0], dtype=tf.int64)

        for bbox, default_box in enumerate(main_box_default_box):
            cl = boxes_cl[bbox]
            gt_clss = tf.tensor_scatter_nd_update(
                gt_clss,
                default_box,
                cl
            )
            gt_cw = tf.tensor_scatter_nd_update(
                gt_cw,
                default_box,
                bboxes_cw
            )
        # step 2: find for each default the box with maximum IoU
        aux_default_boxes = tf.argmax(iou_vals, axis=1)
        # do they satisfy the thresholds?
        for default_box, bbox in enumerate(aux_default_boxes):
            if iou_vals[default_box, bbox] > self.min_pos_iou:
                # add them to ground truth
                cl = boxes_cl[bbox]
                gt_clss = tf.tensor_scatter_nd_update(
                    gt_clss,
                    default_box,
                    cl
                )
                gt_cw = tf.tensor_scatter_nd_update(
                    gt_cw,
                    default_box,
                    bboxes_cw
                )
            elif iou_vals[default_box, bbox] > self.max_neg_iou:
                # add them as neutral
                gt_clss = tf.tensor_scatter_nd_update(
                    gt_clss,
                    default_box,
                    0
                )
                gt_cw = tf.tensor_scatter_nd_update(
                    gt_cw,
                    default_box,
                    bboxes_cw
                )
        gt_locs = self._encode_cw(gt_cw)
        return gt_clss, gt_locs

    def _nms_xy(self, boxes_sc, boxes_xy):
        """Perform non-maximum suppression of candidate boxes for one class.

        This is a greedy algorithm, starting with the highest scoring boxes,
        and then removing all remaining boxes having a too high overlap.

        Args:
            boxes_sc (tensor(n)): Array of scores per box.
            boxes_xy (tensor(n, 4)): Array of candidate boxes (xy format).
        Returns:
            selected (np.ndarray[n2]): Selected indexes.
        """
        # no scores & boxes?
        n = boxes_sc.shape[0]
        if n == 0:
            return (np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32))
        # sort boxes by score descending
        idx = tf.argsort(boxes_sc, direction='DESCENDING')
        boxes_sc = tf.gather(boxes_sc, idx)
        boxes_xy = tf.gather(boxes_xy, idx)
        # get iou values
        iou = iou_xy(boxes_xy, boxes_xy)
        # get numpy versions
        boxes_sc = boxes_sc.numpy()
        boxes_xy = boxes_xy.numpy()
        iou = iou.numpy()
        idx = idx.numpy()
        # indexes selected so far
        selected = []
        for pos in range(n):
            if boxes_sc[pos] < self.min_confidence:
                # go to next position
                continue
            # take next box
            selected.append(idx[pos])
            # row of overlap
            iou_row = iou[pos, :]
            # set score to zero for overlap
            boxes_sc[iou_row > self.iou_threshold] = 0.
            # enough boxes?
            if len(selected) >= self.top_k:
                break
        return selected

    def _nms_xy_org(self, boxes_sc, boxes_xy):
        """Perform non-maximum suppression of candidate boxes for one class.

        This is a greedy algorithm, starting with the highest scoring boxes,
        and then removing all remaining boxes having a too high overlap.

        Args:
            boxes_sc (tensor(n)): Array of scores per box.
            boxes_xy (tensor(n, 4)): Array of candidate boxes (xy format).
        Returns:
            boxes_sc (tensor(n2)): Remaining class scores.
            boxes_xy (tensor(n2, 4)): Remaining bounding boxes.
        """
        if boxes_xy.shape[0] == 0:
            return tf.constant([], dtype=tf.int32)
        selected = [0]
        top_k = tf.math.minimum(self.top_k, boxes_xy.shape[0])
        _, idx = tf.math.top_k(boxes_sc, top_k)
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
            selected.append(tf.argmax(next_indices).numpy())

        return tf.gather(idx, selected)

    def pred_to_boxes(self, pr_conf, pr_locs):
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
        pr_boxes_cw = self._decode_cw(pr_locs)
        pr_boxes_xy = to_xy(pr_boxes_cw)

        boxes_cl = []
        boxes_sc = []
        boxes_xy = []

        for c in range(1, self.n_classes):
            cls_scores = pr_conf[:, c]

            # filter for minimum confidence
            score_idx = cls_scores > self.min_confidence
            cls_xy = pr_boxes_xy[score_idx]
            cls_sc = cls_scores[score_idx]

            nms_idx = self._nms_xy(cls_sc, cls_xy)
            cls_sc = tf.gather(cls_sc, nms_idx)
            cls_xy = tf.gather(cls_xy, nms_idx)
            cls_cl = [c] * cls_xy.shape[0]

            boxes_cl.extend(cls_cl)
            boxes_sc.append(cls_sc)
            boxes_xy.append(cls_xy)

        boxes_cl = np.array(boxes_cl)
        boxes_sc = tf.concat(boxes_sc, axis=0)
        boxes_xy = tf.concat(boxes_xy, axis=0)

        boxes_xy = tf.clip_by_value(boxes_xy, 0.0, 1.0).numpy()
        boxes_sc = boxes_sc.numpy()

        return boxes_cl, boxes_sc, boxes_xy
