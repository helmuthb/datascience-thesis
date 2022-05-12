# -*- coding: utf-8 -*-

"""Collection of utility functions for bounding boxes.
This includes conversion of coordinates, calculation of IoU value,
and preparation of data for detector training.
The functions use numpy for manipulations.
"""

import tensorflow as tf

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2022, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def to_cw(box_yx):
    """Convert from y0/x0/y1/x1 to cy/cx/h/w.
    The first format is used in the ground truth, the
    second format is used in the neural network.
    It is optimized for batches of boxes where the first
    dimensions are used for the batches, but works for single
    boxes as well.

    Args:
        box_yx (tensor [..., 4]): y0/x0/y1/x1 bounding box.
    Returns:
        box_cw (tensor [..., 4]): cy/cx/h/w bounding box.
    """
    box_cw = tf.concat([
        (box_yx[..., :2] + box_yx[..., 2:]) / 2,
        box_yx[..., 2:] - box_yx[..., :2]
    ], axis=-1)
    return box_cw


def to_yx(box_cw):
    """Convert from cy/cx/h/w to y0/x0/y1/x1.
    The first format is used in the neural network, the
    second format is used in the ground truth.

    Args:
        box_cw (tensor [..., 4]): cy/cx/h/w bounding box.
    Returns:
        box_yx (tensor [..., 4]): y0/x0/y1/x1 bounding box.
    """
    box_yx = tf.concat([
        box_cw[..., :2] - box_cw[..., 2:] / 2,
        box_cw[..., :2] + box_cw[..., 2:] / 2
    ], axis=-1)
    return box_yx


def correct_box(box_yx, clip_to: float = None):
    """Correct a box in y0/x0/y1/x1 coordinates.
    If x0 > x1 or y0 > y1 then the box is corrected to x1=x0 or y1=y0.
    Additionally the values are clipped as specified.

    Args:
        box_yx (tensor [n1, 4]): List of boxes in y0/x1/y1/x1.
        clip_to (float): Maximum value for clipping.
    """
    box_tl = box_yx[..., :2]
    box_br = tf.maximum(box_yx[..., 2:], box_tl)
    box_yx = tf.concat([box_tl, box_br], axis=-1)
    if clip_to is None:
        return tf.maximum(box_yx, 0.)
    return tf.clip_by_value(box_yx, 0., clip_to)


def iou_yx(b1_yx, b2_yx, clip_to: float = None):
    """Calculate the Intersection over Union between two lists of boxes.
    One can also specify just two boxes.

    Args:
        b1_yx (tensor [n1, 4]): List 1 of boxes in y0/x0/y1/x1.
        b2_yx (tensor [n2, 4]): List 2 of boxes in y0/x0/y1/x1.
    Returns:
        IoU (tensor, [n1, n2]): IoU measure.
    """
    b1_yx = correct_box(b1_yx, clip_to)
    b2_yx = correct_box(b2_yx, clip_to)
    b1_yx = tf.expand_dims(b1_yx, 1)
    b2_yx = tf.expand_dims(b2_yx, 0)
    # top-left / bottom-right points of the intersection
    i_tl = tf.math.maximum(b1_yx[..., :2], b2_yx[..., :2])
    i_br = tf.math.minimum(b1_yx[..., 2:], b2_yx[..., 2:])
    # area of intersection - clipping as the intersection might be 0
    i_hw = tf.math.maximum(i_br - i_tl, 0.)
    i_ar = i_hw[..., 0] * i_hw[..., 1]
    # area of b1 & b2 boxes
    b1_hw = b1_yx[..., 2:] - b1_yx[..., :2]
    b2_hw = b2_yx[..., 2:] - b2_yx[..., :2]
    b1_ar = b1_hw[..., 0] * b1_hw[..., 1]
    b2_ar = b2_hw[..., 0] * b2_hw[..., 1]
    # intersection over union
    return i_ar / (b1_ar + b2_ar - i_ar + tf.keras.backend.epsilon())


class BBoxUtils(object):
    """Utility class to perform bounding box operations.

    One BBoxUtils class stores parameters used throughout the
    object detection process, including the variances, the number of
    classes, the minimum area of a box, the minimum positive and
    maximum negative IoU value, and the IoU threshold for the non-max
    suppression. Also the number of bounding boxes to be reviewed in
    NMS step is stored in the object.
    """
    def __init__(self, n_classes, default_boxes_cw,
                 variances=(.1, .2), min_area=0.00001,
                 min_confidence=0.3, iou_threshold=.5, top_k=200):
        """Create BBoxUtils object.

        The parameters are used thoughout the object detection process.

        Args:
            n_classes (int): Number of distinct classes.
            variances (tuple[2]): Factors to scale the distortion (horiz/vert).
            min_area (float): Minimum area of a box.
            default_boxes_cw (tensor [n1, 4]): List of default boxes.
            min_area (float): Minimum area of a ground truth box to be used.
            min_confidence (float): Minimum confidence (score) required for
                using a prediction.
            iou_threshold (float): IoU threshold value used to discard
                duplicates (at prediction time) and inferior-matching
                default boxes (at training time).
            top_k (float): Maximum number of default boxes to keep.
        """
        self.n_classes = n_classes
        self.default_boxes_cw = default_boxes_cw
        self.default_boxes_yx = to_yx(default_boxes_cw)
        self.n_default_boxes = default_boxes_cw.shape[0]
        self.variances = tf.convert_to_tensor(variances, dtype=tf.float32)
        self.min_area = min_area
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.top_k = top_k

    def _encode_cw(self, boxes_cw):
        """Calculate distortion needed for adjusting default to bounding box.
        This calculates the delta in y/x of the center, and the
        delta factor in extending/shrinking the height/width of
        the default box to match the bounding box.
        The delta in y/x is divided by the height/width of the default box,
        and the natural logarithm is taken from the factors.
        In addition they are all divided by the variances for scaling.

        Args:
            boxes_cw (tensor [..., n_defaults, 4]): Boxes in cy/cx/h/w.
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
        needed to the default box, by inverting the operations done in
        `_encode_cw`.

        Args:
            locs (tensor [..., n_defaults, 4]): Locations from network.
        Returns:
            boxes_cw (tensor [..., n_defaults, 4]): Boxes in cy/cx/h/w.
        """
        boxes = tf.concat([
            (locs[..., :2] * self.variances[0] *
                self.default_boxes_cw[:, 2:] + self.default_boxes_cw[:, :2]),
            (tf.math.exp(locs[..., 2:] * self.variances[1]) *
                self.default_boxes_cw[:, 2:])
        ], axis=-1)
        return boxes

    def map_defaults_yx(self, boxes_cl, boxes_yx):
        """Assign default boxes for provided bounding boxes.
        First each bounding box is mapped to the default box of highest match,
        such that each default box is used only once.
        The remaining default boxes are mapped to the bounding box with
        highest IoU, as long as the IoU is at least iou_threshold.
        The remaing default boxes are mapped to background.

        Args:
            boxes_cl (tensor [n1, 1]): Array of classes per box.
            boxes_yx (tensor [n1, 4]): Array of bounding boxes corners.
        Returns:
            gt_clss (tensor [n_defaults]): Ground truth classes.
            gt_locs (tensor [n_defaults, 4]): Location ground truth.
        """
        iou = iou_yx(boxes_yx, self.default_boxes_yx, clip_to=1.)
        # for each default find the best groundtruth's IoU
        default_best_gt_iou = tf.math.reduce_max(iou, axis=0)
        # for each default find the best gt index
        default_best_gt = tf.math.argmax(iou, axis=0)
        # for each gt find the best default index
        gt_best_default = tf.math.argmax(iou, axis=1)
        gt_best_default_expanded = tf.expand_dims(gt_best_default, axis=1)

        # best default for each ground truth has priority ...
        default_best_gt = tf.tensor_scatter_nd_update(
            default_best_gt,
            gt_best_default_expanded,
            tf.range(tf.shape(gt_best_default)[0], dtype=tf.int64)
        )
        # ... and is set to IoU=1 to defy minimum IoU criteria
        default_best_gt_iou = tf.tensor_scatter_nd_update(
            default_best_gt_iou,
            gt_best_default_expanded,
            tf.ones_like(gt_best_default, dtype=tf.float32)
        )

        # find box for each default and convert to distortion
        boxes_yx = tf.gather(boxes_yx, default_best_gt)
        boxes_cw = to_cw(boxes_yx)
        gt_locs = self._encode_cw(boxes_cw)

        # find class for each default
        gt_clss = tf.gather(boxes_cl, default_best_gt)

        # set class to 0 (background) if IoU even smaller
        gt_clss = tf.where(
            tf.less(default_best_gt_iou, self.iou_threshold),
            tf.zeros_like(gt_clss),
            gt_clss
        )

        return gt_clss, gt_locs

    def _nms_yx(self, boxes_sc, boxes_yx):
        """Perform non-maximum suppression of candidate boxes for one class.

        This is a greedy algorithm, starting with the highest scoring boxes,
        and then removing all remaining boxes having a too high overlap.

        Args:
            boxes_sc (tensor(n)): Array of scores per box.
            boxes_yx (tensor(n, 4)): Array of candidate boxes (yx format).
        Returns:
            selected (np.ndarray[n2]): Selected indexes.
        """
        # no scores & boxes?
        n = boxes_sc.shape[0]
        if n == 0:
            return tf.constant([], dtype=tf.int32)
        # sort boxes by score descending
        idx = tf.argsort(boxes_sc, direction='DESCENDING')
        # idx = np.argsort(-boxes_sc.numpy())
        boxes_sc = tf.gather(boxes_sc, idx)
        boxes_yx = tf.gather(boxes_yx, idx)
        # get iou values
        iou = iou_yx(boxes_yx, boxes_yx, clip_to=1.)
        # get numpy versions
        boxes_sc = boxes_sc.numpy()
        boxes_yx = boxes_yx.numpy()
        iou = iou.numpy()
        # idx = idx.numpy()
        # indexes selected so far
        selected = []
        for pos in range(min(self.top_k, n)):
            if boxes_sc[pos] < self.min_confidence:
                # go to next position
                continue
            # take next box
            selected.append(idx[pos])
            # row of overlap
            iou_row = iou[pos, :]
            # set score to zero for overlap
            boxes_sc[iou_row > self.iou_threshold] = 0.
        return tf.convert_to_tensor(selected, dtype=tf.int32)

    def pred_to_boxes(self, pr_conf, pr_locs):
        """Convert predictions into boxes and class scores.

        Args:
            pr_conf (tensor(n, n_classes)): confidence predictions.
            pr_locs (tensor(n, 4)): locations predictions.
        Returns:
            pr_cl (np.array(n2)): integer classes.
            pr_sc (np.array(n2)): float scores.
            pr_yx (np.array(n2, 4)): boxes predicted from network.
        """
        pr_cw = self._decode_cw(pr_locs)
        pr_yx = to_yx(pr_cw)
        pr_conf = tf.cast(pr_conf, tf.float32)
        pr_sc = tf.math.softmax(pr_conf, axis=-1)
        # use only non-background
        pr_cl = tf.argmax(pr_sc[..., 1:], axis=-1) + 1
        pr_max_sc = tf.reduce_max(pr_sc[..., 1:], axis=-1)

        # perform non-maximum suppression
        # idx = self._nms_yx(pr_max_sc, pr_yx)
        # """
        idx = tf.image.non_max_suppression(
            boxes=pr_yx,
            scores=pr_max_sc,
            max_output_size=self.top_k,
            iou_threshold=self.iou_threshold,
            score_threshold=self.min_confidence
        )
        # """

        pr_max_sc = tf.gather(pr_max_sc, idx)
        pr_yx = tf.gather(pr_yx, idx)
        pr_cl = tf.gather(pr_cl, idx)

        pr_max_sc = pr_max_sc.numpy()
        pr_yx = tf.clip_by_value(pr_yx, 0.0, 1.0).numpy()
        pr_cl = pr_cl.numpy()

        return pr_cl, pr_max_sc, pr_yx
