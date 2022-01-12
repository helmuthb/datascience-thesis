# -*- coding: utf-8 -*-

"""Collection of utility functions for bounding boxes.
This includes conversion of coordinates, calculation of IoU value,
and preparation of data for detector training.
The functions use numpy for manipulations.
"""

import numpy as np

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def to_cwh(box_xy):
    """Convert from x0/y0/x1/y1 to cx/cy/w/h.
    The first format is used in the ground truth, the
    second format is used in the neural network.
    It is optimized for batches of boxes where the first
    dimensions are used for the batches, but works for single
    boxes as well.

    Args:
        box_xy (np.ndarray [..., 4]): x0/y0/x1/y1 bounding box.
    Returns:
        box_cwh (np.ndarray [..., 4]): cx/cy/w/h bounding box.
    """
    # Added "None" to keep the dimension we are indexing
    x0 = box_xy[..., None, 0]
    y0 = box_xy[..., None, 1]
    x1 = box_xy[..., None, 2]
    y1 = box_xy[..., None, 3]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    return np.hstack([cx, cy, w, h])


def to_xy(box_cwh):
    """Convert from cx/cy/w/h to x0/y0/x1/y1.
    The first format is used in the neural network, the
    second format is used in the ground truth.

    Args:
        box_cwh (np.ndarray [..., 4]): cx/cy/w/h bounding box.
    Returns:
        box_xy (np.ndarray [..., 4]): x0/y0/x1/y1 bounding box.
    """
    cx = box_cwh[..., None, 0]
    cy = box_cwh[..., None, 1]
    w = box_cwh[..., None, 2]
    h = box_cwh[..., None, 3]
    x0 = cx - w/2
    y0 = cy - h/2
    x1 = x0 + w
    y1 = y0 + h
    return np.hstack([x0, y0, x1, y1])


def intersection_xy(b1_xy, b2_xy, pairwise=True):
    """Calculate the area of intersection between two lists of boxes.
    One can also specify just two boxes.

    Args:
        b1_xy (np.ndarray [4] or [n1, 4]): List 1 of boxes in x0/y0/x1/y1.
        b2_xy (np.ndarray [4] or [n2, 4]): List 2 of boxes in x0/y0/x1/y1.
        pairwise (boolean): Flag whether each entry of b1 corresponds
            to the entry with same index in b2 (True), or whether every
            combination of elements from b1 and b2 shall be taken (False).
    Returns:
        intersection_area (np.ndarray, [n1] or [n1, n2]): Intersection area.
    """
    # expand dimension if needed
    if b1_xy.ndim == 1:
        b1_xy = np.expand_dims(b1_xy, axis=0)
    if b2_xy.ndim == 1:
        b2_xy = np.expand_dims(b2_xy, axis=0)
    # if not pairwise: explode both arrays to the same [n1, n2, 4] shape
    if not pairwise:
        # number of elements
        n1 = b1_xy.shape[0]
        n2 = b2_xy.shape[0]
        # explode b1
        b1_xy = np.expand_dims(b1_xy, axis=1)
        b1_xy = np.tile(b1_xy, reps=(1, n2, 1))
        # explode b2_xy
        b2_xy = np.expand_dims(b2_xy, axis=0)
        b2_xy = np.tile(b2_xy, reps=(n1, 1, 1))
    # get maxima of left/bottom and minima of right/top
    xy0_i = np.maximum(b1_xy[..., :2], b2_xy[..., :2])
    xy1_i = np.minimum(b1_xy[..., 2:], b2_xy[..., 2:])
    # get lengths of width/height
    wh_i = np.maximum(0, xy1_i - xy0_i)
    # return resulting area
    return wh_i[..., 0] * wh_i[..., 1]


def iou_xy(b1_xy, b2_xy, pairwise=True):
    """Calculate the Intersection over Union between two lists of boxes.
    One can also specify just two boxes.

    Args:
        b1_xy (np.ndarray [4] or [n1, 4]): List 1 of boxes in x0/y0/x1/y1.
        b2_xy (np.ndarray [4] or [n2, 4]): List 2 of boxes in x0/y0/x1/y1.
        pairwise (boolean): Flag whether each entry of b1 corresponds to the
            entry with same index in b2 (True), or whether every
            combination of elements from b1 and b2 shall be taken (False).
    Returns:
        IoU (np.ndarray, [n1] or [n1, n2]): IoU measure.
    """
    # expand dimension if needed
    if b1_xy.ndim == 1:
        b1_xy = np.expand_dims(b1_xy, axis=0)
    if b2_xy.ndim == 1:
        b2_xy = np.expand_dims(b2_xy, axis=0)
    # calculate boxes width / height
    b1_wh = b1_xy[..., 2:] - b1_xy[..., :2]
    b2_wh = b2_xy[..., 2:] - b2_xy[..., :2]
    if pairwise:
        # calculate boxes area
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    else:
        # if not pairwise: explode both arrays to the same [n1, n2] shape
        # using None to preserve the additional axis
        b1_area = b1_wh[..., None, 0] * b1_wh[..., None, 1]
        b2_area = b2_wh[..., None, 0] * b2_wh[..., None, 1]
        # number of elements
        n1 = b1_xy.shape[0]
        n2 = b2_xy.shape[0]
        # explode b1_area
        b1_area = np.tile(b1_area, reps=(1, n2))
        # explode b2_area & swap afterwards
        b2_area = np.tile(b2_area, reps=(1, n1))
        b2_area = np.swapaxes(b2_area, 0, 1)
    # get intersection area
    i_area = intersection_xy(b1_xy, b2_xy, pairwise)
    # get union area
    u_area = b1_area + b2_area - i_area
    # return elementwise ratio
    return i_area / u_area


def boxes_classes(row):
    """Split an output row into boxes and classes.

    The boxes are defined as distortions from the corresponding default box,
    and the classes are one-hot encoded or sigmoids.

    Args:
        row (array-like(N,B,n_classes+4)): Ground truth or output from
            the model.
    Returns:
        boxes (array-like(N,B,4)): Distortions from the corresponding default.
        classes (array-like(N,B,n_classes)): Class one-hot encoded.
    """
    return row[..., :4], row[..., 4:]


class BBoxUtils(object):
    """Utility class to perform bounding box operations.

    One BBoxUtils class stored parameters used throughout the
    object detection process, including the variances, the number of
    classes, the minimum area of a box, the minimum positive and
    maximum negative IoU value, and the IoU threshold for the non-max
    suppression. Also the number of bounding boxes to be kept after
    NMS step is stored in the object.
    """

    def __init__(self, n_classes, default_boxes_cwh,
                 variances=(.1, .1, .2, .2), min_pos_iou=.05,
                 max_neg_iou=.03, min_area=0.00001, min_confidence=0.2,
                 iou_threshold=0.45, top_k=400):
        """Create BBoxUtils object.

        The parameters are used thoughout the object detection process.

        Args:
            n_classes (int): Number of distinct classes.
            variances (np.ndarray[4]): Factors to scale the distortion.
            min_area (float): Minimum area of a box.
            default_boxes_cwh (np.ndarray [n1, 4]): List of default boxes.
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
        self.default_boxes_cwh = default_boxes_cwh
        self.default_boxes_xy = to_xy(default_boxes_cwh)
        self.n_default_boxes = default_boxes_cwh.shape[0]
        self.variances = variances
        self.min_pos_iou = min_pos_iou
        self.max_neg_iou = max_neg_iou
        self.min_area = min_area
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.top_k = top_k

    def _one_box_encode_cwh(self, default_box_cwh, bbox_cwh):
        """Calculate distortion needed for adjusting default to bounding box.
        This calculates the delta in x/y of the center, and the
        delta factor in extending/shrinking the width/height of
        the default box to match the bounding box.
        The delta in x/y is divided by the width/height of the default box,
        and the natural logarithm is taken from the factors.
        In addition they are all divided by the variances for scaling.

        Args:
            default_box_cwh (np.ndarray[4]): Default box in cx/cy/w/h.
            bbox_cwh (np.ndarray[4]): Box in cx/cy/w/h.
        Returns:
            distort (np.ndarray[4]): dx/dy of the center, fw/fh of the size,
                scaled by width & variances, logarithmic for factor.
        """
        # center: subtract, ...
        c_delta = bbox_cwh[0:2]-default_box_cwh[0:2]
        # ... then scale by default box width/height
        c_delta = c_delta / default_box_cwh[2:4]
        # width/height: logarithm of quotient
        wh_delta = np.log(bbox_cwh[2:4]/default_box_cwh[2:4])
        delta = np.concatenate((c_delta, wh_delta))
        return delta / self.variances

    def _one_box_decode_cwh(self, default_box_cwh, distort):
        """Calculate bounding box back from distortion of default box.
        This calculates the bounding box coordinates back from the distortion
        needed to the default boz, by inverting the operations done in
        `_one_box_encode_cwh`.

        Args:
            default_box_cwh (np.ndarray[4]): Default box in cx/cy/w/h.
            distort (np.ndarray[4]): Distortion as used in the network.
        Returns:
            bbox_cwh (np.ndarray[4]): Bounding box in cx/cy/w/h.
        """
        # reverse adjustment of default
        delta = distort * self.variances
        d_xy = delta[0:2] * default_box_cwh[2:4]
        f_wh = np.exp(delta[2:4]) * default_box_cwh[2:4]
        # get box cwh coordinates
        box_cwh = np.concatenate((default_box_cwh[0:2]+d_xy, f_wh))
        return box_cwh

    def _one_row_gt_cwh(self, default_box_cwh, bbox_cwh, cls):
        """Get one row of ground truth for a default, bounding box, and class.

        Args:
            default_box_cwh (np.ndarray[4]): Default box in cx/cy/w/h.
            bbox_cwh (np.ndarray[4]): Bounding box in cx/cy/w/h.
            cls (int): Class of box, -1 if neutral.
        Returns:
            row_cwh (np.ndarray[n_classes+4]): Ground truth
                row for the one bounding box assigned to the one default box.
                It starts with the classes one-hot encoded and
                ends with the bounding box adjustments.
        """
        if cls == -1:
            return np.zeros((self.n_classes+4))
        return np.concatenate((
            self._one_box_encode_cwh(default_box_cwh, bbox_cwh),
            np.eye(self.n_classes)[cls],
            ))

    def boxes_decode_cwh(self, distorts):
        """Calculate bounding box back from distortion of default box.
        This calculates the box coordinates back from the distortion needed
        to the default box, by inverting the operations done in
        `_one_box_encode_cwh`.

        Args:
            distorts (np.ndarray[n, n2, 4]): Distortion as predicted.
        Returns:
            bboxes_cwh (np.ndarray[n, n2, 4]): Bounding boxes.
        """
        # reverse adjustment of default
        delta = distorts * self.variances
        d_xy = delta[..., 0:2] * self.default_boxes_cwh[..., 2:4]
        f_wh = np.exp(delta[..., 2:4]) * self.default_boxes_cwh[..., 2:4]
        # get box cwh coordinates
        bboxes_cwh = np.concatenate(
            (self.default_boxes_cwh[..., 0:2]+d_xy, f_wh),
            axis=-1)
        return bboxes_cwh

    def _skip_small_boxes_xy(self, bboxes_xy, boxes_cl):
        """Skip bounding boxes where area is smaller than min_area.

        Args:
            bboxes_xy (np.ndarray [n, 4]): List of bounding boxes.
            boxes_cl (np.ndarray [n]): List of classes per box.
        Returns:
            bboxes_xy (np.ndarray [n2, 4]): Bounding boxes above threshold.
            boxes_cl (np.ndarray [n2]): Classes, without small boxes.
        """
        # calculate area
        x0 = bboxes_xy[..., 0]
        y0 = bboxes_xy[..., 1]
        x1 = bboxes_xy[..., 2]
        y1 = bboxes_xy[..., 3]
        area = (x1-x0) * (y1-y0)
        # get filter for classes and boxes
        above = (area > self.min_area) & (boxes_cl != 0)
        above_xy = above[..., np.newaxis]
        above_xy = np.repeat(above_xy, 4, -1)
        # filtered boxes & classes
        shape_xy = (-1,) + bboxes_xy.shape[1:]
        shape_cl = (-1,) + boxes_cl.shape[1:]
        filtered_xy = np.reshape(bboxes_xy[above_xy], shape_xy)
        filtered_cl = np.reshape(boxes_cl[above], shape_cl)
        # return filtered classes and boxes
        return filtered_xy, filtered_cl

    def map_defaults_xy_orig(self, bboxes_xy, boxes_cl):
        """Find the default box for each bounding box with the most overlap.
        First each box is mapped with the default of highest match,
        such that each default is used only once.
        The remaining defaults are mapped to the box with highest IoU,
        as long as the IoU is at least min_pos_iou.
        The remaing default boxes are mapped to background if the largest IoU
        is less than max_neg_iou; otherwise they are marked as neutral.

        Args:
            boxes_xy (np.ndarray [n2, 4]): Array of object boxes corners.
            boxes_cl (np.ndarray [n2, 1]): Array of classes per box.
        Returns:
            gt (np.ndarray, [n1, n_classes+4]): The encoded ground truth,
                where each default is assigned a class (one-hot-encoded) or
                to no class for "neutral", and a distortion to adjust the
                default to the ground truth bounding box.
        """
        # only one box provided?
        if bboxes_xy.ndim == 1:
            bboxes_xy = np.expand_dims(bboxes_xy, 0)
        bboxes_xy, boxes_cl = self._skip_small_boxes_xy(bboxes_xy, boxes_cl)
        # initialize n1, n2
        n1 = self.n_default_boxes
        n2 = bboxes_xy.shape[0]
        # assignment of default to box - ground truth
        gt = np.zeros((n1, self.n_classes+4))
        # no box left after skipping small ones? return as-is
        if n2 == 0:
            return gt
        # calculate iou values for all defaults and all boxes
        iou_vals = iou_xy(self.default_boxes_xy, bboxes_xy, pairwise=False)
        # loop for each box (or defaults if less):
        for _ in range(min(n1, n2)):
            # highest default IoU per box
            max_default_boxes = np.argmax(iou_vals, axis=0)
            max_iou = iou_vals[max_default_boxes, range(n2)]
            mapped_box = np.argmax(max_iou)
            mapped_default = max_default_boxes[mapped_box]
            # add pair to list
            gt[mapped_default] = self._one_row_gt_cwh(
                self.default_boxes_cwh[mapped_default],
                to_cwh(bboxes_xy[mapped_box]),
                boxes_cl[mapped_box])
            # take default out from next rounds
            iou_vals[mapped_default, :] = 0.
        # remaining defaults: get maximum ground truth
        # highest box IoU per default
        max_bboxes = np.argmax(iou_vals, axis=1)
        max_iou = iou_vals[range(n1), max_bboxes]
        for default_box in range(n1):
            # is the maximum IoU above the positive-threshold?
            if max_iou[default_box] > self.min_pos_iou:
                gt[default_box] = self._one_row_gt_cwh(
                    self.default_boxes_cwh[default_box],
                    to_cwh(bboxes_xy[max_bboxes[default_box]]),
                    boxes_cl[max_bboxes[default_box]]
                )
            # alternatively is it above the negative-threshold?
            elif max_iou[default_box] > self.max_neg_iou:
                # then add it as "neutral"
                gt[default_box] = self._one_row_gt_cwh(
                    self.defaults_cwh[default_box],
                    to_cwh(bboxes_xy[max_bboxes[default_box]]),
                    -1
                )
        return gt

    def map_defaults_xy(self, bboxes_xy, boxes_cl):
        """Assign default boxes for provided bounding boxes.
        First each bounding box is mapped to the default box of highest match,
        such that each default box is used only once.
        The remaining default boxes are mapped to the bounding box with
        highest IoU, as long as the IoU is at least min_pos_iou.
        The remaing default boxes are mapped to background if the largest IoU
        is less than max_neg_iou; otherwise they are marked as neutral.

        Args:
            bboxes_xy (np.ndarray [n2, 4]): Array of bounding boxes corners.
            boxes_cl (np.ndarray [n2, 1]): Array of classes per box.
        Returns:
            gt (np.ndarray, [n1, n_classes+4]): The encoded ground truth,
                where each default box is assigned a class (one-hot-encoded) or
                to no class for "neutral", and a distortion to adjust the
                default box to the ground truth bounding box.
        """
        # only one box provided?
        if bboxes_xy.ndim == 1:
            bboxes_xy = np.expand_dims(bboxes_xy, 0)
        bboxes_xy, boxes_cl = self._skip_small_boxes_xy(bboxes_xy, boxes_cl)
        bboxes_cwh = to_cwh(bboxes_xy)
        # initialize n1, n2
        n1 = self.n_default_boxes
        n2 = bboxes_xy.shape[0]
        # ground truth tensor for training
        gt = np.zeros((n1, self.n_classes+4))
        # initialize as "all background" (pos 4 = class 0)
        gt[:, 4] = 1.
        # no box left after skipping small ones? return as-is
        if n2 == 0:
            return gt
        # calculate IoU values for all defaults and all boxes
        iou_vals = iou_xy(self.default_boxes_xy, bboxes_xy, pairwise=False)
        # step 1: map each box to the default with highest IoU
        main_box_default_box = np.argmax(iou_vals, axis=0)
        # removing the selected defaults (setting their IoU to 0)
        # from further mapping
        iou_vals[main_box_default_box, :] = 0.
        # add defaults to ground truth
        for bbox, default_box in enumerate(main_box_default_box):
            gt[default_box] = self._one_row_gt_cwh(
                self.default_boxes_cwh[default_box],
                bboxes_cwh[bbox],
                boxes_cl[bbox])
        # step 2: find for each default the box with maximum IoU
        aux_default_boxes = np.argmax(iou_vals, axis=1)
        # do they satisfy the thresholds?
        for default_box, bbox in enumerate(aux_default_boxes):
            if iou_vals[default_box, bbox] > self.min_pos_iou:
                # add them to ground truth
                gt[default_box] = self._one_row_gt_cwh(
                    self.default_boxes_cwh[default_box],
                    bboxes_cwh[bbox],
                    boxes_cl[bbox])
            elif iou_vals[default_box, bbox] > self.max_neg_iou:
                # add them as neutral, i.e. class = -1
                gt[default_box] = self._one_row_gt_cwh(
                    self.default_boxes_cwh[default_box],
                    bboxes_cwh[bbox],
                    -1)
        return gt

    def _nms_xy(self, boxes_xy, boxes_sc):
        """Perform non-maximum suppression of candidate boxes for one class.

        This is a greedy algorithm, starting with the highest scoring boxes,
        and then removing all remaining boxes having a too high overlap.

        Args:
            boxes_xy (np.ndarray [n, 4]): Array of candidate boxes (xy format).
            boxes_sc (np.ndarray [n]): Array of scores per box.
        Returns:
            boxes_xy (np.ndarray [n2, 4]): Remaining bounding boxes.
            boxes_sc (np.ndarray [n2]): Remaining class scores.
        """
        # number of boxes
        n = boxes_xy.shape[0]
        n_removed = 0
        n_selected = 0
        # selected indexes
        selected = np.zeros((n,), dtype=np.int8)
        # removed indexes (selected or too similar)
        removed = np.zeros((n,), dtype=np.int8)
        # loop while we have boxes to return
        while n_removed < n:
            # remaining indexes to check
            remaining = 1 - removed
            # find index of maximum score
            max_idx = np.argmax(remaining*boxes_sc)
            # corresponding confidence
            confidence = boxes_sc[max_idx]
            # is it too low?
            if confidence < self.min_confidence:
                break
            # add to selected & removed
            n_removed += 1
            n_selected += 1
            selected[max_idx] = 1
            removed[max_idx] = 1
            remaining[max_idx] = 0
            # last one, or top_k already reached?
            if n_removed >= n or n_selected >= self.top_k:
                break
            # filter similar boxes away
            overlap = iou_xy(boxes_xy, boxes_xy[max_idx, ...], pairwise=False)
            too_similar = (overlap > self.iou_threshold) * remaining
            n_found = np.sum(too_similar)
            removed = removed + too_similar
            n_removed += n_found
        # return remaining boxes & scores
        return boxes_xy[selected == 1, ...], boxes_sc[selected == 1]

    def pred_to_boxes(self, pred):
        """Convert predictions into boxes and class scores.

        Args:
            pred (np.ndarray(n, 4+n_classes)): predictions as created by
                the network.
        Returns:
            boxes_xy (np.ndarray(n2, 4)): boxes predicted from network.
            boxes_cl (np.ndarray(n2)): integer classes.
            boxes_sc (np.ndarray(n2)): float scores.
        """
        # split boxes and classes
        p_boxes_dst, p_boxes_sc = boxes_classes(pred)
        p_boxes_cwh = self.boxes_decode_cwh(p_boxes_dst)
        p_boxes_xy = to_xy(p_boxes_cwh)
        # loop through the classes
        n_classes = p_boxes_sc.shape[-1]
        # results
        first_results = True
        boxes_xy = np.empty(shape=(0, 4))
        boxes_cl = np.empty(shape=(0,), dtype=np.int8)
        boxes_sc = np.empty(shape=(0,))
        for i in range(1, n_classes):
            # perform nms
            i_boxes_xy, i_boxes_sc = self._nms_xy(
                p_boxes_xy, p_boxes_sc[..., i])
            i_n = i_boxes_xy.shape[0]
            if i_n > 0:
                # Are these the first results to add?
                if first_results:
                    boxes_xy = i_boxes_xy
                    boxes_cl = np.full(shape=(i_n,), fill_value=i)
                    boxes_sc = i_boxes_sc
                    first_results = False
                else:
                    boxes_xy = np.vstack((boxes_xy, i_boxes_xy))
                    boxes_cl = np.hstack(
                        (boxes_cl, np.full(shape=(i_n,), fill_value=i)))
                    boxes_sc = np.hstack((boxes_sc, i_boxes_sc))
        # return result
        return boxes_xy, boxes_cl, boxes_sc
