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
        pairwise (boolean): Flag whether each entry of b1 corresponds
        to the entry with same index in b2 (True), or whether every
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


def one_box_encode_cwh(anchor_cwh, box_cwh, variances=(.1, .1, .2, .2)):
    """Calculate distortion needed for adjusting anchor to box.
    This calculates the delta in x/y of the center, and the
    delta factor in extending/shrinking the width/height of
    the anchor to match the box.
    The delta in x/y is divided by the width/height of the anchor,
    and the natural logarithm is taken from the factors.
    In addition they are all divided by the variances for scaling.

    Args:
        anchor_cwh (np.ndarray[4]): Anchor in cx/cy/w/h.
        box_cwh (np.ndarray[4]): Box in cx/cy/w/h.
        variances (np.ndarray[4]): Factors to scale the distortion.
    Returns:
        distortion (np.ndarray[4]): dx/dy of the center, fw/fh of the size,
            scaled by width & variances, logarithmic for factor.
    """
    delta = np.concatenate(
        (box_cwh[0:2]-anchor_cwh[0:2],
         np.log(box_cwh[2:4]/anchor_cwh[2:4])))
    return delta / variances


def one_box_decode_cwh(anchor_cwh, distort, variances=(.1, .1, .2, .2)):
    """Calculate box back from distortion of anchor.
    This calculates the box coordinates back from the distortion needed
    to the anchor, by inverting the operations done in anchor_adjust.

    Args:
        anchor_cwh (np.ndarray[4]): Anchor in cx/cy/w/h.
        distort (np.ndarray[4]): Distortion as used in the network.
        variances (np.ndarray[4]): Factors to scale the distortion.
    Returns:
        box_cwh (np.ndarray[4]): Box in cx/cy/w/h.
    """
    # reverse adjustment of anchor
    delta = distort * variances
    d_xy = delta[0:2]
    f_wh = np.exp(delta[2:4])
    # get box cwh coordinates
    box_cwh = np.concatenate((anchor_cwh[0:2]+d_xy, anchor_cwh[2:4]*f_wh))
    return box_cwh


def boxes_decode_cwh(anchors_cwh, distorts, variances=(.1, .1, .2, .2)):
    """Calculate box back from distortion of anchor.
    This calculates the box coordinates back from the distortion needed
    to the anchor, by inverting the operations done in anchor_adjust.

    Args:
        anchors_cwh (np.ndarray[n2, 4]): Anchor in cx/cy/w/h.
        distorts (np.ndarray[n, n2, 4]): Distortion as predicted.
        variances (np.ndarray[4]): Factors to scale the distortion.
    Returns:
        boxes_cwh (np.ndarray[n, n2, 4]): Boxes in cx/cy/w/h.
    """
    # reverse adjustment of anchor
    delta = distorts * variances
    d_xy = delta[..., 0:2]
    f_wh = np.exp(delta[..., 2:4])
    # get box cwh coordinates
    boxes_cwh = np.concatenate(
        (anchors_cwh[..., 0:2]+d_xy, anchors_cwh[..., 2:4]*f_wh),
        axis=-1)
    return boxes_cwh


def one_row_gt_cwh(anchor_cwh, box_cwh, cls, n_classes):
    """Get one row of ground truth for an anchor, a box and a class.

    Args:
        anchor_cwh (np.ndarray[4]): Anchor in cx/cy/w/h.
        box_cwh (np.ndarray[4]): Box in cx/cy/w/h.
        cls (int): Class of box, -1 if neutral.
        n_classes (int): Number of distinct classes.
    Returns:
        row_cwh (np.ndarray[n_classes+4]): Ground truth
            row for the one box assigned to the one anchor.
            It starts with the classes one-hot encoded and
            ends with the bounding box adjustments.
    """
    if cls == -1:
        return np.zeros((n_classes+4))
    return np.concatenate(
        (np.eye(n_classes)[cls],
         one_box_encode_cwh(anchor_cwh, box_cwh)))


def boxes_classes(row):
    """Split an output row into boxes and classes.

    The boxes are defined as distortions from the corresponding anchor box,
    and the classes are one-hot encoded or sigmoids.

    Args:
        row (array-like(N,B,n_classes+4)): Ground truth or output from
            the model.
    Returns:
        boxes (array-like(N,B,4)): Distortions from the corresponding anchor.
        classes (array-like(N,B,n_classes)): Class one-hot encoded.
    """
    return row[..., :4], row[..., 4:]


def skip_small_boxes_xy(boxes_xy, boxes_cl, min_area):
    """Skip boxes where area is smaller than min_area.

    Args:
        boxes_xy (np.ndarray [n, 4]): List of object boxes in x0/y0/x1/y1.
        boxes_cl (np.ndarray [n]): List of classes per box.
        min_area (float): Minimum area of a box.
    Returns:
        boxes_xy (np.ndarray [n2, 4]): Bounding boxes above threshold.
        boxes_cl (np.ndarray [n2]): Classes, without small boxes.
    """
    # calculate area
    x0 = boxes_xy[..., 0]
    y0 = boxes_xy[..., 1]
    x1 = boxes_xy[..., 2]
    y1 = boxes_xy[..., 3]
    area = (x1-x0) * (y1-y0)
    # get filter for classes and boxes
    above = area > min_area
    above_xy = above[..., np.newaxis]
    above_xy = np.repeat(above_xy, 4, -1)
    # filtered boxes & classes
    shape_xy = (-1,) + boxes_xy.shape[1:]
    shape_cl = (-1,) + boxes_cl.shape[1:]
    filtered_xy = np.reshape(boxes_xy[above_xy], shape_xy)
    filtered_cl = np.reshape(boxes_cl[above], shape_cl)
    # return filtered classes and boxes
    return filtered_xy, filtered_cl


def map_anchors_xy(anchors_xy, boxes_xy, boxes_cl, n_classes,
                   min_pos_iou=.5, max_neg_iou=.3, min_area=0.001):
    """Find the anchor for each box with the most overlap.
    First each box is mapped with the anchor of highest match,
    such that each anchor is used only once.
    The remaining anchors are mapped to the box with highest IoU,
    as long as the IoU is at least min_pos_iou.
    The remaing anchors are mapped to background if the largest IoU is less
    than max_neg_iou; otherwise they are marked as neutral.

    Args:
        anchors_xy (np.ndarray [n1, 4]): List of anchor boxes in x0/y0/x1/y1.
        boxes_xy (np.ndarray [n2, 4]): List of object boxes in x0/y0/x1/y1.
        boxes_cl (np.ndarray [n2, 1]): List of classes per box.
        n_classes (int): Number of classes.
        min_pos_iou (float): Minimum IoU value for mapping non-optimal anchors.
        max_neg_iou (float): Maximum IoU value for mapping to background.
        min_area (float): Minimum area of a box.
    Returns:
        gt (np.ndarray, [n1, n_classes+4]): The encoded ground truth,
            where each anchor is assigned a class (one-hot-encoded) or no class
            for "neutral", and a distortion (cx,cy,cw,ch) to adjust the anchor
            to the ground truth bounding box.
    """
    # only one box provided?
    if boxes_xy.ndim == 1:
        boxes_xy = np.expand_dims(boxes_xy, 0)
    boxes_xy, boxes_cl = skip_small_boxes_xy(boxes_xy, boxes_cl, min_area)
    # initialize n1, n2
    n1 = anchors_xy.shape[0]
    n2 = boxes_xy.shape[0]
    # assignment of anchor to box - ground truth
    gt = np.zeros((n1, n_classes+4))
    # no box larger than minimum? return as-is
    if boxes_cl.shape[0] == 0:
        return gt
    # calculate iou values for all anchors and all boxes
    iou_vals = iou_xy(anchors_xy, boxes_xy, pairwise=False)
    # loop for each box (or anchor if less):
    for _ in range(min(n1, n2)):
        # highest anchor IoU per box
        max_anchors = np.argmax(iou_vals, axis=0)
        max_iou = iou_vals[max_anchors, range(n2)]
        mapped_box = np.argmax(max_iou)
        mapped_anchor = max_anchors[mapped_box]
        # add pair to list
        gt[mapped_anchor] = one_row_gt_cwh(
            to_cwh(anchors_xy[mapped_anchor]),
            to_cwh(boxes_xy[mapped_box]),
            boxes_cl[mapped_box],
            n_classes)
        # take anchor out from next rounds
        iou_vals[mapped_anchor, :] = 0.
    # remaining anchors: get maximum ground truth
    # highest box IoU per anchor
    max_boxes = np.argmax(iou_vals, axis=1)
    max_iou = iou_vals[range(n1), max_boxes]
    for anchor in range(n1):
        # is the maximum IoU above the positive-threshold?
        if max_iou[anchor] > min_pos_iou:
            gt[anchor] = one_row_gt_cwh(
                to_cwh(anchors_xy[anchor]),
                to_cwh(boxes_xy[max_boxes[anchor]]),
                boxes_cl[max_boxes[anchor]],
                n_classes)
        # alternatively is it above the negative-threshold?
        elif max_iou[anchor] > max_neg_iou:
            # then add it as "neutral"
            gt[anchor] = one_row_gt_cwh(
                to_cwh(anchors_xy[anchor]),
                to_cwh(boxes_xy[max_boxes[anchor]]),
                -1,
                n_classes
            )
    return gt


def non_maximum_suppression_xy(boxes_xy, boxes_sc, iou_threshold=0.45):
    """Perform non-maximum suppression of candidate boxes.

    This is a greedy algorithm, which searches the highest scoring box
    available and removes then all remaining boxes which have an overlap
    higher than given with the iou_threshold.

    Args:
        boxes_xy (np.ndarray [n, 4]): List of candidate boxes in x0/y0/x1/y1.
        boxes_sc (np.ndarray [n]): List of scores per box.
        iou_threshold (float): Maximum allowed box overlap.
    Returns:
        boxes_xy (np.ndarray [n2, 4]): Remaining bounding boxes.
        boxes_sc (np.ndarray [n2]): Remaining class scores.
    """
    # number of boxes
    n = boxes_xy.shape[0]
    n_removed = 0
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
        # add to selected & removed
        n_removed += 1
        selected[max_idx] = 1
        removed[max_idx] = 1
        remaining[max_idx] = 0
        # last one?
        if n_removed == n:
            break
        # filter similar boxes away
        overlap = iou_xy(boxes_xy, boxes_xy[max_idx, ...], pairwise=False)
        too_similar = (overlap > iou_threshold) * remaining
        n_found = np.sum(too_similar)
        removed = removed + too_similar
        n_removed += n_found
    # return remaining boxes & scores
    return boxes_xy[selected == 1, ...], boxes_sc[selected == 1]


def pred_to_boxes(pred, anchors_cwh, min_confidence=0.2, iou_threshold=0.45):
    """Convert predictions into boxes and classs scores.

    Args:
        pred (np.ndarray(n, 4+n_classes)): predictions as created by
            the network.
        anchors_cwh (np.ndarray(n, 4)): anchor boxes.
        min_confidence (float): only use predictions larger than this.
        iou_threshold (float): TODO threshold used for non-maximum suppression.
    Returns:
        boxes_xy (np.ndarray(n2, 4)): boxes predicted from network.
        boxes_cl (np.ndarray(n2)): integer classes.
        boxes_sc (np.ndarray(n2)): float scores.
    """
    # split boxes and classes
    p_boxes_dst, p_boxes_sc = boxes_classes(pred)
    p_boxes_cwh = boxes_decode_cwh(anchors_cwh, p_boxes_dst)
    p_boxes_xy = to_xy(p_boxes_cwh)
    # loop through the classes
    n_classes = p_boxes_sc.shape[-1]
    # results
    empty = True
    boxes_xy = np.empty(shape=(0, 4))
    boxes_cl = np.empty(shape=(0,), dtype=np.int8)
    boxes_sc = np.empty(shape=(0,))
    for i in range(1, n_classes):
        # filter for minimum confidence
        f_conf = p_boxes_sc[..., i] >= min_confidence
        # perform nms
        i_boxes_xy, i_boxes_sc = non_maximum_suppression_xy(
            p_boxes_xy[f_conf], p_boxes_sc[f_conf, i])
        i_n = i_boxes_xy.shape[0]
        if i_n > 0:
            if empty:
                boxes_xy = i_boxes_xy
                boxes_cl = np.full(shape=(i_n,), fill_value=i)
                boxes_sc = i_boxes_sc
            else:
                boxes_xy = np.vstack((boxes_xy, i_boxes_xy))
                boxes_cl = np.vstack(
                    (boxes_cl, np.full(shape=(i_n,), fill_value=i)))
                boxes_sc = np.vstack((boxes_sc, i_boxes_sc))
    # convert to numpy arrays
    return boxes_xy, boxes_cl, boxes_sc
