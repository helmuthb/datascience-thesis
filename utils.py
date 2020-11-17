import numpy as np


def xy_to_cwh(xy_bbox):
    """Convert from x0/y0/x1/y1 to xc/yc/w/h.
    The first format is used in the ground truth, the
    second format is used in the neural network.

    Args:
        xy_bbox (np.ndarray [..., 4]): x0/y0/x1/y1 bounding box.
    Returns:
        cwh_bbox (np.ndarray [..., 4]): xc/yc/w/h bounding box.
    """
    rank = np.rank(xy_bbox)
    x0 = xy_bbox[..., 0]
    y0 = xy_bbox[..., 1]
    x1 = xy_bbox[..., 2]
    y1 = xy_bbox[..., 3]
    xc = (x0 + x1) / 2
    yc = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    return np.concatenate([xc, yc, w, h], axis=rank-1)


def cwh_to_xy(cwh_bbox):
    """Convert from xc/yc/w/h to x0/y0/x1/y1.
    The first format is used in the neural network, the
    second format is used in the ground truth.

    Args:
        cwh_bbox (np.ndarray [..., 4]): xc/yc/w/h bounding box.
    Returns:
        xy_bbox (np.ndarray [..., 4]): x0/y0/x1/y1 bounding box.
    """
    rank = np.rank(cwh_bbox)
    xc = cwh_bbox[..., 0]
    yc = cwh_bbox[..., 1]
    w = cwh_bbox[..., 2]
    h = cwh_bbox[..., 3]
    x0 = xc - w/2
    y0 = yc - h/2
    x1 = x0 + w
    y1 = y0 + h
    return np.concatenate([x0, y0, x1, y1], axis=rank-1)


def intersection_area(b1, b2, pairwise=True):
    """Calculate the area of intersection between two lists of boxes.
    One can also specify just one box.

    Args:
        b1 (np.ndarray [4] or [n1, 4]): List 1 of boxes in x0/y0/x1/y1.
        b2 (np.ndarray [4] or [n2, 4]): List 2 of boxes in x0/y0/x1/y1.
        pairwise (boolean): Flag whether each entry of b1 corresponds
        to the entry with same index in b2 (True), or whether every
        combination of elements from b1 and b2 shall be taken (False).
    Returns:
        intersection_area (np.ndarray, [n1] or [n1, n2]): Intersection area.
    """
    # expand dimension if needed
    if b1.ndim == 1:
        b1 = np.expand_dims(b1, axis=0)
    if b2.ndim == 1:
        b2 = np.expand_dims(b2, axis=0)
    # if not pairwise: explode both arrays to the same [n1, n2, 4] shape
    if not pairwise:
        # number of elements
        n1 = b1.shape[0]
        n2 = b1.shape[0]
        # explode b1
        b1 = np.expand_dims(b1, axis=1)
        b1 = np.tile(b1, reps=(1, n2, 1))
        # explode b2
        b2 = np.expand_dims(b2, axis=0)
        b2 = np.tile(b2, reps=(n1, 1, 1))
    # get maxima of left/bottom and minima of right/top
    xy0_i = np.maximum(b1[..., :2], b2[..., :2])
    xy1_i = np.minimum(b1[..., 2:], b2[..., 2:])
    # get lengths of width/height
    wh_i = np.maximum(0, xy1_i - xy0_i)
    # return resulting area
    return wh_i[..., 0] * wh_i[..., 1]


def iou_ratio(b1, b2, pairwise=True):
    """Calculate the Intersection over Union between two lists of boxes.
    One can also specify just one box.

    Args:
        b1 (np.ndarray [4] or [n1, 4]): List 1 of boxes in x0/y0/x1/y1.
        b2 (np.ndarray [4] or [n2, 4]): List 2 of boxes in x0/y0/x1/y1.
        pairwise (boolean): Flag whether each entry of b1 corresponds
        to the entry with same index in b2 (True), or whether every
        combination of elements from b1 and b2 shall be taken (False).
    Returns:
        IoU (np.ndarray, [n1] or [n1, n2]): IoU measure.
    """
    # expand dimension if needed
    if b1.ndim == 1:
        b1 = np.expand_dims(b1, axis=0)
    if b2.ndim == 1:
        b2 = np.expand_dims(b2, axis=0)
    # calculate boxes width / height
    b1_wh = b1[..., 2:] - b1[..., :2]
    b2_wh = b2[..., 2:] - b2[..., :2]
    # calculate boxes area
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 2]
    # if not pairwise: explode both arrays to the same [n1, n2, 4] shape
    if not pairwise:
        # number of elements
        n1 = b1.shape[0]
        n2 = b1.shape[0]
        # explode b1_area
        b1_area = np.expand_dims(b1_area, axis=1)
        b1_area = np.tile(b1_area, reps=(1, n2, 1))
        # explode b2_area
        b2_area = np.expand_dims(b2_area, axis=0)
        b2_area = np.tile(b2_area, reps=(n1, 1, 1))
    # get intersection area
    i_area = intersection_area(b1, b2, pairwise)
    # get union area
    u_area = b1_area + b2_area - i_area
    # return elementwise ratio
    return i_area / u_area


def iou(bbox1, bbox2):
    """Get the 'Intersection over Union' of two bounding boxes.
    The bounding boxes are expected to be in the x0/y0/x1/y1 format.

    Args:
        bbox1 (list [4]): bounding box 1.
        bbox2 (list [4]): bounding box 2.
    Returns:
        iou (float): IoU for the two boxes.
    """
    # intersection rectangle
    x0_i = max(bbox1[0], bbox2[0])
    y0_i = max(bbox1[1], bbox2[1])
    x1_i = min(bbox1[2], bbox2[2])
    y1_i = min(bbox1[3], bbox2[3])
    # area of intersection
    area_i = (x1_i-x0_i) * (y1_i-y0_i)
    # area of the two boxes
    area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    # area union - make sure > 0
    area_u = max(1, area1 + area2 - area_i)
    return area_i / area_u


