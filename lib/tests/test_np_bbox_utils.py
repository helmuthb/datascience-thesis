#! /usr/bin/env pytest
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from ..np_bbox_utils import (boxes_decode_cw, map_defaults_xy,
                             non_maximum_suppression_xy, to_cw, to_xy,
                             intersection_xy, iou_xy, one_box_encode_cw,
                             one_box_decode_cw, one_row_gt_cw)


def test_one_to_cw():
    xy_bbox = np.array([.1, .25, .5, .75])
    cw_bbox = to_cw(xy_bbox)
    assert cw_bbox[0] == pytest.approx(0.3)
    assert cw_bbox[1] == pytest.approx(0.5)
    assert cw_bbox[2] == pytest.approx(0.4)
    assert cw_bbox[3] == pytest.approx(0.5)


def test_multiple_to_cw():
    from numpy.random import default_rng
    rng = default_rng()
    boxes = []
    boxes1 = []
    for _ in range(10):
        vals = rng.uniform(size=4)
        # make sure x0 <= x1 and y0 <= y1
        if vals[0] > vals[2]:
            vals[0], vals[2] = vals[2], vals[0]
        if vals[1] > vals[3]:
            vals[1], vals[3] = vals[3], vals[1]
        boxes.append(vals)
        # create cw box
        boxes1.append(to_cw(np.array(vals)))
    xy_boxes = np.array(boxes)
    cw_boxes = to_cw(xy_boxes)
    for i in range(10):
        assert cw_boxes[i, 0] == pytest.approx(boxes1[i][0])
        assert cw_boxes[i, 1] == pytest.approx(boxes1[i][1])
        assert cw_boxes[i, 2] == pytest.approx(boxes1[i][2])
        assert cw_boxes[i, 3] == pytest.approx(boxes1[i][3])


def test_one_to_xy():
    cw_bbox = np.array([.3, .5, .4, .5])
    xy_bbox = to_xy(cw_bbox)
    assert xy_bbox[0] == pytest.approx(0.1)
    assert xy_bbox[1] == pytest.approx(0.25)
    assert xy_bbox[2] == pytest.approx(0.5)
    assert xy_bbox[3] == pytest.approx(0.75)


def test_multiple_to_xy():
    from numpy.random import default_rng
    rng = default_rng()
    boxes = []
    boxes1 = []
    for _ in range(10):
        vals = rng.uniform(size=4)
        # make sure cx >= w/2 and cy >= h/2
        if vals[0] < vals[2]/2:
            vals[0], vals[2] = vals[2], vals[0]
        if vals[1] < vals[3]/2:
            vals[1], vals[3] = vals[3], vals[1]
        boxes.append(vals)
        # create xy box
        boxes1.append(to_xy(np.array(vals)))
    cw_boxes = np.array(boxes)
    xy_boxes = to_xy(cw_boxes)
    for i in range(10):
        assert xy_boxes[i, 0] == pytest.approx(boxes1[i][0])
        assert xy_boxes[i, 1] == pytest.approx(boxes1[i][1])
        assert xy_boxes[i, 2] == pytest.approx(boxes1[i][2])
        assert xy_boxes[i, 3] == pytest.approx(boxes1[i][3])


def test_intersection_areas_pairwise():
    b1 = np.array([[0., 0., 1., 1.], [0., 0., 1., .5]])
    b2 = np.array([[0., 0., 1., 1.], [0., 0., .5, 1.]])
    areas = intersection_xy(b1, b2, pairwise=True)
    assert areas.shape == (2,)
    assert areas[0] == pytest.approx(1.)
    assert areas[1] == pytest.approx(.25)


def test_intersection_areas_2x2():
    b1 = np.array([[0., 0., 1., 1.], [0., 0., 1., .5]])
    b2 = np.array([[0., 0., 1., 1.], [0., 0., .3, 1.]])
    areas = intersection_xy(b1, b2, pairwise=False)
    assert areas.shape == (2, 2)
    assert areas[0, 0] == pytest.approx(1.)
    assert areas[0, 1] == pytest.approx(.3)
    assert areas[1, 0] == pytest.approx(.5)
    assert areas[1, 1] == pytest.approx(.15)


def test_intersection_areas_1x2():
    b1 = np.array([[0., 0., 1., .5]])
    b2 = np.array([[0., 0., 1., 1.], [0., 0., .3, 1.]])
    areas = intersection_xy(b1, b2, pairwise=False)
    assert areas.shape == (1, 2)
    assert areas[0, 0] == pytest.approx(.5)
    assert areas[0, 1] == pytest.approx(.15)


def test_intersection_areas_2x1():
    b1 = np.array([[0., 0., 1., 1.], [0., 0., 1., .5]])
    b2 = np.array([[0., 0., .3, 1.]])
    areas = intersection_xy(b1, b2, pairwise=False)
    assert areas.shape == (2, 1)
    assert areas[0, 0] == pytest.approx(.3)
    assert areas[1, 0] == pytest.approx(.15)


def test_iou():
    b1 = np.array([0., 0., 1., .5])
    b2 = np.array([0., 0., 1., 1.])
    iou_val = iou_xy(b1, b2)
    assert iou_val == pytest.approx(.5)


def test_iou_pairwise():
    b1 = np.array([[0., 0., 1., 1.], [0., 0., 1., .5]])
    b2 = np.array([[0., 0., 1., 1.], [0., 0., .5, 1.]])
    iou_vals = iou_xy(b1, b2, pairwise=True)
    assert iou_vals.shape == (2,)
    assert iou_vals[0] == pytest.approx(1.)
    assert iou_vals[1] == pytest.approx(1/3)


def test_iou_2x2():
    b1 = np.array([[0., 0., 1., 1.], [0., 0., 1., .5]])
    b2 = np.array([[0., 0., 1., 1.], [0., 0., .2, 1.]])
    iou_vals = iou_xy(b1, b2, pairwise=False)
    assert iou_vals.shape == (2, 2)
    assert iou_vals[0, 0] == pytest.approx(1.)
    assert iou_vals[0, 1] == pytest.approx(.2)
    assert iou_vals[1, 0] == pytest.approx(.5)
    assert iou_vals[1, 1] == pytest.approx(1/6)


def test_iou_1x1():
    b1 = np.array([[0., 0., 1., .5]])
    b2 = np.array([[0., 0., 1., 1.]])
    iou_vals = iou_xy(b1, b2, pairwise=False)
    assert iou_vals.shape == (1, 1)
    assert iou_vals[0, 0] == pytest.approx(.5)


def test_iou_1x2():
    b1 = np.array([[0., 0., 1., .5]])
    b2 = np.array([[0., 0., 1., 1.], [0., 0., .2, 1.]])
    iou_vals = iou_xy(b1, b2, pairwise=False)
    assert iou_vals.shape == (1, 2)
    assert iou_vals[0, 0] == pytest.approx(.5)
    assert iou_vals[0, 1] == pytest.approx(1/6)


def test_iou_2x1():
    b1 = np.array([[0., 0., 1., 1.], [0., 0., .2, 1.]])
    b2 = np.array([[0., 0., 1., .5]])
    areas = iou_xy(b1, b2, pairwise=False)
    assert areas.shape == (2, 1)
    assert areas[0, 0] == pytest.approx(.5)
    assert areas[1, 0] == pytest.approx(1/6)


def test_adjust():
    defaults_xy = np.array([
        [.15, .15, .45, .45],  # default left-top
        [.35, .35, .65, .65],  # default in the center
        [.55, .55, .85, .85],  # default right-bottom
    ])
    boxes_xy = np.array([
        [.15, .15, .45, .45],  # box left-top
        [.45, .45, .55, .55],  # box in the center
        [.45, .45, .75, .75],  # box right-bottom
    ])
    # calculate cw boxes
    defaults_cw = to_cw(defaults_xy)
    boxes_cw = to_cw(boxes_xy)
    # calculate all distortions
    dist = [one_box_encode_cw(a, b) for a, b in zip(defaults_cw, boxes_cw)]
    # calculate all (hopefully) original boxes
    b2_cw = np.array(
        [one_box_decode_cw(a, d) for a, d in zip(defaults_cw, dist)])
    b2b_cw = boxes_decode_cw(defaults_cw, np.array(dist))
    # 1st case - identical boxes: distortion = 0
    assert np.all(dist[0] == 0.)
    # 2nd case - box a bit smaller than default
    # center not moved
    assert dist[1][0] == 0.
    assert dist[1][1] == 0.
    # scaling < 0
    assert dist[1][2] < 0.
    assert dist[1][3] < 0.
    # 3rd case - box same size but moved
    # center moved negatively
    assert dist[2][0] < 0.
    assert dist[2][1] < 0.
    # scaling == 0
    assert dist[2][2] == pytest.approx(0.)
    assert dist[2][3] == pytest.approx(0.)
    # check whether boxes were reconstructed
    assert np.all(boxes_cw == pytest.approx(b2_cw))
    assert np.all(boxes_cw == pytest.approx(b2b_cw))


def test_one_row():
    defaults_xy = np.array([
        [.15, .15, .45, .45],  # default left-top
        [.35, .35, .65, .65],  # default in the center
        [.55, .55, .85, .85],  # default right-bottom
    ])
    boxes_xy = np.array([
        [.15, .15, .45, .45],  # box left-top
        [.45, .45, .55, .55],  # box in the center
        [.45, .45, .75, .75],  # box right-bottom
    ])
    defaults_cw = to_cw(defaults_xy)
    boxes_cw = to_cw(boxes_xy)
    clses = [-1, 0, 1]
    n_classes = 10
    # calculate all rows
    rows = [one_row_gt_cw(a, b, c, n_classes)
            for a, b, c in zip(defaults_cw, boxes_cw, clses)]
    # all rows have shape (n_classes+4,)
    for r in rows:
        assert r.shape == (n_classes+4,)
    # class -1 and identical box / default should go to all zero
    print(rows[0])
    assert np.all(rows[0] == pytest.approx(0.))
    # class 0 should mean element [0] to be 1
    assert rows[1][0] == pytest.approx(1.)
    # class 1 should mean element [0] to be 0, and [1] to 1
    assert rows[2][0] == pytest.approx(0.)
    assert rows[2][1] == pytest.approx(1.)


def test_map_defaults():
    defaults_xy = np.array([
        [.15, .15, .45, .45],  # default left-top
        [.55, .15, .85, .45],  # default right-top
        [.35, .35, .65, .65],  # default in the center
        [.15, .55, .45, .85],  # default left-bottom
        [.55, .55, .85, .85],  # default right-bottom
    ])
    boxes_xy = np.array([
        [.15, .15, .45, .45],  # box left-top
        [.45, .45, .55, .55],  # box in the center
        [.45, .45, .75, .75],  # box right-bottom
    ])
    clses = np.array([1, 2, 3])
    n_classes = 10
    # get defaults
    rows = map_defaults_xy(defaults_xy, boxes_xy, clses, n_classes)
    # assert shape (5, n_classes+4)
    assert rows.shape == (5, n_classes+4)
    # assert first row to be class 1
    assert rows[0, 0] == pytest.approx(0.)
    assert rows[0, 1] == pytest.approx(1.)
    assert np.all(rows[0, 2:n_classes] == pytest.approx(0.))


def test_nms():
    boxes_xy = np.array([
        [.15, .15, .45, .45],  # box left-top
        [.45, .45, .55, .55],  # box in the center
        [.45, .45, .75, .75],  # box right-bottom
    ])
    scores = np.array([.2, .5, .3])
    b2, s2 = non_maximum_suppression_xy(boxes_xy, scores, iou_threshold=.5)
    assert b2.shape == (1, 4)
    assert b2.shape[0] == s2.shape[0]
    assert s2[0] == .5
    assert np.all(b2[0, ...] == np.array([.45, .45, .55, .55]))
