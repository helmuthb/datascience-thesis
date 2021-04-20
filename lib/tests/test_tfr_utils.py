#! /usr/bin/env pytest
# -*- coding: utf-8 -*-

import os
import json
import pytest

import tensorflow as tf
import numpy as np

from ..tfr_utils import parse_json, wrap_in_list, int64_feature
from ..tfr_utils import float_feature, bytes_feature, load_file
from ..tfr_utils import BBox, img_to_example, img_from_example
from ..tfr_utils import read_tfrecords, write_tfrecords


def test_wrap_in_list():
    int_item = 12
    int_item_wrapped = [12]
    w1 = wrap_in_list(int_item)
    assert isinstance(w1, list)
    assert isinstance(w1[0], int)
    assert w1[0] == int_item
    w2 = wrap_in_list(int_item_wrapped)
    assert isinstance(w2, list)
    assert isinstance(w2[0], int)
    assert w2[0] == int_item_wrapped[0]


def test_int64_feature():
    int_item = 12
    int_item_wrapped = [12]
    f1 = int64_feature(int_item)
    assert isinstance(f1, tf.train.Feature)
    f2 = int64_feature(int_item_wrapped)
    assert isinstance(f2, tf.train.Feature)


def test_float_feature():
    int_item = 12.5
    int_item_wrapped = [12.5]
    f1 = float_feature(int_item)
    assert isinstance(f1, tf.train.Feature)
    f2 = float_feature(int_item_wrapped)
    assert isinstance(f2, tf.train.Feature)


def test_bytes_feature():
    str_item = "hello"
    str_item_wrapped = ["hello", "world"]
    f1 = bytes_feature(str_item)
    assert isinstance(f1, tf.train.Feature)
    f2 = bytes_feature(str_item_wrapped)
    assert isinstance(f2, tf.train.Feature)


def test_load_file(tmp_path):
    f = tmp_path / "dummy.txt"
    content = "hello world"
    f.write_text(content)
    assert load_file(f.as_posix()) == content.encode()


def test_img_example():
    dir = os.path.dirname(os.path.realpath(__file__))
    f_jpg = dir + "/test_data/rs00079.jpg"
    f_png = dir + "/test_data/rs00079.png"
    metadata = {
        'height': 36,
        'width': 64,
        'name': 'rs00079',
        'format': 'jpeg'
    }
    boxes = [
        BBox(1, 'car', 0, 17, 29, 36),
        BBox(1, 'car', 40, 15, 64, 36)
    ]
    e = img_to_example(f_jpg, metadata, boxes, f_png)
    assert isinstance(e, bytes)
    # convert back ...
    i, b, c, p = img_from_example(e)
    assert isinstance(i, tf.Tensor)
    assert isinstance(b, tf.Tensor)
    assert isinstance(c, tf.Tensor)
    assert isinstance(p, tf.Tensor)
    c = c.numpy()
    assert np.all(c == np.array([1, 1]))
    b = b.numpy()
    assert np.all(b == np.array([[0, 17, 29, 36], [40, 15, 64, 36]]))
    assert i.shape == (36, 64, 3)
    assert p.shape == (36, 64, 1)


def test_tfrecords(tmp_path):
    dir = os.path.dirname(os.path.realpath(__file__))
    f_jpg = dir + "/test_data/rs00079.jpg"
    f_png = dir + "/test_data/rs00079.png"
    metadata = {
        'height': 36,
        'width': 64,
        'name': 'rs00079',
        'format': 'jpeg'
    }
    boxes = [
        BBox(1, 'car', 0, 17, 29, 36),
        BBox(1, 'car', 40, 15, 64, 36)
    ]
    img1 = {
        'image_path': f_jpg,
        'metadata': metadata,
        'objects': boxes,
        'mask': f_png
    }
    outfile1 = tmp_path / "test-out1"
    write_tfrecords([img1], outfile1.as_posix(), 1, True)
    outfile2 = tmp_path / "test-out2"
    write_tfrecords([img1], outfile2.as_posix(), 1, False)
    # read records from first write
    ds1 = read_tfrecords(outfile1)
    for elem in ds1.take(1):
        img, boxes, cls, mask = elem
        assert img.shape == (36, 64, 3)
        assert boxes.shape == (2, 4)
        assert cls.shape == (2,)
        assert mask.shape == (36, 64, 1)
    ds2 = read_tfrecords(outfile2)
    for elem in ds2.take(1):
        img, boxes, cls, mask = elem
        assert img.shape == (36, 64, 3)
        assert boxes.shape == (2, 4)
        assert cls.shape == (2,)
        assert mask.shape == (36, 64, 1)


def test_parse_json():
    json_str = """
    {
    "frame": "rs00000",
    "imgHeight": 1080,
    "imgWidth": 1920,
    "objects": [
        {
            "boundingbox": [
                1066,
                571,
                1077,
                590
            ],
            "label": "track-sign-front"
        },
        {
            "boundingbox": [
                1024,
                599,
                1052,
                612
            ],
            "label": "switch-unknown"
        }
        ]
    }
    """
    json_obj = json.loads(json_str)
    classes = ["background", "track-sign-front", "switch-unknown"]
    metadata, objects = parse_json(json_obj, classes)
    assert metadata["format"] == "jpg"
    assert metadata["width"] == 1920
    assert metadata["height"] == 1080
    assert metadata["name"] == "rs00000"
    assert len(objects) == 2
    assert isinstance(objects[0], BBox)
    assert objects[0].cl == 1
    assert objects[0].lb == "track-sign-front"
    assert objects[0].x0 == pytest.approx(1066/1920)
    assert objects[0].x1 == pytest.approx(1077/1920)
    assert objects[0].y0 == pytest.approx(571/1080)
    assert objects[0].y1 == pytest.approx(590/1080)
    assert isinstance(objects[1], BBox)
    assert objects[1].cl == 2
    assert objects[1].lb == "switch-unknown"
    assert objects[1].x0 == pytest.approx(1024/1920)
    assert objects[1].x1 == pytest.approx(1052/1920)
    assert objects[1].y0 == pytest.approx(599/1080)
    assert objects[1].y1 == pytest.approx(612/1080)
