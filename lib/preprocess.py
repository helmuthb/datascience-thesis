# -*- coding: utf-8 -*-

"""
Preprocessing of images for SSD.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List

from .np_bbox_utils import BBoxUtils

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def filter_classes_bbox(classes: List[str], subset: List[str]):
    """Filter bounding boxes to be only for a specific set of classes.

    Args:
        classes (list(str)): List of classes as in the data set.
        subset (list(str)): List of classes to be used.
    """
    # make sure that subset[0] == classes[0]
    if classes[0] != subset[0]:
        raise ValueError(f"Background '{classes[0]}' must be first in subset")
    # get indices of subclasses used
    indices = [classes.index(c) for c in subset]

    def _do_filter(boxes_xy, boxes_cl):
        # to numpy for boxes and classes
        boxes_xy = boxes_xy.numpy()
        boxes_cl = boxes_cl.numpy()
        # empty set of results
        ret_xy = []
        ret_cl = []
        # find classes in the list of indices
        for i, c in enumerate(boxes_cl):
            if c in indices:
                ret_xy.append(boxes_xy[i])
                ret_cl.append(indices.index(c))
        # is the list empty?
        if len(ret_cl) == 0:
            ret_xy.append([0., 0., 1., 1.])
            ret_cl.append(0)
        return np.array(ret_xy), np.array(ret_cl)

    def _filter_wrap(image, boxes_xy, boxes_cl, mask, name):
        ret_xy, ret_cl = tf.py_function(
            _do_filter,
            (boxes_xy, boxes_cl),
            (tf.float32, tf.uint8)
        )
        return image, ret_xy, ret_cl, mask, name

    return _filter_wrap


def filter_classes_mask(classes: List[str], subset: List[str]):
    """Filter segmentation mask to be only for a specific set of classes.

    Args:
        classes (list(str)): List of classes as in the data set.
        subset (list(str)): List of classes to be used.
    """
    # make sure that subset[0] == classes[0]
    if classes[0] != subset[0]:
        raise ValueError(f"Background '{classes[0]}' must be first in subset")
    # get indices of subclasses used
    indices = [classes.index(c) for c in subset]
    # create mapping list
    translate = []
    for c in classes:
        if c in indices:
            translate.append(indices.index(c))
        else:
            # treat as background
            translate.append(0)

    def _do_filter(mask):
        # to numpy for mask
        mask = mask.numpy()
        # vectorized function for mapping
        max_i = len(translate)
        np_translate = np.vectorize(lambda i: translate[i] if i < max_i else 0)
        # apply on mask
        return np_translate(mask)

    def _filter_wrap(image, boxes_xy, boxes_cl, mask, name):
        ret_mask = tf.py_function(
            _do_filter,
            (mask,),
            (tf.uint8,)
        )
        return image, boxes_xy, boxes_cl, ret_mask, name

    return _filter_wrap


def preprocess(size: Tuple[int], bbox_utils: BBoxUtils, n_seg: int):
    """Preprocess image: resize, scale, filter small boxes.

    Args:
        size (tuple(int)): Target image size.
        bbox_util (BBoxUtils): Bounding box utility class.
        n_seg (int): Number of classes used for segmentation.
"""

    def _preprocess(image, boxes_xy, boxes_cl, mask):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # to numpy for boxes and classes
        boxes_xy = boxes_xy.numpy()
        boxes_cl = boxes_cl.numpy()
        # map defaults to boxes
        gt = bbox_utils.map_defaults_xy(boxes_xy, boxes_cl)
        # ground truth as tensor
        gt = tf.convert_to_tensor(gt, dtype=tf.float32)
        # resize mask
        mask = tf.image.resize(mask, size, method='nearest')
        # reshape - get rid of last dimension
        mask = tf.reshape(mask, shape=size)
        # restrict mask to values from 0 to num_classes
        mask = tf.clip_by_value(mask, 0, n_seg-1)
        mask = tf.cast(mask, tf.uint8)
        # return preprocessed image & data
        return image, mask, gt

    def _preprocess_wrap(image, boxes_xy, boxes_cl, mask, name):
        image, mask, gt = tf.py_function(
            _preprocess,
            (image, boxes_xy, boxes_cl, mask),
            (tf.float32, tf.uint8, tf.float32)
        )
        return image, (mask, gt)

    return _preprocess_wrap
