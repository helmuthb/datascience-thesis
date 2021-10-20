# -*- coding: utf-8 -*-

"""
Preprocessing of images for SSD.
"""

import tensorflow as tf
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


def subset_names(subset: List[List]) -> List[str]:
    """Get the names of the subset of classes.
    """
    return [s[0] for s in subset]


def class_indices(classes: List[str], subset: List[List]) -> List[int]:
    """Get the indices of the classes in the subset.
    """
    indices = []
    for c in classes:
        # default index for each class - 0 = background
        idx = 0
        for i, s in enumerate(subset):
            if c in s:
                idx = i
                # do not search in further subsets
                break
        # Append index to list of indices
        indices.append(idx)
    return indices


def filter_classes_bbox(classes: List[str], subset: List[List]):
    """Filter bounding boxes to be only for a specific set of classes.

    Args:
        classes (list(str)): List of classes as in the data set.
        subset (list(list)): List of list of classes.
    """
    indices = class_indices(classes, subset)
    # make sure that 0 maps to 0
    if indices[0] != 0:
        raise ValueError(f"Background '{classes[0]}' must map to 0")
    tf_indices = tf.convert_to_tensor(indices, tf.uint8)

    def _do_filter(boxes_xy, boxes_cl):
        # convert classes to subset
        boxes_cl = tf.nn.embedding_lookup(
            tf_indices,
            boxes_cl
        )
        # only return coordinates / classes not 0
        cl_mask = (boxes_cl > 0)
        cl_mask.set_shape([None])
        return boxes_xy[cl_mask], boxes_cl[cl_mask]

    def _filter_wrap(image, boxes_xy, boxes_cl, mask, name):
        ret_xy, ret_cl = _do_filter(boxes_xy, boxes_cl)
        return image, ret_xy, ret_cl, mask, name

    return _filter_wrap


def filter_classes_mask(classes: List[str], subset: List[str]):
    """Filter segmentation mask to be only for a specific set of classes.

    Args:
        classes (list(str)): List of classes as in the data set.
        subset (list(str)): List of classes to be used.
    """
    indices = class_indices(classes, subset)
    # make sure that 0 maps to 0
    if indices[0] != 0:
        raise ValueError(f"Background '{classes[0]}' must map to 0")
    # extend to 255 elements
    while len(indices) < 256:
        indices.append(0)
    tf_indices = tf.convert_to_tensor(indices, tf.uint8)

    def _do_filter(mask):
        # convert classes to subset
        mask = tf.gather(tf_indices, tf.cast(mask, tf.int32))
        return mask

    def _filter_wrap(image, boxes_xy, boxes_cl, mask, name):
        ret_mask = _do_filter(mask)
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
