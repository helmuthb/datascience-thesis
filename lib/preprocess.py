# -*- coding: utf-8 -*-

"""
Preprocessing of images for SSD.
"""

import tensorflow as tf
from typing import Tuple, List

from .np_bbox_utils import BBoxUtils as BBoxUtilsNp
from .tf_bbox_utils import BBoxUtils as BBoxUtilsTf

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


def filter_empty_samples(image, boxes_cl, boxes_xy, mask, name):
    return tf.math.greater(tf.shape(boxes_cl)[0], tf.constant(0))


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

    def _do_filter(boxes_cl, boxes_xy):
        # convert classes to subset
        boxes_cl = tf.nn.embedding_lookup(
            tf_indices,
            boxes_cl
        )
        # only return coordinates / classes not 0
        cl_mask = (boxes_cl > 0)
        cl_mask.set_shape([None])
        return boxes_cl[cl_mask], boxes_xy[cl_mask]

    def _filter_wrap(image, boxes_cl, boxes_xy, mask, name):
        ret_cl, ret_xy = _do_filter(boxes_cl, boxes_xy)
        return image, ret_cl, ret_xy, mask, name

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

    def _filter_wrap(image, boxes_cl, boxes_xy, mask, name):
        ret_mask = _do_filter(mask)
        return image, boxes_cl, boxes_xy, ret_mask, name

    return _filter_wrap


def preprocess_np(size: Tuple[int], bbox_utils: BBoxUtilsNp, n_seg: int):
    """Preprocess image: resize, scale, filter small boxes, drop name.

    Args:
        size (tuple(int)): Target image size.
        bbox_util (BBoxUtils): Bounding box utility class - None for no
            object detection.
        n_seg (int): Number of classes used for segmentation - 0 for no
            image segmentation.
"""

    def _preprocess_ssd_np(image, boxes_cl, boxes_xy, mask, name):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # to numpy for classes and boxes
        boxes_cl = boxes_cl.numpy()
        boxes_xy = boxes_xy.numpy()
        # map defaults to boxes
        gt_clss, gt_locs = bbox_utils.map_defaults_xy(boxes_cl, boxes_xy)
        # ground truth as tensor
        gt_clss = tf.convert_to_tensor(gt_clss, dtype=tf.int64)
        gt_locs = tf.convert_to_tensor(gt_locs, dtype=tf.float32)
        # return preprocessed image & data
        return image, gt_clss, gt_locs

    def _preprocess_deeplab(image, boxes_cl, boxes_xy, mask, name):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # resize mask
        mask = tf.image.resize(mask, size, method='nearest')
        # reshape - get rid of last dimension
        mask = tf.reshape(mask, shape=size)
        # restrict mask to values from 0 to num_classes
        mask = tf.clip_by_value(mask, 0, n_seg-1)
        mask = tf.cast(mask, tf.uint8)
        # return preprocessed image & data
        return image, mask

    def _preprocess_both_np(image, boxes_cl, boxes_xy, mask, name):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # to numpy for classes and boxes
        boxes_cl = boxes_cl.numpy()
        boxes_xy = boxes_xy.numpy()
        # map defaults to boxes
        gt_clss, gt_locs = bbox_utils.map_defaults_xy(boxes_cl, boxes_xy)
        # ground truth as tensor
        gt_clss = tf.convert_to_tensor(gt_clss, dtype=tf.int64)
        gt_locs = tf.convert_to_tensor(gt_locs, dtype=tf.float32)
        # resize mask
        mask = tf.image.resize(mask, size, method='nearest')
        # reshape - get rid of last dimension
        mask = tf.reshape(mask, shape=size)
        # restrict mask to values from 0 to num_classes
        mask = tf.clip_by_value(mask, 0, n_seg-1)
        mask = tf.cast(mask, tf.uint8)
        # return preprocessed image & data
        return image, gt_clss, gt_locs, mask

    def _preprocess_ssd(image, boxes_cl, boxes_xy, mask, name):
        image, gt_clss, gt_locs = tf.py_function(
            _preprocess_ssd_np,
            (image, boxes_cl, boxes_xy, mask, name),
            (tf.float32, tf.int64, tf.float32)
        )
        image.set_shape((size[0], size[1], 3))
        return image, (gt_clss, gt_locs)

    def _preprocess_both(image, boxes_cl, boxes_xy, mask, name):
        image, gt_clss, gt_locs, mask = tf.py_function(
            _preprocess_both_np,
            (image, boxes_cl, boxes_xy, mask, name),
            (tf.float32, tf.int64, tf.float32, tf.uint8)
        )
        image.set_shape((size[0], size[1], 3))
        mask.set_shape((size[0], size[1]))
        return image, (gt_clss, gt_locs, mask)

    if bbox_utils is None:
        return _preprocess_deeplab
    elif n_seg == 0:
        return _preprocess_ssd
    else:
        return _preprocess_both


def preprocess_tf(size: Tuple[int], bbox_utils: BBoxUtilsTf, n_seg: int):
    """Preprocess image: resize, scale, filter small boxes, drop name.

    Args:
        size (tuple(int)): Target image size.
        bbox_util (BBoxUtils): Bounding box utility class - None for no
            object detection.
        n_seg (int): Number of classes used for segmentation - 0 for no
            image segmentation.
"""

    def _preprocess_ssd(image, boxes_cl, boxes_xy, mask, name):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # map defaults to boxes
        gt_clss, gt_locs = bbox_utils.map_defaults_xy(boxes_cl, boxes_xy)
        # return preprocessed image & data
        return image, (gt_clss, gt_locs)

    def _preprocess_deeplab(image, boxes_cl, boxes_xy, mask, name):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # resize mask
        mask = tf.image.resize(mask, size, method='nearest')
        # reshape - get rid of last dimension
        mask = tf.reshape(mask, shape=size)
        # restrict mask to values from 0 to num_classes
        mask = tf.clip_by_value(mask, 0, n_seg-1)
        mask = tf.cast(mask, tf.uint8)
        # return preprocessed image & data
        return image, mask

    def _preprocess_both(image, boxes_cl, boxes_xy, mask, name):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # map defaults to boxes
        gt_clss, gt_locs = bbox_utils.map_defaults_xy(boxes_cl, boxes_xy)
        # resize mask
        mask = tf.image.resize(mask, size, method='nearest')
        # reshape - get rid of last dimension
        mask = tf.reshape(mask, shape=size)
        # restrict mask to values from 0 to num_classes
        mask = tf.clip_by_value(mask, 0, n_seg-1)
        mask = tf.cast(mask, tf.uint8)
        # return preprocessed image & data
        return image, (gt_clss, gt_locs, mask)

    if bbox_utils is None:
        return _preprocess_deeplab
    elif n_seg == 0:
        return _preprocess_ssd
    else:
        return _preprocess_both
