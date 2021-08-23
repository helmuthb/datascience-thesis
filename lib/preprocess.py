# -*- coding: utf-8 -*-

"""
Preprocessing of images for SSD.
"""

import tensorflow as tf

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


def preprocess_det(size: tuple[int], bbox_util: BBoxUtils):
    """Get preprocessing function for object detection.

    The resulting function will expect image, and optionally
    bboxes and classes, and will return the same after resizing and scaling.
    Optionally it can also be augmented.
    In addition, if the ground truth is provided, it will also be filtered
    and the bounding box data will be prepared for comparison or training.

    Args:
        size (tuple(int)): Output size of the images.
        bbox_util (BBoxUtils): Bounding box utility object.
    """

    def _preprocess(image, boxes_xy, boxes_cl):
        """Preprocess image: resize, scale, filter small boxes.

        Args:
            image (tf.Tensor): Image data.
            boxes_xy (tf.Tensor): Bounding boxes per object.
            boxes_cl (tf.Tensor): Classes per object.
        """
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # to numpy for boxes and classes
        boxes_xy = boxes_xy.numpy()
        boxes_cl = boxes_cl.numpy()
        # map anchors to boxes
        gt = bbox_util.map_anchors_xy(boxes_xy, boxes_cl)
        # return preprocessed image & data
        return image, tf.convert_to_tensor(gt, dtype=tf.float32)

    def _preprocess_wrap(image, boxes_xy, boxes_cl, mask):
        return tf.py_function(
            _preprocess,
            (image, boxes_xy, boxes_cl),
            (tf.float32, tf.float32)
        )

    return _preprocess_wrap


def preprocess_seg(size, num_classes):
    """
    """
    def _preprocess(image, mask):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # resize mask
        mask = tf.image.resize(mask, size, method='nearest')
        # reshape - get rid of last dimension
        mask = tf.reshape(mask, shape=size)
        # restrict mask to values from 0 to num_classes
        mask = tf.clip_by_value(mask, 0, num_classes-1)
        mask = tf.cast(mask, tf.uint8)
        return image, mask

    def _preprocess_wrap(image, boxes_xy, boxes_cl, mask):
        return tf.py_function(
            _preprocess,
            (image, mask),
            (tf.float32, tf.uint8)
        )

    return _preprocess_wrap


def preprocess(size: tuple[int], bbox_utils: BBoxUtils, n_seg: int):
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
        # map anchors to boxes
        gt = bbox_utils.map_anchors_xy(boxes_xy, boxes_cl)
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
