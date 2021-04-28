# -*- coding: utf-8 -*-

"""
Preprocessing of images for SSD.
"""

import tensorflow as tf

from .np_bbox_utils import map_anchors_xy, to_xy

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def preprocess_det(size, tf_anchors_cwh, n_classes):
    """Get preprocessing function for object detection.

    The resulting function will expect image, and optionally
    bboxes and classes, and will return the same after resizing and scaling.
    Optionally it can also be augmented.
    In addition, if the ground truth is provided, it will also be filtered
    and the bounding box data will be prepared for comparison or training.

    Args:
        size (tuple(int)): Output size of the images.
        tf_anchors_cwh (tf.Tensor [n_anchor, 4]): Anchor boxes in cx/cy/w/h.
        n_classes (int): Number of classes.
    """
    anchors_cwh = tf_anchors_cwh.numpy()
    anchors_xy = to_xy(anchors_cwh)

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
        gt = map_anchors_xy(anchors_xy, boxes_xy, boxes_cl, n_classes)
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


def preprocess(size, tf_anchors_cwh, n_seg, n_det):
    """
    """
    anchors_cwh = tf_anchors_cwh.numpy()
    anchors_xy = to_xy(anchors_cwh)

    def _preprocess(image, boxes_xy, boxes_cl, mask):
        # resize image
        image = tf.image.resize(image, size, antialias=True)
        # scale image color values to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        # to numpy for boxes and classes
        boxes_xy = boxes_xy.numpy()
        boxes_cl = boxes_cl.numpy()
        # map anchors to boxes
        gt = map_anchors_xy(anchors_xy, boxes_xy, boxes_cl, n_det)
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

    def _preprocess_wrap(image, boxes_xy, boxes_cl, mask):
        image, mask, gt = tf.py_function(
            _preprocess,
            (image, boxes_xy, boxes_cl, mask),
            (tf.float32, tf.uint8, tf.float32)
        )
        return image, (mask, gt)

    return _preprocess_wrap
