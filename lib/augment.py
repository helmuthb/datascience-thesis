# -*- coding: utf-8 -*-

"""
Augmentation of images for training.
"""

import tensorflow as tf
import numpy as np
from albumentations import (
    Rotate, RandomBrightnessContrast, JpegCompression, HueSaturationValue,
    HorizontalFlip, CropAndPad, RandomResizedCrop, BboxParams, Compose
)

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


class Augment():
    """Augment image: resize, scale, flip, adjust contrast.
    """
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.transforms = []
        self.t = None

    def rotate(self, limit=40):
        self.transforms.append(Rotate(limit=limit))

    def random_brightness_contrast(self, b_limit=0.1, c_limit=0.1, p=0.5):
        self.transforms.append(RandomBrightnessContrast(
            brightness_limit=b_limit,
            contrast_limit=c_limit,
            p=p))

    def jpeg_compression(self, quality_lower=85, quality_upper=100, p=0.5):
        self.transforms.append(JpegCompression(
            quality_lower=quality_lower,
            quality_upper=quality_upper,
            p=p))

    def hsv(self, hue=20, sat=30, val=20, p=0.5):
        self.transforms.append(HueSaturationValue(
            hue_shift_limit=hue,
            sat_shift_limit=sat,
            val_shift_limit=val,
            p=p))

    def horizontal_flip(self, p=0.5):
        self.transforms.append(HorizontalFlip(p=p))

    def crop_and_pad(self, percent=0.1, p=0.5):
        self.transforms.append(CropAndPad(percent=percent, p=p))

    def random_resized_crop(self, scale=(0.08, 1), ratio=(0.75, 1.33), p=0.5):
        self.transforms.append(RandomResizedCrop(scale, ratio, p))

    def __call__(self, image, boxes_cl, boxes_xy, mask, name):
        def _tuple_float(t):
            return [float(i) for i in t]

        if self.t is None:
            self.t = Compose(
                self.transforms,
                bbox_params=BboxParams(format='albumentations'))
        t = self.t
        boxes = [list(b) + [str(c)] for c, b in zip(boxes_cl, boxes_xy)]
        out = t(
            image=image,
            bboxes=boxes,
            mask=mask
        )
        # print(boxes_xy)
        bout_cl = [int(b[4]) for b in out['bboxes']]
        bout_xy = np.array([b[:4] for b in out['bboxes']], np.single)
        return (out['image'].astype(np.single),
                bout_cl, bout_xy,
                out['mask'], name)

    def tf_wrap(self):
        def _tf_wrap(image, boxes_cl, boxes_xy, mask, name):
            image, cl, xy, mask, name = tf.numpy_function(
                self,
                (image, boxes_cl, boxes_xy, mask, name),
                (tf.float32, tf.int64, tf.float32, tf.uint8, tf.string)
            )
            image.set_shape([None, None, 3])
            mask.set_shape([None, None, 1])
            return (image, cl, xy, mask, name)
        return _tf_wrap
