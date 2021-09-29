# -*- coding: utf-8 -*-

"""
Augmentation of images for training.
"""

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

    def __call__(self, images, boxes_xy, boxes_cl, mask, name):
        if self.t is None:
            self.t = Compose(
                self.transforms,
                bbox_params=BboxParams(format='albumentations'))
        t = self.t
        boxes = [list(b) + [str(c)] for b, c in zip(boxes_xy, boxes_cl)]
        out = t(
            image=images,
            bboxes=boxes,
            mask=mask
        )
        bout_xy = [b[:4] for b in out['bboxes']]
        bout_cl = [int(b[4]) for b in out['bboxes']]
        return (out['image'], bout_xy, bout_cl,
                out['mask'], name)