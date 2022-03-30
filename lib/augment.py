# -*- coding: utf-8 -*-

"""
Augmentation of images for training.
"""

from typing import Callable

import tensorflow as tf

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2022, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def augmentor(height: int, width: int, seed: int = 42) -> Callable:
    """Augment image: flip, adjust contrast.
    """
    tf_seed = tf.Variable(initial_value=[seed, 0], trainable=False)
    brightness_delta_max = 0.3
    contrast_min = 0.7
    contrast_max = 1.3
    hue_delta_max = 0.2
    # disable for test ...
    brightness_delta_max = hue_delta_max = 0.0
    contrast_min = contrast_max = 0.0

    @tf.function
    def _augmentor(img, b_cl, b_yx, mask, has_mask, name):
        # random brightness adjustment
        img = tf.image.stateless_random_brightness(
            img, brightness_delta_max, tf_seed
        )
        return (img, b_cl, b_yx, mask, has_mask, name)
        # random contrast adjustment
        img = tf.image.stateless_random_contrast(
            img, contrast_min, contrast_max, tf_seed
        )
        # random hue adjustment
        img = tf.image.stateless_random_hue(
            img, hue_delta_max, tf_seed
        )
        tf_seed.assign_add(delta=[0, 1])
        return (img, b_cl, b_yx, mask, has_mask, name)
    return _augmentor
