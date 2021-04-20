# -*- coding: utf-8 -*-

"""
This package defines the DeepLab segmentation model.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Softmax

from .layers import aspp, decoder

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def add_deeplab_features(
        base: Model, num_classes: int, output_stride: int = 8) -> Model:
    """Add separate layers on top of MobileNet for DeepLab.
    """
    inputs = base.input
    outputs = base.output
    # Add ASPP blocks
    x = aspp(outputs, name="aspp", output_stride=output_stride)
    # fetch skip feature
    skip = base.get_layer(name="bottleneck_2")
    # Add decoder block
    x = decoder(x, skip)
    # final prediction block
    x = Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        name="deeplab_conv2d")(x)
    # resize to original size
    img_size = inputs.shape.as_list[1:3]
    x = tf.image.resize(x, img_size, name="deeplab_resize")
    # softmax activation
    x = Softmax(name="deeplab_softmax")(x)
    return Model(inputs=inputs, outputs=x, name="deeplab")
