# -*- coding: utf-8 -*-

"""
This package defines the DeepLabV3+ segmentation model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Softmax
import tensorflow as tf

from .layers import ImageResize, aspp, decoder

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
        base: Model, num_classes: int, config: dict) -> tf.Tensor:
    """Add separate layers on top of MobileNet for DeepLab.
    """
    inputs = base.input
    output = base.get_layer(name=config['out_layer']).output
    # Add ASPP blocks
    x = aspp(output, name="aspp", output_stride=config['output_stride'])
    # fetch skip feature
    skip = base.get_layer(name=config['skip_feature']).output
    # Add decoder block
    x = decoder(x, skip, name="decoder")
    # final prediction block
    x = Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        name="deeplab_conv2d")(x)
    # resize to original size
    img_size = inputs.shape[1:3]
    x = ImageResize(img_size=img_size, name="deeplab_resize")(x)
    # softmax activation
    x = Softmax(name="deeplab_output")(x)
    # return outputs
    return x
