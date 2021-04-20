# -*- coding: utf-8 -*-

"""
Model for MobileNetV2.
"""

from tensorflow.keras.models import Model
from .layers import bottleneck, conv_bn_relu6

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def mobilenetv2(input_layer):
    """Base MobileNetV2 model, reimplemented in Keras.
    The top layer is not part of the model as it is used as backend
    for object detection
    and segmentation.
    """
    # first layer: Conv2D with 32 channels
    layer = conv_bn_relu6(
        inputs=input_layer,
        name="conv1",
        num_filters=32,
        kernel_size=(3, 3),
        strides=2)
    # adding Bottleneck blocks
    bottleneck_configs = (
        # t, c, n, s
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )
    i = 1
    for t, c, n, s in bottleneck_configs:
        for seq_num in range(n):
            if seq_num > 0:
                # only the first layer in a sequence has strides > 1
                s = 1
            layer = bottleneck(
                inputs=layer,
                num_filters=c,
                name=f"bottleneck_{i}",
                strides=s,
                expand_ratio=t
            )
            # count layer
            i += 1
    # Pointwise Conv2D layer
    layer = conv_bn_relu6(
        inputs=layer,
        name="conv_pw1",
        num_filters=1280,
        kernel_size=(1, 1),
        strides=1
    )
    return Model(inputs=input_layer, outputs=layer)
