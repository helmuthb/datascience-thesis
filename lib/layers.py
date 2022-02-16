# -*- coding: utf-8 -*-

"""
Additional layers as needed for SSD-Lite and MobileNetV2.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, BatchNormalization, Conv2D, ReLU, DepthwiseConv2D, AveragePooling2D,
    Concatenate, Dropout, Add, ZeroPadding2D)
import tensorflow.keras.backend as K

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def _round8(val):
    """Make the number val divisible by 8.

    By default the value will be rounded to the nearest number divisible
    by 8. However, if the result is smaller than 8, or if the result
    is more than 10% smaller than the original value, then it will be
    rounded up.
    """
    new_val = max(8, (int(val+4)//8)*8)
    if new_val < 0.9 * val:
        new_val = new_val + 8
    return new_val


def conv_bn_relu6(
        inputs, name, num_filters, strides, kernel_size, dilation_rate=1):
    """Perform Conv2D, followed by BatchNorm and using ReLU6 as activation.
    """
    conv = Conv2D(
            filters=num_filters,
            strides=strides,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False,
            activation=None,
            dilation_rate=dilation_rate,
            name=f"{name}_conv",
        )(inputs)
    bn = BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform',
            name=f"{name}_bn"
        )(conv)
    return ReLU(6., name=f"{name}_relu6")(bn)


def depthwise_bn_relu6(inputs, name, strides, dilation_rate=1, padding='same'):
    """Depthwise 3x3 Conv2D, as used in Inverted ResNet block.
    """
    conv = DepthwiseConv2D(
            kernel_size=3,
            strides=strides,
            padding=padding,
            use_bias=False,
            activation=None,
            dilation_rate=dilation_rate,
            name=f"{name}_conv")(inputs)
    bn = BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform',
            name=f"{name}_bn")(conv)
    return ReLU(6., name=f"{name}_relu6")(bn)


def separable_conv2d(inputs, name, num_filters, strides, dilation_rate=1):
    """Separable convolution, composed of a Depthwise 3x3 Conv2D
    and a 1x1 convolution.
    """
    depthwise = depthwise_bn_relu6(
            inputs=inputs,
            name=f"{name}_dw",
            strides=strides,
            dilation_rate=dilation_rate
            )
    return conv_bn_relu6(
            inputs=depthwise,
            name=f"{name}_pw",
            num_filters=num_filters,
            kernel_size=(1, 1),
            strides=1
            )


def bottleneck(
        inputs, name, num_filters, strides, expand_ratio):
    """Inverted ResNet block.
    Compared to the original ResNet block - which first squeezed and
    then expanded back to the orignal size - this block works the other
    way round. It first expands the size by adding many filters and
    then squeezes it back to the original size.
    If strides == 1 then the original tensor is added to the end result.
    """
    num_filters = _round8(num_filters)
    input_shape = inputs.shape[1:]
    in_channels = input_shape[2]
    hidden_dim = in_channels * expand_ratio
    if expand_ratio != 1:
        expand = conv_bn_relu6(
            inputs=inputs,
            num_filters=hidden_dim,
            kernel_size=(1, 1),
            strides=1,
            name=f"{name}_expand"
        )
    else:
        expand = inputs
    expand_size = K.int_shape(expand)[1:3]
    if expand_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - expand_size[0] % 2, 1 - expand_size[1] % 2)
    padding = ((1 - adjust[0], 1), (1 - adjust[1], 1))
    zeropad = ZeroPadding2D(padding=padding, name=f"{name}_pad")(expand)
    depthwise = depthwise_bn_relu6(
        inputs=zeropad,
        strides=strides,
        name=f"{name}_dw",
        padding='same' if strides == 1 else 'valid'
    )
    project_conv = conv_bn_relu6(
        inputs=depthwise,
        num_filters=num_filters,
        kernel_size=(1, 1),
        strides=1,
        name=f"{name}_project"
    )
    if strides == 1 and num_filters == in_channels:
        added = Add(name=f"{name}_added")([project_conv, inputs])
    else:
        added = project_conv
    return added


class AdaptiveAvgPool2D(Layer):
    """AvgPooling layer which will output a dimension of (1, 1) at the end.
    """
    def __init__(self, name):
        super().__init__(name=name)

    def build(self, input_shape):
        self.avg_layer = AveragePooling2D(
            pool_size=input_shape[1:3],
            padding='valid'  # no padding
        )

    def call(self, inputs):
        return self.avg_layer(inputs)


class ImageResize(Layer):
    """Keras Layer for resizing of an image.
    """
    def __init__(self, name, img_size):
        super().__init__(name=name)
        self.img_size = img_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.img_size)

    def get_config(self):
        config = super().get_config()
        config.update({'img_size': self.img_size})
        return config


def aspp(inputs, name, output_stride):
    """
    """
    # original size
    img_size = inputs.shape[1:3]
    # rates in atrous convolutions
    if output_stride == 8:
        atrous_rates = [12, 24, 36]
    elif output_stride == 16:
        atrous_rates = [6, 12, 18]
    else:
        raise ValueError(
            f"`output_stride` only supports 8 or 16, not {output_stride}")
    # part 1: get a 1x1 average pool layer, and blow it up
    part1 = AdaptiveAvgPool2D(name=f"{name}_p1_average_pool")(inputs)
    part1 = conv_bn_relu6(
        part1,
        name=f"{name}_p1_conv1x1",
        num_filters=256,
        strides=1,
        kernel_size=(1, 1))
    part1 = ImageResize(name=f"{name}_p1_resize", img_size=img_size)(part1)
    # part 2: 1x1 convolution
    part2 = conv_bn_relu6(
        inputs,
        name=f"{name}_p2_conv1x1",
        num_filters=256,
        strides=1,
        kernel_size=(1, 1))
    # part 3-5: atrous convolutions
    partX = [
        separable_conv2d(
            inputs,
            name=f"{name}_p{i}_{r}",
            num_filters=256,
            strides=1,
            dilation_rate=r
            )
        for i, r in enumerate(atrous_rates, 3)]
    # concatenate
    concat = Concatenate(name=f"{name}_concat")([part1, part2] + partX)
    # project
    projected = conv_bn_relu6(
        concat,
        name=f"{name}_project",
        num_filters=256,
        strides=1,
        kernel_size=(1, 1),
        dilation_rate=1
    )
    # add dropout
    return Dropout(rate=0.5, name=f"{name}_dropout")(projected)


def decoder(inputs, inputs2, name):
    """
    """
    # resize inputs to the size of inputs2
    img_size = inputs2.shape[1:3]
    x = ImageResize(img_size=img_size, name=f"{name}_resize")(inputs)
    # 1x1 convolution on the second inputs
    y = conv_bn_relu6(
        inputs=inputs2,
        name=f"{name}_projection",
        num_filters=48,
        strides=1,
        kernel_size=(1, 1)
    )
    # concatenate
    x = Concatenate(name=f"{name}_concat")([x, y])
    # separable convolution
    x = separable_conv2d(
        inputs=x,
        name=f"{name}_sepconv1",
        num_filters=256,
        strides=1,
    )
    # second separable convolution
    x = separable_conv2d(
        inputs=x,
        name=f"{name}_sepconv2",
        num_filters=256,
        strides=1,
    )
    return x
