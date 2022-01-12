# -*- coding: utf-8 -*-

"""
This package defines the DeepLabV3+ segmentation model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Softmax

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
        base: Model, num_classes: int, output_stride: int = 8) -> Model:
    """Add separate layers on top of MobileNet for DeepLab.
    """
    inputs = base.input
    outputs = base.output
    names = {layer.name for layer in base.layers}
    # Add ASPP blocks
    x = aspp(outputs, name="aspp", output_stride=output_stride)
    # fetch skip feature
    if "bottleneck_2_project_bn" in names:
        # In case of locally created model
        skip = base.get_layer(name="bottleneck_2_project_bn").output
    elif "block_1_project_BN" in names:
        # In case of Keras provided model
        skip = base.get_layer(name="block_1_project_BN").output
    else:
        raise ValueError("Base model is unknown")
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
    # return Model(inputs=inputs, outputs=x, name="deeplab")
