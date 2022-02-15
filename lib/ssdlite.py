# -*- coding: utf-8 -*-

"""
This package defines the SSD-Lite model.
"""

from math import sqrt
from itertools import product

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

from .layers import bottleneck, depthwise_bn_relu6

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


# A-priori object scales to use for default prediction boxes
OBJ_SCALES = [0.2, 0.4, 0.55, 0.7, 0.8, 0.9]

# A-priori aspect ratios to use for default prediction boxes
ASPECT_RATIOS = [
    [1., 2., 3., 0.5, 1./3.],
    [1., 2., 3., 0.5, 1./3.],
    [1., 2., 3., 0.5, 1./3.],
    [1., 2., 3., 0.5, 1./3.],
    [1., 2., 3., 0.5, 1./3.],
    [1., 2., 3., 0.5, 1./3.]
]


def add_ssdlite_features(base: Model) -> Model:
    """Add feature layers on top of a MobileNetV2 model.
    """
    inputs = base.input
    outputs = base.output
    config = [
        {"num_filters": 512, "strides": 2, "expand_ratio": .2},
        {"num_filters": 256, "strides": 2, "expand_ratio": .5},
        {"num_filters": 256, "strides": 2, "expand_ratio": .5},
        {"num_filters": 64,  "strides": 2, "expand_ratio": .5}
    ]
    for i, cfg in enumerate(config, start=1):
        outputs = bottleneck(inputs=outputs, name=f"feat{i}", **cfg)
    return Model(inputs=inputs, outputs=outputs)


def ssdlite_base_layers(model: Model):
    """From an extended MobileNetV2 model, get the base layers used for SSDlite.
    The following layers are used by SSDlite:
    * The layer following the first five bottleneck blocks,
      after performing the next expansion (called bottleneck_14.expand)
    * The final layer (called conv_pw1)
    * Four additional bottleneck layers (called feat1 till feat4)
    * Four additional layers which are Bottleneck layers
    Args:
        model (Model): MobileNetV2 model.
    Returns:
        layers (6-Tuple): Six layers to be used by SSDlite.
    """
    # get names of layers to identify correct ones
    names = {layer.name for layer in model.layers}
    # get first layer of interest
    if "bottleneck_14_expand_relu6" in names:
        layer1 = model.get_layer(name="bottleneck_14_expand_relu6")
    elif "block_13_expand_relu" in names:
        layer1 = model.get_layer(name="block_13_expand_relu")
    else:
        raise ValueError("Unknown base model")
    # get second layer of interest = final layer
    if "conv_pw1_relu6" in names:
        layer2 = model.get_layer(name="conv_pw1_relu6")
    elif "out_relu" in names:
        layer2 = model.get_layer(name="out_relu")
    else:
        raise ValueError("Unknown base model")
    # add four additional Bottleneck layers
    layer3 = model.get_layer(name="feat1_project_bn")
    layer4 = model.get_layer(name="feat2_project_bn")
    layer5 = model.get_layer(name="feat3_project_bn")
    layer6 = model.get_layer(name="feat4_project_bn")
    # return tuple of layers
    return layer1, layer2, layer3, layer4, layer5, layer6


def get_num_default_ratios():
    """Get the number of default box ratios.

    For each layer and aspect ratio, add 1.
    In addition, for each layer add one default box with the
    geometric mean between the scale and the next scale.
    This explains the below formula.

    Returns:
        num_default_ratios (list(int)): Number of default ratios per layer.
    """
    return [len(a)+1 for a in ASPECT_RATIOS]


def head_layer(prefix, index, n_out, n_boxes, inp):
    """Head layer creating the boxes / class scores.
    Args:
        prefix (str): Prefix ("classes" or "boxes").
        index (int): Number of layer.
        n_out (int): Either 4 (for boxes) or number of classes.
        n_boxes (int): Number of default boxes to output.
        inp (tf.keras.layers.Layer): Layer input.
    Returns:
        layer (tf.keras.layers.Layer): Output layer
            of size [N, n_boxes, n_out].
    """
    dw_relu = depthwise_bn_relu6(
        inputs=inp.output,
        name=f"{prefix}_dw{index}",
        strides=1
    )
    conv_2d = tf.keras.layers.Conv2D(
        filters=n_out * n_boxes,
        kernel_size=1,
        name=f"{prefix}_conv{index}"
    )(dw_relu)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=f"{prefix}_bn{index}"
    )(conv_2d)
    reshape = tf.keras.layers.Reshape(
        [-1, n_out],
        name=f"{prefix}_reshape{index}"
    )(bn)
    return reshape


def detection_head(n_classes, *layers):
    """Convolutional heads for bounding boxes and class scores.
    Args:
        n_classes (int): Number of classes to predict.
        l (6*tuple): Tuple of layers from MobileNetV2 & SSDLite.
    Returns:
        classes (tf.keras.layers.Layer): Output layer for classes
            of size [N, n_defaults, n_classes].
        boxes (tf.keras.layers.Layer): Output layer for boxes
            of size [N, n_defaults, 4].
    """
    # Number of default boxes per layer to consider
    n_defaults = get_num_default_ratios()
    # Outputs for class predictions
    out_classes = []
    for i, (n_a, layer) in enumerate(zip(n_defaults, layers)):
        out = head_layer("classes", i, n_classes, n_a, layer)
        out_classes.append(out)
    # concatenate
    classes = tf.keras.layers.Concatenate(axis=1, name='ssd_conf_output')(
        out_classes
    )
    # Outputs for bounding boxes
    out_boxes = []
    for i, (n_a, layer) in enumerate(zip(n_defaults, layers)):
        out = head_layer("boxes", i, 4, n_a, layer)
        out_boxes.append(out)
    # concatenate
    boxes = tf.keras.layers.Concatenate(axis=1, name='ssd_bbox_output')(
        out_boxes
    )
    # return bounding boxes & classes
    return classes, boxes


def get_default_boxes_cw(*layers):
    """Get the default bounding boxes for the given layers.
    """
    boxes = []
    for i in range(6):
        scale = OBJ_SCALES[i]
        # width, height of the layer
        width, height = layers[i].output_shape[1:3]
        for xi, yi in product(range(width), range(height)):
            # rounded from 0 to 1
            x = (xi + 0.5) / width
            # rounded from 0 to 1
            y = (yi + 0.5) / height
            for ratio in ASPECT_RATIOS[i]:
                sqrt_ratio = sqrt(ratio)
                boxes.append([x, y, scale*sqrt_ratio, scale/sqrt_ratio])
            # additional box: geom. mean between this
            # and the next object scale
            if i < 5:
                next_scale = OBJ_SCALES[i+1]
                additional_scale = sqrt(scale * next_scale)
            else:
                additional_scale = 1.
            boxes.append([x, y, additional_scale, additional_scale])
    # clip the result
    return tf.convert_to_tensor(np.clip(boxes, 0., 1.), dtype=tf.float32)
