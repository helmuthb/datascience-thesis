# -*- coding: utf-8 -*-

"""
This package defines the SSD-Lite model.
"""

from math import sqrt
from itertools import product
from typing import List

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


def ssdlite_base_outputs(model: Model, config: dict) -> tuple:
    """From an extended MobileNetV2 model, get the base layers used for SSDlite.
    The following layers are used by SSDlite:
    * The layer following the first five bottleneck blocks,
      after performing the next expansion (called bottleneck_14.expand)
    * The final layer (called conv_pw1)
    * Four additional bottleneck layers (called feat1 till feat4)
    * Four additional layers which are Bottleneck layers
    Args:
        model (Model): Base model.
    Returns:
        outputs (tuple): Outputs to be used by SSDlite.
    """
    outputs = []
    # get existing layers of interest
    for layer_name in config['feature_layers']:
        output = model.get_layer(name=layer_name).output
        outputs.append(output)
    # add additional layers of interest
    output = model.get_layer(name=config['out_layer']).output
    # layer = model.output
    for ssd_layer in config['ssd_layers']:
        output = bottleneck(inputs=output, **ssd_layer)
        outputs.append(output)
    # return model & layers
    return outputs


def get_num_default_ratios(config: dict) -> List[int]:
    """Get the number of default box ratios.

    Each aspect ratio is used twice. And we add 2 - one
    for ratio 1:1, and one for the larger size derived from
    the next layer.
    This explains the below formula.

    Returns:
        num_default_ratios (list(int)): Number of default ratios per layer.
    """
    return [2*len(a)+2 for a in config['aspect_ratios']]


def head_output(prefix: str, index: int, n_out: int, n_boxes: int,
                inp: tf.Tensor) -> tf.Tensor:
    """Head layer creating the boxes / class scores.
    Args:
        prefix (str): Prefix ("classes" or "boxes").
        index (int): Number of layer.
        n_out (int): Either 4 (for boxes) or number of classes.
        n_boxes (int): Number of default boxes to output.
        inp (tf.keras.KerasTensor): Layer input.
    Returns:
        output (tf.keras.layers.Layer): Output of size [N, n_boxes, n_out].
    """
    dw_relu = depthwise_bn_relu6(
        inputs=inp,
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


def detection_heads(n_classes: int, layers: tuple, config: dict) -> tuple:
    """Convolutional heads for bounding boxes and class scores.
    Args:
        n_classes (int): Number of classes to predict.
        layers (tuple): Tuple of base layers for SSDLite.
    Returns:
        classes (tf.keras.layers.Layer): Output layer for classes
            of size [N, n_defaults, n_classes].
        boxes (tf.keras.layers.Layer): Output layer for boxes
            of size [N, n_defaults, 4].
    """
    # Number of default boxes per layer to consider
    n_defaults = get_num_default_ratios(config)
    # Outputs for class predictions
    out_classes = []
    for i, (n_a, layer) in enumerate(zip(n_defaults, layers)):
        out = head_output("classes", i, n_classes, n_a, layer)
        out_classes.append(out)
    # concatenate
    classes = tf.keras.layers.Concatenate(axis=1, name='ssd_conf_output')(
        out_classes
    )
    # Outputs for bounding boxes
    out_boxes = []
    for i, (n_a, layer) in enumerate(zip(n_defaults, layers)):
        out = head_output("boxes", i, 4, n_a, layer)
        out_boxes.append(out)
    # concatenate
    boxes = tf.keras.layers.Concatenate(axis=1, name='ssd_bbox_output')(
        out_boxes
    )
    # return bounding boxes & classes
    return classes, boxes


def get_default_boxes_cw(outputs: tuple, config: dict) -> tf.Tensor:
    """Get the default bounding boxes for the given layers.
    """
    boxes = []
    for i in range(len(config['obj_scales'])):
        scale = config['obj_scales'][i]
        # width, height of the layer
        width, height = outputs[i].shape[1:3]
        for xi, yi in product(range(width), range(height)):
            # rounded from 0 to 1
            x = (xi + 0.5) / width
            # rounded from 0 to 1
            y = (yi + 0.5) / height
            for ratio in config['aspect_ratios'][i]:
                sqrt_ratio = sqrt(ratio)
                boxes.append([x, y, scale*sqrt_ratio, scale/sqrt_ratio])
                boxes.append([x, y, scale/sqrt_ratio, scale*sqrt_ratio])
            # additional box: square
            boxes.append([x, y, scale, scale])
            # additional box: geom. mean between this and the next scale
            if i+1 < len(config['obj_scales']):
                next_scale = config['obj_scales'][i+1]
                additional_scale = sqrt(scale * next_scale)
            else:
                additional_scale = 1.
            boxes.append([x, y, additional_scale, additional_scale])
    # clip the result
    return tf.convert_to_tensor(np.clip(boxes, 0., 1.), dtype=tf.float32)
