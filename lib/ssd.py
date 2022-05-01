# -*- coding: utf-8 -*-

"""
This package defines the SSD model.
"""

from math import sqrt
from itertools import product
from typing import List

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization

from .layers import ssd_extra, ssd_extra_simple, bottleneck, depthwise_bn_relu6

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2022, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def ssd_base_outputs(model: Model, config: dict) -> tuple:
    """From an extended MobileNetV2 model, get the base layers used for SSD.
    The following layers are used by SSD:
    * The layer following the first five bottleneck blocks,
      after performing the next expansion (called block_13_expand_relu)
    * The final layer (called out_relu)
    * Four additional bottleneck layers (called feat1 till feat4)
    Args:
        model (Model): Base model.
    Returns:
        outputs (tuple): Outputs to be used by SSDlite.
    """
    outputs = []
    # get existing layers of interest
    out_layers = zip(config.ssd.out_layers, config.ssd.out_batchnorm)
    for layer_name, with_bn in out_layers:
        output = model.get_layer(name=layer_name).output
        if with_bn:
            output = BatchNormalization(name=layer_name + "_bn")(output)
        outputs.append(output)
    # add additional layers of interest
    output = model.get_layer(name=config.ssd.last_layer).output
    # layer = model.output
    for ssd_layer in config.ssd.extra_layers:
        if config.ssd.extra_layers_type == 'ssd':
            output = ssd_extra(inputs=output, **ssd_layer)
        elif config.ssd.extra_layers_type == 'ssd_simple':
            output = ssd_extra_simple(inputs=output, **ssd_layer)
        elif config.ssd.extra_layers_type == 'ssdlite':
            output = bottleneck(inputs=output, **ssd_layer)
        else:
            raise ValueError(f'Configuration `ssd.extra_layers_type` has '
                             f'unknown value "{config.ssd.extra_layers_type}"')
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
    return [2*len(a)+2 for a in config.ssd.aspect_ratios]


def head_output_ssd(prefix: str, index: int, n_out: int, n_boxes: int,
                    inp: tf.Tensor) -> tf.Tensor:
    """Head layer creating the boxes / class scores as used in SSD.
    Args:
        prefix (str): Prefix ("classes" or "boxes").
        index (int): Number of layer.
        n_out (int): Either 4 (for boxes) or number of classes.
        n_boxes (int): Number of default boxes to output.
        inp (tf.keras.KerasTensor): Layer input.
    Returns:
        output (tf.keras.layers.Layer): Output of size [N, n_boxes, n_out].
    """
    conv_2d = tf.keras.layers.Conv2D(
        filters=n_out * n_boxes,
        kernel_size=3,
        padding='same',
        name=f"{prefix}{index}_conv"
    )(inp)
    reshape = tf.keras.layers.Reshape(
        [-1, n_out],
        name=f"{prefix}{index}_reshape"
    )(conv_2d)
    return reshape


def head_output_ssdlite(prefix: str, index: int, n_out: int, n_boxes: int,
                        inp: tf.Tensor) -> tf.Tensor:
    """Head layer creating the boxes / class scores as used in SSDlite.
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
        name=f"{prefix}{index}_dw",
        strides=1
    )
    conv_2d = tf.keras.layers.Conv2D(
        filters=n_out * n_boxes,
        kernel_size=1,
        name=f"{prefix}{index}_conv"
    )(dw_relu)
    reshape = tf.keras.layers.Reshape(
        [-1, n_out],
        name=f"{prefix}{index}_reshape"
    )(conv_2d)
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
        if config.ssd.detector == 'ssd':
            out = head_output_ssd("classes", i+1, n_classes, n_a, layer)
        else:
            out = head_output_ssdlite("classes", i+1, n_classes, n_a, layer)
        out_classes.append(out)
    # concatenate
    classes = tf.keras.layers.Concatenate(axis=1, name='ssd_classes')(
        out_classes
    )
    # Outputs for bounding boxes
    out_boxes = []
    for i, (n_a, layer) in enumerate(zip(n_defaults, layers)):
        if config.ssd.detector == 'ssd':
            out = head_output_ssd("boxes", i+1, 4, n_a, layer)
        else:
            out = head_output_ssdlite("boxes", i+1, 4, n_a, layer)
        out_boxes.append(out)
    # concatenate
    boxes = tf.keras.layers.Concatenate(axis=1, name='ssd_boxes')(
        out_boxes
    )
    # return bounding boxes & classes
    return classes, boxes


def get_default_boxes_cw(outputs: tuple, config: dict) -> tf.Tensor:
    """Get the default bounding boxes for the given layers.
    """
    boxes = []
    obj_scales = config.ssd.obj_scales
    for i in range(len(obj_scales)):
        scale = obj_scales[i]
        height, width = outputs[i].shape[1:3]
        # for xi, yi in product(range(width), range(height)):
        for yi, xi in product(range(height), range(width)):
            # rounded from 0 to 1
            x = (xi + 0.5) / width
            # rounded from 0 to 1
            y = (yi + 0.5) / height
            # square box
            boxes.append([y, x, scale, scale])
            # additional box: geom. mean between this and the next scale
            if i+1 < len(obj_scales):
                next_scale = obj_scales[i+1]
                additional_scale = sqrt(scale * next_scale)
            else:
                additional_scale = 1.
            boxes.append([y, x, additional_scale, additional_scale])
            # additional boxes: for each ratio
            for ratio in config.ssd.aspect_ratios[i]:
                sqrt_ratio = sqrt(ratio)
                boxes.append([y, x, scale*sqrt_ratio, scale/sqrt_ratio])
                boxes.append([y, x, scale/sqrt_ratio, scale*sqrt_ratio])
    # clip the result
    return tf.convert_to_tensor(np.clip(boxes, 0., 1.), dtype=tf.float32)
