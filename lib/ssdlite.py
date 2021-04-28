# -*- coding: utf-8 -*-

"""
This package defines the SSD-Lite model.
"""

from math import sqrt
from itertools import product

import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import bottleneck
from .mobilenet import mobilenetv2

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


# A-priori object scales to use for anchor prediction boxes
OBJ_SCALES = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]

# A-priori aspect ratios to use for anchor prediction boxes
ASPECT_RATIOS = [
    [1., 2., 0.5],
    [1., 2., 3., 0.5, 1/3],
    [1., 2., 3., 0.5, 1/3],
    [1., 2., 3., 0.5, 1/3],
    [1., 2., 0.5],
    [1., 2., 0.5]
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
    # get first layer of interest
    layer1 = model.get_layer(name="bottleneck_14_expand_relu6")
    # get second layer of interest = final layer
    layer2 = model.get_layer(name="conv_pw1_relu6")
    # add four additional Bottleneck layers
    layer3 = model.get_layer(name="feat1_project_bn")
    layer4 = model.get_layer(name="feat2_project_bn")
    layer5 = model.get_layer(name="feat3_project_bn")
    layer6 = model.get_layer(name="feat4_project_bn")
    # return tuple of layers
    return layer1, layer2, layer3, layer4, layer5, layer6


def get_num_anchor_boxes():
    """Get the number of anchor boxes.

    For each layer and aspect ratio, add 1.
    In addition, for each layer add one anchor box with the
    geometric mean between the scale and the next scale.
    This explains the below formula.

    Returns:
        num_anchor_boxes (list(int)): Number of anchor boxes per layer
    """
    return [len(a)+1 for a in ASPECT_RATIOS]


def detection_head(n_classes, *layers):
    """Convolutional heads for bounding boxes and class scores.
    Args:
        n_classes (int): Number of classes to predict.
        l (6*tuple): Tuple of layers from MobileNetV2 & SSDLite.
    Returns:
        layer (tf.keras.layers.Layer): Combined output layer
            of size [N, n_anchors, 4+n_classes].
            The first 4 in the last dimension are for
            bounding boxes, the remaining are for the classes.
    """
    # Number of anchor boxes per layer to consider
    n_anchors = get_num_anchor_boxes()
    # Outputs for class predictions
    out_classes = [
        tf.keras.layers.Conv2D(
            filters=n_a*n_classes,
            kernel_size=1,
            name=f"classes_conv{i}"
        )(layer.output)
        for i, n_a, layer in zip(range(6), n_anchors, layers)
    ]
    # reshape & concatenate
    classes = tf.keras.layers.Concatenate(axis=1, name='classes_concat')([
        tf.keras.layers.Reshape(
            [-1, n_classes],
            name=f"classes_reshape{i}")(out_c)
        for i, out_c in enumerate(out_classes)
    ])
    # Outputs for bounding boxes
    out_bboxes = [
        tf.keras.layers.Conv2D(
            filters=n_a*4,
            kernel_size=1,
            name=f"bbox_conv{i}"
        )(layer.output)
        for i, n_a, layer in zip(range(6), n_anchors, layers)
    ]
    # reshape & concatenate
    bboxes = tf.keras.layers.Concatenate(axis=1, name='bbox_concat')([
        tf.keras.layers.Reshape([-1, 4], name=f"bbox_reshape{i}")(out_bbox)
        for i, out_bbox in enumerate(out_bboxes)
    ])
    # return bounding boxes & classes concatenated
    return tf.keras.layers.Concatenate(name="ssd_output")([bboxes, classes])


def combine_boxes_classes(bboxes, classes, n_classes):
    """Combine the bounding boxes and (sparse) classes in one tensor.
    It will perform a one-hot encoding of the bounding boxes,
    and concatenate the resulting tensor with the classes.
    Args:
        bboxes (tf.Tensor, [N, n_anchors, 4]): Tensor with bounding boxes.
        classes (tf.Tensor, [N, n_anchors]): Dense tensor with classes.
        n_classes (int): Number of classes.
    Returns:
        output (tf.Tensor, [N, n_anchors, 4+n_classes]): Combined tensor.
    """
    # one-hot encoding of classes
    one_hot = tf.one_hot(classes, n_classes, dtype=tf.float32, axis=-1)
    # return concatenation
    return tf.concat([bboxes, one_hot], axis=-1)


def get_anchor_boxes_cwh(*layers):
    """Get the anchor bounding boxes for the given layers.
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
    return tf.clip_by_value(boxes, 0., 1.)


def ssdlite(input_shape, n_classes):
    """Get SSDLite model and anchor boxes.

    Args:
        input_shape (list(integer)): (height, width).
        n_classes (integer): Number of classes.
    Returns:
        ssdlite (tf.keras model): SSDLite model, with added loss.
        anchors (float array, [n_priors, 4]): All the anchor boxes
            ((x, y, w, h) within [0., 1.]).
    """
    input_layer = tf.keras.layers.Input(
        shape=(input_shape[0], input_shape[1], 3))
    # Base model
    base = mobilenetv2(input_layer)
    print(base.summary())
    ext_base = add_ssdlite_features(base)
    l1, l2, l3, l4, l5, l6 = ssdlite_base_layers(ext_base)
    # add class and location predictions
    prediction = detection_head(n_classes, l1, l2, l3, l4, l5, l6)
    # create model
    model = tf.keras.Model(inputs=input_layer, outputs=prediction)
    # calculate anchor boxes
    anchor_boxes_cwh = get_anchor_boxes_cwh(l1, l2, l3, l4, l5, l6)
    # return both
    return model, anchor_boxes_cwh


def get_predicted_boxes_cwh(pred_adj, anchors_cwh):
    """Get the corrected boxes, based on the anchors and the predictions.
    The anchor boxes (provided in center mode) and the predicted
    adjustments are combined to get the predicted bounding boxes.

    Args:
        pred_adj (tensor): location predictions, as adjustments of
            center and width / height.
        anchors_cwh (np.ndarray): anchor boxes, in (cx/cy/w/h) format.
    Returns:
        pred_cwh (np.ndarray): anchor boxes.
    """
