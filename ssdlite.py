import tensorflow as tf
import tensorflow.keras.layers as layers
import tf.keras.applications as applications
import numpy as np
from math import sqrt


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


def round8(val):
    """Make the number v divisible by 8.

    By default the value will be rounded to the nearest number divisible
    by 8. However, if the result is smaller than 8, or if the result
    is more than 10% smaller than the original value, then it will be
    rounded up.
    """
    new_val = max(8, (int(8+4)//8)*8)
    if new_val < 0.9 * val:
        new_val = new_val + 8
    return new_val


def inverted_res_block(num_filters, alpha, name):
    """Inverted res block.
    """
    # number of output filters (adapted by alpha)
    num_filters = int(num_filters * alpha)
    num_filters = round8(num_filters)
    return layers.Sequential([
        # Expand
        layers.Conv2D(
            filters=num_filters//2,
            kernel_size=1,
            use_bias=False,
            activation=None,
            name=f"{name}_conv_expand"),
        layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=f"{name}_bn_expand"),
        layers.ReLU(6., name=f"{name}_relu6_expand"),
        # Depthwise
        layers.DepthwiseConv2D(
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            activation=None,
            name=f"{name}_conv_depthwise"),
        layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=f"{name}_bn_depthwise"),
        layers.ReLU(6., name=f"{name}_relu6_depthwise"),
        # Project
        layers.Conv2D(
            filters=num_filters,
            kernel_size=1,
            use_bias=False,
            activation=None,
            name=f"{name}_conv_project"),
        layers.BatchNormalization(name=f"{name}_bn_project"),
        layers.ReLU(6., name=f"{name}_relu6_project")
    ], name=name)


def ssdlite_base(input_shape, alpha):
    """Load base MobileNetV2 together with four separable conv layers.
    Args:
        input_shape (tuple): Input shape to use.
        alpha (float): Float between 0 and 1 for the width of the network.
    Returns:
        layer (tf.keras.layers.Layer): a layer for the combined network
        feat1 (tf.keras.layers.Layer): relu layer of block 13 from MobileNetV2
        feat2 (tf.keras.layers.Layer): output layer of the MobileNetV2
        feat3 (tf.keras.layers.Layer): output layer of the 1st ssd conv layer
        feat4 (tf.keras.layers.Layer): output layer of the 2nd ssd conv layer
        feat5 (tf.keras.layers.Layer): output layer of the 3rd ssd conv layer
        feat6 (tf.keras.layers.Layer): output layer of the 4th ssd conv layer
    """
    mnetv2 = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        alpha=alpha)
    # get outputs from mobile net
    feat1 = mnetv2.get_layer('block_13_expand_relu')
    # adjust alpha to make following channels divisible
    if alpha == 0.35:
        alpha = 0.25
    sconv1 = inverted_res_block(512, alpha, 'sep1')
    sconv2 = inverted_res_block(256, alpha, 'sep2')
    sconv3 = inverted_res_block(256, alpha, 'sep3')
    sconv4 = inverted_res_block(128, alpha, 'sep4')
    model = layers.Sequential(
        [mnetv2, sconv1, sconv2, sconv3, sconv4],
        name='ssdlitebase')
    return model, feat1, mnetv2, sconv1, sconv2, sconv3, sconv4


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
        layers (6*tuple): Tuple of layers from MobileNetV2 & SSDLite.
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
        layers.Conv2D(
            filters=n_anchors[i]*4,
            kernel_size=1,
            name=f"classes_conv{i}"
        )(layers[i].output)
        for i in range(6)
    ]
    # reshape & concatenate
    classes = layers.Concatenate(axis=1, name='classes_concat')([
        layers.Reshape([-1, 4], name=f"classes_reshape{i}")(out_classes[i])
        for i in range(6)
    ])
    # Outputs for bounding boxes
    out_bbox = [
        layers.Conv2D(
            filters=n_anchors[i]*4,
            kernel_size=1,
            name=f"bbox_conv{i}"
        )(layers[i].output)
        for i in range(6)
    ]
    # reshape & concatenate
    bboxes = layers.Concatenate(axis=1, name='bbox_concat')([
        layers.Reshape([-1, 4], name=f"bbox_reshape{i}")(out_bbox[i])
        for i in range(6)
    ])
    # return bounding boxes & classes concatenated
    return layers.Concatenate()([bboxes, classes])


def split_boxes_classes(output):
    """Split the output from the model into boxes and classes.
    """
    return output[..., :4], output[..., 4:]


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


def get_anchor_boxes(*layers):
    """Get the anchor bounding boxes for the given layers.
    """
    boxes = []
    for i in range(6):
        scale = OBJ_SCALES[i]
        # width, height of the layer
        feat_shape = layers[i].output_shape[1:3]
        for xi in range(feat_shape[0]):
            # rounded from 0 to 1
            x = (xi + 0.5) / feat_shape[0]
            for yi in range(feat_shape[1]):
                # rounced from 0 to 1
                y = (yi + 0.5) / feat_shape[1]
                for ratio in ASPECT_RATIOS:
                    sqrt_ratio = sqrt(ratio)
                    boxes.append(x, y, scale*sqrt_ratio, scale/sqrt_ratio)
                # additional box: geom. mean between this
                # and the next object scale
                if i < 5:
                    next_scale = OBJ_SCALES[i+1]
                    additional_scale = sqrt(scale * next_scale)
                else:
                    additional_scale = 1.
                boxes.append([x, y, additional_scale, additional_scale])
    # clip the result (but should not exceed it anyway)
    return np.clip(boxes, 0., 1.)


def ssd_loss(y_true, y_pred):
    """Calculate the loss tensor for the model.

    It takes the model's outputs as well as the ground truth
    and creates the tensor to get the loss.
    Args:
        y_true (tf.Tensor, [N, n_anchors, 4+n_classes]): true bboxes & classes
        y_pred (tf.Tensor, [N, n_anchors, 4+n_classes]): pred bboxes & classes
    """
    # split into classes & bounding boxes
    bbox_true, cls_true = split_boxes_classes(y_true)
    bbox_pred, cls_pred = split_boxes_classes(y_pred)
    # anchors which have a class
    anchors_positive = cls_true[..., 0] > 0
    anchors_positive_float = tf.cast(anchors_positive, tf.float32)
    # number of positives
    n_positive = tf.reduce_sum(anchors_positive_float, axis=1)
    # classification losses for all boxes
    conf_loss_all = tf.keras.losses.categorical_crossentropy(
        cls_true, cls_pred, from_logits=True
    )
    # positive / negative cases loss
    conf_loss_positive = conf_loss_all[anchors_positive]
    conf_loss_negative = conf_loss_all * (1 - anchors_positive_float)
    # number of negatives with non-zero loss per batch
    n_negative = tf.math.count_nonzero(conf_loss_negative, axis=1)
    # number of "hard negatives" (ratio 3:1 to the positives)
    # only these "hard negatives" are counted in confidence loss
    # also we don't count more than all negative samples
    n_hard_negative = tf.minimum(3*n_positive, n_negative)
    # to find the top n_hard_negative we reshape the negative loss to 1-dim
    conf_loss_negative_1d = tf.reshape(conf_loss_negative, [-1])
    conf_loss_negative_1d_len = tf.shape(conf_loss_negative_1d.shape)[0]
    # get vals & indexes
    vals, indexes = tf.math.top_k(
        input=conf_loss_negative_1d,
        k=n_hard_negative,
        sorted=False)
    # create a mask with the indexes
    one_hot = tf.one_hot(indexes, conf_loss_negative_1d_len, dtype=tf.float32)
    mask = 1 - tf.reduce_sum(one_hot, axis=0)
    # and apply it
    conf_loss_hard_negative = tf.reduce_sum(mask * conf_loss_negative_1d)
    # average the conf losses
    conf_loss = (tf.reduce_sum(conf_loss_hard_negative) +
                 tf.reduce_sum(conf_loss_positive))
    conf_loss = conf_loss / tf.reduce_sum(n_positive)
    # localization loss: delta of bounding boxes which have a true class
    loc_loss = tf.reduce_mean(tf.abs(bbox_true - bbox_pred)[anchors_positive])
    # add the two losses
    return conf_loss + loc_loss


def ssdlite(input_shape, n_classes, alpha=1.0):
    """Get SSDLite model and anchor boxes.

    Args:
        input_shape (list(integer)): (height, width).
        n_classes (integer): Number of classes.
        alpha (float): Width multiplier.
    Returns:
        ssdlite (tf.keras model): SSDLite model, with added loss.
        anchors (float array, [n_priors, 4]): All the anchor boxes
            ((x, y, w, h) within [0., 1.]).
    """
    input_layer = layers.Input(shape=(input_shape[0], input_shape[1], 3))
    # Base model
    base, l1, l2, l3, l4, l5, l6 = ssdlite_base(input_shape, alpha)
    # add class and location predictions
    prediction = detection_head(n_classes, l1, l2, l3, l4, l5, l6)
    # create model
    model = tf.keras.Model(inputs=input_layer, outputs=prediction)
    # calculate anchor boxes
    anchor_boxes = get_anchor_boxes(l1, l2, l3, l4, l5, l6)
    # return both
    return model, anchor_boxes
