import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from lib.ssdlite import (
    add_ssdlite_features, detection_head, get_default_boxes_cw,
    ssdlite_base_layers)
from lib.deeplab import add_deeplab_features


def ssd_deeplab_model(size, n_det, n_seg):
    """
    """
    input_layer = tf.keras.layers.Input(
        shape=(size[0], size[1], 3))
    # base model
    base = MobileNetV2(
        input_tensor=input_layer,
        include_top=False)
    # add deeplab layers
    deeplab_output = add_deeplab_features(base, n_seg)
    # add SSDlite layers
    ext_base = add_ssdlite_features(base)
    l1, l2, l3, l4, l5, l6 = ssdlite_base_layers(ext_base)
    # add class and location predictions
    ssd_outputs = detection_head(n_det, l1, l2, l3, l4, l5, l6)
    # create models
    ssd_model = tf.keras.Model(
        inputs=input_layer,
        outputs=ssd_outputs
    )
    deeplab_model = tf.keras.Model(
        inputs=input_layer,
        outputs=deeplab_output
    )
    combined_outputs = (*ssd_outputs, deeplab_output)
    combined_model = tf.keras.Model(
        inputs=input_layer,
        outputs=combined_outputs)
    # calculate default boxes
    default_boxes_cw = get_default_boxes_cw(l1, l2, l3, l4, l5, l6)
    # return combined model and the defaults
    return combined_model, default_boxes_cw, base, deeplab_model, ssd_model


def loss_list(ssd_f, deeplab_f, ssd_only, deeplab_only):
    def _losses_combined(gt, pr):
        ssd_losses = ssd_f(gt[0], gt[1], pr[0], pr[1])
        deeplab_loss = deeplab_f(gt[2], pr[2])
        return (*ssd_losses, deeplab_loss)

    def _losses_ssd(gt, pr):
        return ssd_f(gt[0], gt[1], pr[0], pr[1])

    def _losses_deeplab(gt, pr):
        return (deeplab_f(gt, pr),)

    if ssd_only:
        return _losses_ssd
    elif deeplab_only:
        return _losses_deeplab
    else:
        return _losses_combined


def get_training_step(model, losses, weights, optimizer, ssd_only,
                      deeplab_only, alpha=5e-4):
    w = tf.convert_to_tensor(weights, dtype=tf.float32)
    a = tf.convert_to_tensor(alpha, dtype=tf.float32)

    @tf.function
    def _step1(img, gt):
        with tf.GradientTape() as tape:
            pr = model(img)
            ll = losses(gt, pr)
            loss = w[0] * ll[0]
            # weight decay
            weight_loss = [tf.nn.l2_loss(w) for w in model.trainable_variables]
            weight_loss = a * tf.reduce_sum(weight_loss)
            loss += weight_loss
        delta = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(delta, model.trainable_variables))
        return (loss, *ll)

    @tf.function
    def _step2(img, gt):
        with tf.GradientTape() as tape:
            pr = model(img)
            ll = losses(gt, pr)
            loss = w[0] * ll[0] + w[1] * ll[1]
            # weight decay
            weight_loss = [tf.nn.l2_loss(w) for w in model.trainable_variables]
            weight_loss = alpha * tf.reduce_sum(weight_loss)
            loss += weight_loss
        delta = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(delta, model.trainable_variables))
        return (loss, *ll)

    @tf.function
    def _step3(img, gt):
        with tf.GradientTape() as tape:
            pr = model(img)
            ll = losses(gt, pr)
            loss = w[0] * ll[0] + w[1] * ll[1] + w[2] * ll[2]
            # weight decay
            weight_loss = [tf.nn.l2_loss(w) for w in model.trainable_variables]
            weight_loss = alpha * tf.reduce_sum(weight_loss)
            loss += weight_loss
        delta = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(delta, model.trainable_variables))
        return (loss, *ll)

    if ssd_only:
        return _step2
    elif deeplab_only:
        return _step1
    else:
        return _step3
