from typing import Callable
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input as mnet2_prep)
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input as vgg16_prep)

from lib.ssd import (
    detection_heads, get_default_boxes_cw, ssd_base_outputs)
from lib.deeplab import add_deeplab_features


def ssd_deeplab_model(n_det: int, n_seg: int, config: dict) -> tuple:
    """
    """
    width = config['width']
    input_layer = tf.keras.layers.Input(shape=(width, width, 3))
    # base model
    if config['base'] == "MobileNetV2":
        base = MobileNetV2(input_tensor=input_layer, include_top=False)
        prep = mnet2_prep
    elif config['base'] == "VGG16":
        base = VGG16(input_tensor=input_layer, include_top=False)
        prep = vgg16_prep
    else:
        raise ValueError(f"Base model '{config['base']}' unknown")
    # provide values for n_det / n_seg to create working models
    if n_det == 0:
        n_det = n_seg
    elif n_seg == 0:
        n_seg = n_det
    # add deeplab layers
    deeplab_output = add_deeplab_features(base, n_seg, config)
    # add SSDlite layers
    ssd_outputs_raw = ssd_base_outputs(base, config)
    # add class and location predictions
    ssd_outputs = detection_heads(n_det, ssd_outputs_raw, config)
    # create models
    s_model = tf.keras.Model(inputs=input_layer, outputs=ssd_outputs)
    d_model = tf.keras.Model(inputs=input_layer, outputs=deeplab_output)
    combined_outputs = (*ssd_outputs, deeplab_output)
    c_model = tf.keras.Model(inputs=input_layer, outputs=combined_outputs)
    # calculate default boxes
    d_boxes_cw = get_default_boxes_cw(ssd_outputs_raw, config)
    # return combined model and the defaults
    return c_model, base, d_model, s_model, d_boxes_cw, prep


def loss_list(ssd_f: Callable, deeplab_f: Callable,
              n_det: int, n_seg: int) -> Callable:
    def _losses_combined(gt, pr):
        ssd_losses = ssd_f(gt[0], gt[1], pr[0], pr[1])
        deeplab_loss = deeplab_f(gt[2], pr[2])
        return (*ssd_losses, deeplab_loss)

    def _losses_ssd(gt, pr):
        return ssd_f(gt[0], gt[1], pr[0], pr[1])

    def _losses_deeplab(gt, pr):
        return (deeplab_f(gt, pr),)

    if n_seg == 0:
        return _losses_ssd
    elif n_det == 0:
        return _losses_deeplab
    else:
        return _losses_combined


def get_training_step(model: Model, losses: Callable, weights: list,
                      optimizer: Optimizer,
                      n_det: int, n_seg: int, alpha: float):
    w = tf.convert_to_tensor(weights, dtype=tf.float32)
    a = tf.convert_to_tensor(alpha, dtype=tf.float32)

    @tf.function
    def _step1(img, gt):
        with tf.GradientTape() as tape:
            pr = model(img, training=True)
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
            pr = model(img, training=True)
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
            pr = model(img, training=True)
            ll = losses(gt, pr)
            loss = w[0] * ll[0] + w[1] * ll[1] + w[2] * ll[2]
            # weight decay
            weight_loss = [tf.nn.l2_loss(w) for w in model.trainable_variables]
            weight_loss = alpha * tf.reduce_sum(weight_loss)
            loss += weight_loss
        delta = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(delta, model.trainable_variables))
        return (loss, *ll)

    if n_seg == 0:
        return _step2
    elif n_det == 0:
        return _step1
    else:
        return _step3
