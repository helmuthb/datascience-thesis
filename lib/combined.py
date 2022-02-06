import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from lib.ssdlite import (
    add_ssdlite_features, detection_head, get_default_boxes_cwh,
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
    default_boxes_cwh = get_default_boxes_cwh(l1, l2, l3, l4, l5, l6)
    # return combined model and the defaults
    return combined_model, default_boxes_cwh, base, deeplab_model, ssd_model


def combined_losses(ssd_f, deeplab_f):
    def _losses(gt, pr):
        ssd_losses = ssd_f(gt[0], gt[1], pr[0], pr[1])
        deeplab_loss = deeplab_f(gt[2], pr[2])
        return (*ssd_losses, deeplab_loss)
    return _losses


@tf.function
def training_step(img, gt, model, losses, optimizer, weights):
    with tf.GradientTape() as g:
        pr = model(img)
        batch_losses = losses(gt, pr)
        loss = (weights[0] * (batch_losses[0] + batch_losses[1]) +
                weights[1] * batch_losses[2])
        # weight decay
        weight_loss = [tf.nn.l2_loss(w) for w in model.trainable_variables]
        weight_loss = 5e-4 * tf.reduce_sum(weight_loss)
        loss += weight_loss
    delta = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(delta, model.trainable_variables))
    return (loss, *batch_losses)


class CustomSchedule(LearningRateSchedule):
    """Custom schedule, based on the code in Transformer tutorial.
    https://www.tensorflow.org/text/tutorials/transformer#optimizer
    """
    def __init__(self, peak_rate, warmup_steps):
        super().__init__()
        self.peak_rate = peak_rate
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_steps ** -1.5

    def get_config(self):
        config = {}
        config['peak_rate'] = self.peak_rate
        config['warmup_steps'] = self.warmup_steps
        return config

    def __call__(self, step):
        a1 = tf.math.rsqrt(step)
        a2 = step * self.warmup_factor
        return self.peak_rate * tf.math.minimum(a1, a2)
