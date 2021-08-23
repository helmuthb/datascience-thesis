# -*- coding: utf-8 -*-

"""Functions used as loss functions in training.
Smooth L1-loss as introduced by Faster R-CNN,
and log softmax loss as used in SSD are also
implemented here.
"""

import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def smooth_l1(labels, preds):
    """Smooth L1 Loss as introduced by Faster R-CNN.
    It takes the L1 loss if larger than 1, and otherwise
    the L2 loss.

    Args:
        labels (tf.Tensor): One tensor containing the ground truth.
        preds (tf.Tensor): Another tensor containing the prediction.
    Returns:
        smooth_l1 (tf.Tensor): Smooth L1 loss.
    """
    delta_y = tf.abs(preds - labels)
    delta_y2 = (preds - labels) ** 2
    delta_smooth = tf.where(
        tf.math.less(delta_y, 1.),
        0.5*delta_y2,
        delta_y-0.5)
    return tf.reduce_sum(delta_smooth, axis=-1)


def softmax_loss(labels, preds):
    """Softmax Loss as used in SSD.

    Args:
        label (tf.Tensor): One tensor containing the ground truth.
        preds (tf.Tensor): Another tensor containing the predictions.
    Returns:
        softmax_loss (tf.Tensor): Softmax loss.
    """
    # ensure no prediction is zero (for logarithm)
    preds = tf.math.maximum(preds, 1.e-15)
    # cross entropy  - parts to sum up
    entropy_parts = labels * tf.math.log(preds)
    return -1. * tf.reduce_sum(entropy_parts, axis=-1)


class SSDLoss(Loss):
    """SSD Loss for a training or validation batch.
    It takes the ground truth (class predictions and box adjustments) and
    the predictions for multiple boxes per instance.
    """
    def __init__(self, num_classes, reduction=Reduction.AUTO,
                 name='ssd_loss'):
        super(SSDLoss, self).__init__(reduction=reduction, name=name)
        self.num_classes = num_classes

    def get_config(self):
        config = super(SSDLoss, self).get_config()
        config['num_classes'] = self.num_classes
        return config

    def call(self, labels, logits):
        # classification loss: using softmax_cross_entropy_with_logits
        if True:
            cls_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels[:, :, 4:],
                logits=logits[:, :, 4:],
            )
        # calculate softmax of logits
        y_pred = tf.nn.softmax(logits)
        # classification loss: using softmax loss
        # cls_loss = softmax_loss(labels[:, :, :-4], y_pred[:, :, :-4])
        # localization loss: using smooth L1 loss
        loc_loss = smooth_l1(labels[:, :, :4], y_pred[:, :, :4])
        # which true items are negative (i.e. "background" class)?
        y_neg = labels[:, :, 4]
        # which true items are positive (i.e. a class != background)?
        y_pos = tf.reduce_sum(labels[:, :, 5:], axis=-1)
        # number of positives & negatives in the batch
        n_pos = tf.reduce_sum(y_pos)
        n_neg = tf.cast(tf.math.count_nonzero(y_neg*cls_loss), tf.float32)
        # how many negatives to "hard mine"?
        n_neg_hard = tf.cast(
            tf.minimum(n_neg, tf.maximum(3*n_pos, 3)),
            tf.int32)

        def neg_loss():
            # we now calculate the class prediction loss for negatives
            # and pick the top n_neg_hard in the batch
            loss_lin = tf.reshape(cls_loss * y_neg, [-1])
            vals, idxs = tf.nn.top_k(loss_lin, k=n_neg_hard)
            # we get the last value and use it to filter
            min_loss = vals[-1]
            mask = tf.logical_and(tf.cast(y_neg, tf.bool), cls_loss > min_loss)
            mask = tf.cast(mask, tf.float32)
            # for the negative loss there is no location loss so that's all
            return tf.reduce_sum(cls_loss * mask, axis=-1)

        def zero_loss():
            return tf.zeros([tf.shape(labels)[0]])

        neg_cls_loss = tf.cond(tf.equal(n_neg, 0), zero_loss, neg_loss)
        # now the positive loss sum - location and classification
        pos_loc_loss = tf.reduce_sum(loc_loss * y_pos, axis=-1)
        pos_cls_loss = tf.reduce_sum(cls_loss * y_pos, axis=-1)
        # we return the total loss
        loss = (neg_cls_loss + pos_loc_loss + pos_cls_loss)
        # ... relative to the number of positive boxes
        loss /= tf.maximum(1., n_pos)
        # ... and multiplied by the batch size
        return loss * tf.cast(tf.shape(labels)[0], tf.float32)
