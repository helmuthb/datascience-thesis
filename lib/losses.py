# -*- coding: utf-8 -*-

"""Functions used as loss functions in training.
Smooth L1-loss as introduced by Faster R-CNN,
and log softmax loss as used in SSD are also
implemented here.
"""

import tensorflow as tf
from tensorflow.keras.losses import (
    Loss, sparse_categorical_crossentropy
)

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


class SSDLosses():
    """SSD Loss components.
    """
    def __init__(self, neg_ratio, focal_weight=0.5, box_weight=0.5,
                 epsilon=1e-7, gamma=2., alpha=1.):
        self.neg_ratio = tf.constant(neg_ratio)
        self.focal_weight = tf.constant(focal_weight)
        self.box_weight = tf.constant(box_weight)
        self.epsilon = tf.constant(epsilon)
        self.gamma = tf.constant(gamma)
        self.alpha = tf.constant(alpha)

    def focal_loss(self, gt_cl, gt_cl2, pr_conf):
        # eps = K.epsilon()
        pr_conf = tf.clip_by_value(
            tf.nn.softmax(pr_conf),
            self.epsilon,
            1.-self.epsilon)
        # pr_conf = K.clip(pr_conf, eps, 1.-eps)
        gt_conf_int = tf.one_hot(gt_cl2, tf.shape(pr_conf)[-1], dtype=tf.int32)
        pt = tf.where(
            tf.equal(1, gt_conf_int),
            pr_conf,
            1-pr_conf
        )
        loss = -tf.math.pow(1.-pt, self.gamma) * tf.math.log(pt)
        # set the loss to 0 zero if neutral
        loss = tf.where(
            tf.expand_dims(tf.less(gt_cl, 0), axis=-1),
            tf.zeros_like(loss),
            loss
        )
        return self.alpha * tf.reduce_sum(loss)

    def _neg_mining(self, gt_cl, gt_cl2, pr_conf, num_pos_img):
        """Calculate indexes with hard-negatives.
        """
        # calculate classification losses (without reduction)
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )
        loss = cross_entropy(gt_cl2, pr_conf)
        # set the loss to 0 zero if neutral
        loss = tf.where(
            tf.less(gt_cl, 0),
            tf.zeros_like(loss),
            loss
        )
        # number of negatives looked for
        num_neg = num_pos_img * self.neg_ratio

        # find the top losses
        rank = tf.argsort(loss, axis=1, direction='DESCENDING')
        rank = tf.argsort(rank, axis=1)
        neg_idx = rank < tf.expand_dims(num_neg, 1)

        # return hard-mined negative indexes
        return neg_idx

    @tf.function
    def __call__(self, gt_cl, gt_locs, pr_conf, pr_locs):
        """"Calculate the SSD losses from the predicted logits.
        """
        # hard-mining of negatives
        # gt_cls2 = tf.where(
        #     tf.less(gt_clss, 0),
        #     tf.zeros_like(gt_clss),
        #     gt_clss
        # )
        gt_cl2 = tf.minimum(gt_cl, tf.zeros_like(gt_cl))
        # positive indexes: where we have an object in the gt
        pos_idx = gt_cl > 0

        # focal loss
        conf_loss_focal = self.focal_loss(gt_cl, gt_cl2, pr_conf)

        # box_loss
        # number of positive indexes per image
        num_pos_img = tf.reduce_sum(tf.cast(pos_idx, tf.int32), axis=1)
        neg_idx = self._neg_mining(gt_cl, gt_cl2, pr_conf, num_pos_img)
        conf_idx = tf.math.logical_or(pos_idx, neg_idx)
        # classification loss
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='sum'
        )
        conf_loss_box = cross_entropy(gt_cl2[conf_idx], pr_conf[conf_idx])

        conf_loss = (self.box_weight * conf_loss_box +
                     self.focal_weight * conf_loss_focal)
        num_pos = tf.cast(tf.reduce_sum(num_pos_img), dtype=tf.float32)

        # localization loss
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
        locs_loss = smooth_l1_loss(gt_locs[pos_idx], pr_locs[pos_idx])

        # return losses adjusted by num_pos
        return conf_loss/num_pos, locs_loss/num_pos


class DeeplabLoss(Loss):
    """DeepLab Loss used for training or validation batch.
    """
    def __init__(self, name='deeplab_loss', **kwargs):
        super().__init__(
            name=name,
            **kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, labels, logits):
        """"Calculate the DeepLab loss from the predicted logits.
        """
        return sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
