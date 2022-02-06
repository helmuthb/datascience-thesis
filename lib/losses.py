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
    def __init__(self, neg_ratio):
        self.neg_ratio = neg_ratio

    def negative_mining(self, gt_clss, pr_conf):
        """Calculate indexes with hard-negatives.
        """
        # calculate classification losses (without reduction)
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
            )
        loss = cross_entropy(gt_clss, pr_conf)
        # positive indexes: where we have an object in the gt
        pos_idx = gt_clss > 0
        # number of positive indexes per image
        num_pos = tf.reduce_sum(tf.cast(pos_idx, tf.int32), axis=1)
        # number of negatives looked for
        num_neg = num_pos * self.neg_ratio

        # find the top losses
        rank = tf.argsort(loss, axis=1, direction='DESCENDING')
        rank = tf.argsort(rank, axis=1)
        neg_idx = rank < tf.expand_dims(num_neg, 1)

        # return positive & negative indexes
        return pos_idx, neg_idx

    def __call__(self, gt_clss, gt_locs, pr_conf, pr_locs):
        """"Calculate the SSD losses from the predicted logits.
        """
        # hard-mining of negatives
        pos_idx, neg_idx = self.negative_mining(gt_clss, pr_conf)
        conf_idx = tf.math.logical_or(pos_idx, neg_idx)
        # classification loss
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='sum'
        )
        conf_loss = cross_entropy(gt_clss[conf_idx], pr_conf[conf_idx])
        # localization loss
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
        locs_loss = smooth_l1_loss(gt_locs[pos_idx], pr_locs[pos_idx])
        # return losses adjusted by num_pos
        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))
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
