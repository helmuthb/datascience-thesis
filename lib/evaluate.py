# -*- coding: utf-8 -*-

"""Functions for evaluation of segmentation and object detection.
"""

import numpy as np


__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def segmentation_miou(gt: np.ndarray, pred: np.ndarray):
    """Calculate the mean intersection over union (miou) for segmentation.
    The two arrays (ground truth and prediction) have to have the
    same size and dimensions.

    Args:
        gt (np.ndarray(height, width)): Ground truth segmentation.
        pred (np.ndarray(height, width, n_classes)): Predicted segmentation.
    """
    # Number of classes
    n_classes = pred.shape[2]
    # Assign a class for each prediction pixel
    pred = np.argmax(pred, axis=2)
    # Prepare: match all values larger than n_classes to 0
    gt = np.where(gt > n_classes, 0, gt)
    pred = np.where(pred > n_classes, 0, pred)
    # Reshape to linear array
    gt = gt.reshape((-1,))
    pred = pred.reshape((-1,))
    # Frequency counts, both gt and pred, for each class
    fc_gt = np.bincount(gt, minlength=n_classes)
    fc_pred = np.bincount(pred, minlength=n_classes)
    # Category matrix: combination index of gt and pred
    cat = gt * n_classes + pred
    # Frequency count of category matrix
    fc_cat = np.bincount(cat, minlength=n_classes*n_classes)
    fc_cat = fc_cat.reshape((n_classes, n_classes))
    # Diagonal of category matrix = intersection
    i = np.diagonal(fc_cat)
    # Union? ground truth + prediction - intersection
    u = fc_gt + fc_pred - i
    # Individual IoU values
    iou = i / u
    # Return mean IoU
    return np.nanmean(iou)
