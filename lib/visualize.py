# -*- coding: utf-8 -*-

"""Tools for visualiztion of inference and original images.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List


__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'

# Alternative colormap
COLORMAP2 = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

# Colormap for visualizations
COLORMAP = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]
]


def annotate_segmentation(img: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                          file_prefix: str):
    """Annotate an image with the detected & ground-truth segmentation.
    Three images are created:
    * A segmentation map of the ground truth.
    * A segmentation map of the prediction.
    * A delta image (black-green) of matches and discrepancies of
      ground truth and prediction. Areas of match are colored green,
      areas of discrepancies are colored black.
    In the future an overlay of the original image with the annotation
    is possible, therefore the original image is passed as a parameter.
    The annotations are rescaled to be of same size as the original
    (using nearest value).

    Args:
        img (np.ndarray(height, width, 3): Image to annotate.
        gt (np.ndarray(height, width)): Ground truth segmentation.
        pred (np.ndarray(height, width, n_classes)): Predicted segmentation.
        file_prefix (str): Prefix for files to be created.
    """
    # get image width / height / n_classes
    img_width = img.shape[1]
    img_height = img.shape[0]
    n_classes = pred.shape[2]
    # resize ground truth
    if gt.shape[1] != img_width or gt.shape[0] != img_height:
        gt = cv2.resize(
            src=gt,
            dsize=(img_width, img_height),
            interpolation=cv2.INTER_NEAREST)
    # resize prediction
    if pred.shape[1] != img_width or pred.shape[0] != img_height:
        pred = cv2.resize(
            src=pred,
            dsize=(img_width, img_height),
            interpolation=cv2.INTER_NEAREST)
    # get the index for each pixel in the prediction
    pred = np.argmax(pred, axis=2)
    # remove last index from ground truth
    # gt = np.squeeze(gt, axis=2)
    # add last index
    pred = np.expand_dims(pred, axis=2)
    # get differences image
    delta = np.zeros(img.shape, dtype="uint8")
    delta[np.where((pred == gt).all(axis=2))] = (0, 255, 0)
    # get colorized images
    gt_img = np.zeros(img.shape, dtype="uint8")
    pred_img = np.zeros(img.shape, dtype="uint8")
    for i in range(n_classes):
        gt_img[np.where((gt == i).all(axis=2))] = COLORMAP[i]
        pred_img[np.where((pred == i).all(axis=2))] = COLORMAP[i]
    # save annotations
    cv2.imwrite(file_prefix + "-delta.png", delta)
    cv2.imwrite(file_prefix + "-gt.png", gt_img)
    cv2.imwrite(file_prefix + "-pred.png", pred_img)


def annotate_boxes(img: np.ndarray,
                   b_cl: np.ndarray, b_sc: np.ndarray, b_xy: np.ndarray,
                   classes: List[str], file_name: str):
    """Annotate an image with the detected (or original) boxes.
    The resulting image is saved into the specified file name.

    Args:
        img (np.ndarray(width, height, 3): Image to annotate.
        b_cl (np.ndarray [n]): Classes corresponding to bounding boxes.
        b_sc (np.ndarray [n]): Class score for bounding boxes.
        b_xy (np.ndarray [n, 4]): x0/y0/x1/y1 bounding boxes in [0, 1].
        classes (list[str]): List of class names.
        file_name (str): Name of the file where the result is saved to.
    """
    # get image width / height
    img_width = img.shape[1]
    img_height = img.shape[0]
    # blow up bounding boxes
    boxes_xy = b_xy.copy()
    boxes_xy[:, 0] *= img_width
    boxes_xy[:, 1] *= img_height
    boxes_xy[:, 2] *= img_width
    boxes_xy[:, 3] *= img_height
    # draw image
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(img / 256.)
    ax = plt.gca()
    for i, box_xy in enumerate(boxes_xy):
        cl = b_cl[i]
        color = [x/256 for x in COLORMAP[cl]]
        if cl >= 0 and cl < len(classes):
            label = classes[cl]
        else:
            label = "unknown"
        # Score is optional
        if b_sc is not None:
            sc = b_sc[i].round(2)
            label = f"{label}: {sc}"
        # draw bounding box
        x0 = box_xy[0]
        y0 = box_xy[1]
        width = box_xy[2] - box_xy[0]
        height = box_xy[3] - box_xy[1]
        rect = plt.Rectangle((x0, y0), width, height, color=color,
                             fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, label, size='x-large', color='white',
                bbox={'facecolor': color, 'alpha': 1.0})
    # save annotated image
    plt.savefig(file_name)
    plt.close(fig)
