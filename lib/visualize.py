# -*- coding: utf-8 -*-

"""Tools for visualiztion of inference and original images.
"""

import math
import numpy as np
import cv2
import io
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


def get_colormap(n_items: int, as_int: bool) -> list:
    cm = plt.cm.get_cmap('tab20')
    loops = math.ceil(n_items / len(cm.colors))
    colors = cm.colors * loops
    if as_int:
        colors = [(256*r, 256*g, 256*b) for (r, g, b) in colors]
    return colors


def resize_like(img: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Resize an image so that it matches the size of the other image.
    """
    # get image width / height
    old_width = img.shape[1]
    old_height = img.shape[0]
    new_width = other.shape[1]
    new_height = other.shape[0]
    if old_width != new_width or old_height != new_height:
        img = cv2.resize(
            src=img,
            dsize=(new_width, new_height),
            interpolation=cv2.INTER_NEAREST)
    return img


def pred_classes(pred: np.ndarray) -> np.ndarray:
    """Return for each position the class with the highest confidence.
    """
    # get the index for each pixel in the prediction
    pred = np.argmax(pred, axis=2)
    # add last index
    pred = np.expand_dims(pred, axis=2)


def delta_image(img: np.ndarray, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """Annotate an image with coloring, showing if the prediction is correct.
    """
    # resize gt & pred to match img
    gt = resize_like(gt, other=img)
    pr = resize_like(pr, other=img)
    # get the index for each pixel in the prediction
    pr = pred_classes(pr)
    # get differences image
    delta = np.zeros(img.shape, dtype="uint8")
    delta[:, :] = (127, 0, 0)
    delta[np.where((pr == gt).all(axis=2))] = (0, 127, 0)
    # add red/green to half of the original image
    return delta + img/2.


def classes_image(cls_values: np.ndarray, n_classes: int) -> np.ndarray:
    """Get a colored image, where each class gets its own color.
    """
    # calculate target shape
    shape = (cls_values.shape[0], cls_values.shape[1], 3)
    # get colormap
    colormap = get_colormap(n_classes, as_int=True)
    # prepare output image
    out_img = np.zeros(shape, dtype="uint8")
    for i in range(n_classes):
        out_img[np.where((cls_values == i).all(axis=2))] = colormap[i]
    return out_img


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
        gt (np.ndarray(height, width, 1)): Ground truth segmentation.
        pred (np.ndarray(height, width, n_classes)): Predicted segmentation.
        file_prefix (str): Prefix for files to be created.
    """
    # get number of classes
    n_classes = pred.shape[2]
    # resize gt & pred to match img
    gt = resize_like(gt, other=img)
    pred = resize_like(pred, other=img)
    # get the index for each pixel in the prediction
    pred = pred_classes(pred)
    # get differences image
    delta = delta_image(img, gt, pred)
    # get colorized images
    gt_img = classes_image(gt, n_classes)
    pred_img = classes_image(pred, n_classes)
    # save annotations
    cv2.imwrite(file_prefix + "-delta.png", delta)
    cv2.imwrite(file_prefix + "-gt.png", gt_img)
    cv2.imwrite(file_prefix + "-pred.png", pred_img)


def boxes_image(img: np.ndarray, b_cl: np.ndarray, b_sc: np.ndarray,
                b_yx: np.ndarray, classes: List[str]) -> np.ndarray:
    """Annotate an image with the detected (or original) boxes.

    Args:
        img (np.ndarray(width, height, 3): Image to annotate.
        b_cl (np.ndarray [n]): Classes corresponding to bounding boxes.
        b_sc (np.ndarray [n]): Class score for bounding boxes.
        b_yx (np.ndarray [n, 4]): y0/x0/y1/x1 bounding boxes in [0, 1].
        classes (list[str]): List of class names.
    """
    # get image height / width
    img_height = img.shape[0]
    img_width = img.shape[1]
    colormap = get_colormap(len(classes), as_int=True)
    # blow up bounding boxes
    boxes_yx = b_yx.copy()
    boxes_yx[:, 0] *= img_height
    boxes_yx[:, 1] *= img_width
    boxes_yx[:, 2] *= img_height
    boxes_yx[:, 3] *= img_width
    # draw image
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(img / 256.)
    ax = plt.gca()
    for i, box_yx in enumerate(boxes_yx):
        cl = b_cl[i]
        if cl == 0:
            continue
        color = [x/256. for x in colormap[cl]]
        if cl >= 0 and cl < len(classes):
            label = classes[cl]
        else:
            label = "unknown"
        # Score is optional
        if b_sc is not None:
            sc = b_sc[i].round(2)
            label = f"{label}: {sc}"
        # draw bounding box
        y0 = box_yx[0]
        x0 = box_yx[1]
        height = box_yx[2] - box_yx[0]
        width = box_yx[3] - box_yx[1]
        rect = plt.Rectangle((x0, y0), width, height, color=color,
                             fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, label, size='x-large', color='white',
                bbox={'facecolor': color, 'alpha': 1.0})
    # return annotated image
    with io.BytesIO() as buff:
        plt.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    plt.close(fig)
    return data.reshape((int(h), int(w), -1))[:, :, :3]


def annotate_boxes(img: np.ndarray,
                   b_cl: np.ndarray, b_sc: np.ndarray, b_yx: np.ndarray,
                   classes: List[str], file_name: str):
    """Annotate an image with the detected (or original) boxes.
    The resulting image is saved into the specified file name.

    Args:
        img (np.ndarray(width, height, 3): Image to annotate.
        b_cl (np.ndarray [n]): Classes corresponding to bounding boxes.
        b_sc (np.ndarray [n]): Class score for bounding boxes.
        b_yx (np.ndarray [n, 4]): y0/x0/y1/x1 bounding boxes in [0, 1].
        classes (list[str]): List of class names.
        file_name (str): Name of the file where the result is saved to.
    """
    # get image height / width
    img_height = img.shape[0]
    img_width = img.shape[1]
    # colormap = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
    colormap = get_colormap(len(classes), as_int=True)
    # blow up bounding boxes
    boxes_yx = b_yx.copy()
    boxes_yx[:, 0] *= img_height
    boxes_yx[:, 1] *= img_width
    boxes_yx[:, 2] *= img_height
    boxes_yx[:, 3] *= img_width
    # draw image
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(img / 256.)
    ax = plt.gca()
    for i, box_yx in enumerate(boxes_yx):
        cl = b_cl[i]
        color = [x/256. for x in colormap[cl]]
        if cl >= 0 and cl < len(classes):
            label = classes[cl]
        else:
            label = "unknown"
        # Score is optional
        if b_sc is not None:
            sc = b_sc[i].round(2)
            label = f"{label}: {sc}"
        # draw bounding box
        y0 = box_yx[0]
        x0 = box_yx[1]
        height = box_yx[2] - box_yx[0]
        width = box_yx[3] - box_yx[1]
        rect = plt.Rectangle((x0, y0), width, height, color=color,
                             fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, label, size='x-large', color='white',
                bbox={'facecolor': color, 'alpha': 1.0})
    # save annotated image
    plt.savefig(file_name)
    plt.close(fig)
