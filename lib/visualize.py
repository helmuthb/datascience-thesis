# -*- coding: utf-8 -*-

"""Tools for visualiztion of inference and original images.
"""

import numpy as np
from matplotlib import pyplot as plt


__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def annotate_detection(img: np.ndarray, b_xy: np.ndarray, b_cl: np.ndarray,
                       b_sc: np.ndarray, classes: list[str], file_name: str):
    """Annotate an image with the detected (or original) boxes.
    The resulting image is saved into the specified file name.

    Args:
        img (np.ndarray(width, height, 3): Image to annotate.
        b_xy (np.ndarray [n, 4]): x0/y0/x1/y1 bounding boxes in [0, 1].
        b_cl (np.ndarray [n]): Classes corresponding to bounding boxes.
        b_sc (np.ndarray [n]): Class score for bounding boxes.
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
    # colors to be used for annotation
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # draw image
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(img / 256.)
    ax = plt.gca()
    for i, box_xy in enumerate(boxes_xy):
        cl = b_cl[i]
        color = colors[cl]
        # Score is optional
        if b_sc:
            sc = b_sc[i]
            label = f"{classes[cl]}: {sc:.2f}"
        else:
            label = classes[cl]
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
