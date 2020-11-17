import tensorflow as tf


def skip_small_boxes(bboxes, classes, threshold=0.001):
    """Skip boxes where area is smaller than threshold.

    Args:
        classes (tf.Tensor): Classes in the image.
        bboxes (tf.Tensor): Bounding boxes in the image.
    Returns:
        bboxes (tf.Tensor): Bounding boxes above threshold.
        classes (tf.Tensor): Classes, without small boxes.
    """
    # calculate area
    x0 = bboxes[..., 1]
    y0 = bboxes[..., 2]
    x1 = bboxes[..., 3]
    y1 = bboxes[..., 4]
    area = (x1-x0) * (y1-y0)
    # filter classes and boxes
    above = area > threshold
    # return filtered classes and boxes
    return bboxes[above], classes[above]


def preprocess_func(size, augmentation=False, anchors=None):
    """Get preprocessing function for images (and data - if available).

    The resulting function will expect image, and optionally
    bboxes and classes, and will return the same after resizing and scaling.
    Optionally it can also be augmented.
    In addition, if the ground truth is provided, it will also be filtered
    and the bounding box data will be prepared for comparison or training.

    Args:
        size (tuple(int)): Output size of the images.
        augmentation (boolean): Flag whether the dataset shall be augmented.
        anchors (np.ndarray [n_anchor, 4]): anchor boxes
    """

    def preprocess(image, bboxes=None, classes=None):
        """Preprocess image: resize, scale, filter small boxes.

        Args:
            image (tf.Tensor): Image data.
            bboxes (tf.Tensor): Bounding boxes per object.
            classes (tf.Tensor): Classes per object.
        """
        # resize image
        image = tf.image.resize(image, size)
        # optional augmentation
        if augmentation:
            # TODO: Augmentation
            pass
        # scale image to [0, 1] range
        image = tf.clip_by_value(image / 255, 0., 1.)
        if bboxes != None and classes != None:
            #
        # clip boxes (could be larger due to augmentation)
        bboxes = tf.clip_by_value(bboxes, 0., 1.)
        # filter very small boxes
        classes, bboxes = skip_small_boxes(classes, bboxes)
        # return preprocessed image & data
        return image, bboxes, classes

    return preprocess
