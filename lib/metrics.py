from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU, Metric

from lib.np_bbox_utils import BBoxUtils, iou_xy

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


class MeanIoUMetric(MeanIoU):
    """Wrapper over the Keras MeanIoU metric using class scores.
    This metric uses the scores before chosing the maximum as input
    for calculating the mean intersection over union metric.
    """
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        """Create the metric. It uses the same arguments as the Keras
        provided metric. The name is by default set to 'mean_iou' for
        a short and clearly readible display.
        """
        super().__init__(num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """This function is called by Keras whenever a batch has
        been processed to update the internal statistics of the metric.
        It calculates the class with maximum score and uses it as input
        to the Keras MeanIoU metric.
        """
        return super().update_state(
            tf.identity(y_true),
            tf.argmax(y_pred, axis=-1),
            sample_weight=sample_weight)


class MeanAveragePrecisionMetric(Metric):
    """Keras implementation of mean average precision for object detection.
    """
    def __init__(self, num_classes, default_boxes_cwh, iou_threshold=0.5,
                 name='map', **kwargs):
        """Create a metric object.
        It requires the number of detection classes to be used and the set
        of default boxes.

        Args:
            num_classes (int): Number of detection classes in the model.
            default_boxes_cwh (np.ndarray [n1, 4]): List of default boxes.
            iou_threshold (float): Minimum intersection over union for
                considering a prediction a match for a ground truth box.
            name (string): Name of the metric object, by default set to
                'map' for a concise display in the logs.
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.default_boxes_cwh = default_boxes_cwh
        self.iou_threshold = iou_threshold
        self.bbox_util = BBoxUtils(num_classes, np.array(default_boxes_cwh))
        # collect the predictions: score value if matched, 0 if not
        # organized as dict with the class as index
        self.match_scores = defaultdict(list)
        # collect the predictions: score value if not matched, 0 if matched
        # organized as dict with the class as index
        self.nomatch_scores = defaultdict(list)
        # collect the ground truth: number of ground truth boxes
        # organized as dict with the class as index
        self.gt_count = defaultdict(lambda: 0)
        # collect the relevant classes
        self.classes = set()

    def get_config(self):
        """Get the config dictionary for a MeanAveragePrecision metric object.
        This method is implemented to support serializing of a model
        using this metric.
        """
        config = super().get_config()
        config['num_classes'] = self.num_classes
        config['default_boxes_cwh'] = self.default_boxes_cwh
        config['iou_threshold'] = self.iou_threshold
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state counting up for the mean average precision.

        Args:
            y_true (tf.Tensor): Ground truth values, consisting of default
                boxes adjusted for the real box locations.
            y_pred (tf.Tensor): Predicted adjustments of the default boxes.
            sample_weight: not supported parameter. For the metric all
                samples are weighted equally.
        """
        # inner function using numpy arguments
        def numpy_update_state(y_true, y_pred):
            # loop through batch items
            n_batch = y_true.shape[0]
            for i in range(n_batch):
                y_t = y_true[i, ...]
                y_p = y_pred[i, ...]
                # 1. get boxes from gt and predictions
                t_xy, t_cl, _ = self.bbox_util.pred_to_boxes(y_t)
                p_xy, p_cl, sc = self.bbox_util.pred_to_boxes(y_p)
                # no predictions?
                if len(p_xy) == 0:
                    for cl in set(t_cl):
                        self.gt_count[cl] += len(t_xy[t_cl == cl])
                    continue
                # 2. match gt boxes with predictions
                # Find for each prediction box the best-matching gt box
                # Classes: as we only look at precision, we only care about
                # predicted classes
                # classes = set(p_cl).union(set(t_cl))
                classes = set(p_cl)
                # add to status
                self.classes = self.classes.union(classes)
                for cl in classes:
                    # mask for this class
                    cl_t_mask = (t_cl == cl)
                    cl_p_mask = (p_cl == cl)
                    # true & pred boxes
                    cl_t_xy = t_xy[cl_t_mask]
                    cl_p_xy = p_xy[cl_p_mask]
                    # pred scores
                    cl_sc = sc[cl_p_mask]
                    # number of ground-truth boxes for this class
                    self.gt_count[cl] += len(cl_t_xy)
                    # no prediction boxes?
                    if len(cl_p_xy) == 0:
                        continue
                    # no ground-truth boxes? we will add all
                    # predicted boxes to the unmatched ones
                    if len(cl_t_xy) == 0:
                        self.nomatch_scores[cl] += cl_sc.tolist()
                        continue
                    # get IoU value for box combinations
                    iou = iou_xy(cl_t_xy, cl_p_xy, pairwise=False)
                    # filter - which IoU-values are above threshold?
                    iou_flag = (iou > self.iou_threshold)
                    # look in each row for at least one match
                    iou_flag = np.max(iou_flag, axis=0)
                    # masked matched scores (0 if not matched)
                    match_scores = (iou_flag * cl_sc).tolist()
                    # masked unmatched scores (0 if matched)
                    nomatch_scores = ((1 - iou_flag) * cl_sc).tolist()
                    # add to common structure
                    self.match_scores[cl] += match_scores
                    self.nomatch_scores[cl] += nomatch_scores
            return True
        # call using numpy_function
        tf.numpy_function(numpy_update_state, [y_true, y_pred], tf.bool)

    def reset_state(self):
        """Reset the common structure to its initial value.
        """
        self.match_scores = defaultdict(list)
        self.nomatch_scores = defaultdict(list)
        self.gt_count = defaultdict(lambda: 0)
        self.classes = set()

    def result(self):
        """Calculate the scalar metric.
        """
        def numpy_result():
            # first call - no classes
            if len(self.classes) == 0:
                return 0.
            # collected average precision per class so far
            avg_precision = []
            # loop through the relevant classes
            for cl in self.classes:
                # convert into numpy arrays
                match_scores = np.array(self.match_scores[cl])
                nomatch_scores = np.array(self.nomatch_scores[cl])
                all_scores = np.array(
                    self.match_scores[cl] + self.nomatch_scores[cl])
                # get the unique, sorted scores for integration
                points = np.unique(all_scores)
                # start position = 0
                pos = 0.
                # average so far
                avg = 0.
                # integrate over points
                for point in points:
                    # width of the segment
                    width = point - pos
                    if (width == 0.):
                        continue
                    # count tp
                    tp = np.sum(match_scores > pos)
                    # count fp
                    fp = np.sum(nomatch_scores > pos)
                    # continue to the next position
                    pos = point
                    if tp+fp == 0:
                        # here we define the precision as 0
                        continue
                    # add to average precision
                    avg += width * (tp / (tp+fp))
                # add average to list
                avg_precision.append(avg)
            # calculate mean average precision
            return np.mean(avg_precision)
        # call using numpy_function
        # tf.numpy_function(numpy_update_state, [y_true, y_pred], tf.bool)
        return tf.numpy_function(numpy_result, [], tf.float64)
