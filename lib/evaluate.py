# -*- coding: utf-8 -*-

"""Functions for evaluation of segmentation and object detection.
"""

from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.np_bbox_utils import iou_xy


__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


class SegEval(object):
    """Class for calculating metrics related to image segmentation.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.cm = np.zeros(
            (num_classes, num_classes), dtype=float)
        self.cm_size = num_classes * num_classes

    def evaluate_sample(self, gt: np.ndarray, pr: np.ndarray):
        """Add the statistics for another sample to the evaluator.

        Args:
            gt (np.ndarray(height, width)): Ground truth segmentation.
            pr (np.ndarray(height, width, n_classes)): Predicted segmentation.
        """
        # resize prediction to match ground truth
        if (pr.shape[0] != gt.shape[0]) or (pr.shape[1] != gt.shape[1]):
            pr = cv2.resize(
                src=pr,
                dsize=(gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        # get the estimated class for each prediction
        if len(pr.shape) > 2:
            pr = np.argmax(pr, axis=2)
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        # get mask of valid labels
        valid = (gt >= 0) & (gt < self.num_classes)
        # assign label to each point: gt*num_classes + pr
        label = self.num_classes * gt[valid].astype(int) + pr[valid]
        # create counts for confusion matrix
        count = np.bincount(label, minlength=self.cm_size)
        # reshape for confusion matrix
        image_cm = count.reshape(self.num_classes, self.num_classes)
        # add to the overall confusion matrix
        self.cm += image_cm

    def pixel_accuracy(self):
        """Calculate the pixel accuracy after evaluating images.
        """
        return np.diag(self.cm).sum() / self.cm.sum()

    def class_accuracy(self):
        """Calculate the class accuracy after evaluating images.
        """
        class_acc = np.diag(self.cm) / self.cm.sum(axis=1)
        # division by zero? set 0
        class_acc[np.isnan(class_acc)] = 0
        return class_acc

    def mean_accuracy(self):
        """Calculate the mean class accuracy after evaluating images.
        """
        return np.mean(self.class_accuracy())

    def class_iou(self):
        """Calculate the class-wise IoU after evaluating images.
        """
        # intersection: diagonal
        i = np.diag(self.cm)
        # union: ground truth + prediction - intersection
        u = np.sum(self.cm, axis=0) + np.sum(self.cm, axis=1) - i
        # intersection over union
        iou = i / u
        iou[np.isnan(iou)] = 0
        return iou

    def class_dice_coefficient(self):
        """Calculate the class-wise Dice Coeffciient after evaluaing images.
        """
        # intersection: diagonal
        i = np.diag(self.cm)
        # union: ground truth + prediction - intersection
        u = np.sum(self.cm, axis=0) + np.sum(self.cm, axis=1) - i
        # dice coefficient
        dc = 2*i / (u+i)
        dc[np.isnan(dc)] = 0
        return dc

    def mean_dice_coefficient(self):
        """Calculate the mean dice coefficient after evaluating images.
        """
        return np.mean(self.class_dice_coefficient())

    def mean_iou(self):
        """Calculate the mean IoU after evaluating images.
        """
        return np.mean(self.class_iou())

    def fw_iou(self):
        """Calculate frequency-weighted IoU.
        """
        freq = np.sum(self.cm, axis=1) / np.sum(self.cm)
        freq[np.isnan(freq)] = 0
        iou = self.class_iou()
        return (freq*iou).sum()

    def _classnames(self, class_names):
        """Check the list of classnames and add 'background' if needed.
        """
        if len(class_names) < self.num_classes:
            return ["background"] + class_names
        else:
            return class_names

    def plot_iou(self, class_names: List[str], fname: str):
        """Plot the IoU values, descending.
        """
        fig = plt.figure()
        # adjust classnames if needed
        class_names = self._classnames(class_names)
        # calculate class-wise iou & mean iou
        class_iou = self.class_iou()
        mean_iou = self.mean_iou()
        # title of the plot
        title = f"Mean IoU: {mean_iou:.2%}"
        # sort by iou value
        idxs = np.argsort(class_iou)
        iou_sorted = [class_iou[i] for i in idxs]
        cls_sorted = [class_names[i] for i in idxs]
        # create bar plot
        ypos = range(self.num_classes)
        plt.barh(ypos, iou_sorted)
        # set plot title
        plt.title(title)
        # add class names to the left
        plt.yticks(ypos, cls_sorted, size='small')
        # add label for x-axis
        plt.xlabel("Intersection over Union")
        # make figure larger
        fig.set_figwidth(fig.get_figwidth() + 1)
        fig.tight_layout()
        # save plot
        plt.savefig(fname)
        plt.close()

    def plot_cm(self, class_names: List[str], fname: str):
        """Plot the confusion matrix.
        """
        fig = plt.figure()
        # adjust classnames if needed
        class_names = self._classnames(class_names)
        # calculate class-wise iou & mean iou
        class_iou = self.class_iou()
        mean_iou = self.mean_iou()
        # normalize the confusion matrix
        cm = self.cm.astype(float)
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0
        # title of the plot
        title = f"Mean IoU: {mean_iou:.2%}"
        # sort by iou value (inverse)
        idxs = np.argsort(-class_iou)
        cls_sorted = [class_names[i] for i in idxs]
        # sort the confusion matrix
        cm2 = np.zeros((self.num_classes, self.num_classes), dtype=float)
        for i, x in enumerate(idxs):
            for j, y in enumerate(idxs):
                cm2[i, j] = cm[x, y]
        # plot background colors
        plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        # plot labels
        marks = range(self.num_classes)
        plt.xticks(marks, cls_sorted, rotation=90, size='small')
        plt.yticks(marks, cls_sorted, size='small')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        # define threshold for label color
        threshold = cm.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                v = cm2[i, j]
                vs = f"{v:.2f}"
                c = "black" if v < threshold else "yellow"
                plt.text(j, i, vs, ha='center', color=c, size='x-small')
        # make figure larger
        fig.set_figheight(fig.get_figheight() + 2)
        fig.tight_layout()
        plt.savefig(fname)
        plt.close()


class DetEval(object):
    """Class for calculating metrics related to object detection.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.gt_count = np.zeros(num_classes, dtype=int)
        # store true positivies as list of lists (by class)
        # each entry is a tuple (iou, score)
        self._tp = [[] for _ in range(num_classes)]
        # store false positives as list of lists (by class)
        # each entry is the score value
        self._fp = [[] for _ in range(num_classes)]
        # prepared numpy-arrays of tp / fp values
        self.tp = None
        self.fp = None
        # data type for tp / fp
        self.tp_type = np.dtype([
            ('class', 'i4'), ('iou', 'f4'), ('score', 'f4')
        ])
        self.fp_type = np.dtype([
            ('class', 'i4'), ('score', 'f4')
        ])

    def evaluate_sample(self, gt_box: np.ndarray, gt_cls: np.ndarray,
                        pr_box: np.ndarray, pr_cls: np.ndarray,
                        pr_scr: np.ndarray):
        """Add the statistics for another sample to the evaluator.

        Args:
            gt_box (np.ndarray(n1, 4)): Ground truth boxes.
            gt_cls (np.ndarray(n1)): Ground truth classes.
            pr_box (np.ndarray(n2, 4)): Predicted object boxes.
            pr_cls (np.ndarray(n2)): Predicted object classes.
            pr_scr (np.ndarray(n2)): Predicted object scores.
        """
        # reset the prepared numpy-arrays
        self.tp = None
        self.fp = None
        # get set of relevant classes
        gt_classes = set(gt_cls.tolist())
        pr_classes = set(pr_cls.tolist())
        # sort predictions by score (descending)
        idxs = np.argsort(-pr_scr)
        pr_scr = pr_scr[idxs]
        pr_box = pr_box[idxs]
        pr_cls = pr_cls[idxs]
        # loop through all relevant classes
        for c in gt_classes.union(pr_classes):
            # ignore invalid classes
            if c < 0 or c >= self.num_classes:
                continue
            # relevant groundtruth boxes
            gt_c_box = gt_box[gt_cls == c]
            # add to number of groundtruth boxes
            self.gt_count[c] += len(gt_c_box)
            # iterate through all predictions for this class
            pr_mask = (pr_cls == c)
            pr_c_box = pr_box[pr_mask]
            pr_c_scr = pr_scr[pr_mask]
            if len(gt_c_box) > 0:
                # set of matched ground boxes so far
                gt_matched = set()
                # calculate intersections over union
                iou = iou_xy(pr_c_box, gt_c_box, pairwise=False)
                for i, scr in enumerate(pr_c_scr):
                    # find ground truth with highest IoU
                    gt_idx = np.argmax(iou[i, :])
                    max_iou = iou[i, gt_idx]
                    if gt_idx not in gt_matched:
                        # add it to the true positives for the class
                        self._tp[c].append((max_iou, scr))
                        # add ground truth to matched boxes
                        gt_matched.add(gt_idx)
                    else:
                        self._fp[c].append(scr)
            else:
                # no ground truth for this class - all predictions are
                # false positives
                for scr in pr_c_scr:
                    self._fp[c].append(scr)

    def _prepare(self):
        """Create prepared numpy-arrays out of the tp/fp lists of lists.
        """
        if self.tp is None:
            tp_len = sum([len(x) for x in self._tp])
            self.tp = np.zeros(tp_len, dtype=self.tp_type)
            i = 0
            for c, c_l in enumerate(self._tp):
                for x in c_l:
                    self.tp[i] = (c, x[0], x[1])
                    i += 1
        if self.fp is None:
            fp_len = sum([len(x) for x in self._fp])
            self.fp = np.zeros(fp_len, dtype=self.fp_type)
            i = 0
            for c, c_l in enumerate(self._fp):
                for x in c_l:
                    self.fp[i] = (c, x)
                    i += 1

    def _all_scores(self, cls, min_iou=0.5):
        """Return a numpy array of all relevant score values of a class.
        """
        self._prepare()
        tp_mask = (self.tp['iou'] >= min_iou) & (self.tp['class'] == cls)
        tp_score = self.tp['score'][tp_mask]
        fp_mask = self.fp['class'] == cls
        fp_score = self.fp['score'][fp_mask]
        return np.unique(np.concatenate((tp_score, fp_score)))

    def counts(self, cls, min_score, min_iou=0.5):
        """Return the number of true positives and false positives.
        These numbers depend on the iou threshold and the score threshold.
        """
        self._prepare()
        # true positives: score and iou is large enough
        mask1 = ((self.tp['score'] >= min_score) &
                 (self.tp['iou'] >= min_iou) &
                 (self.tp['class'] == cls))
        tp = len(self.tp[mask1])
        # false positives, part 1: iou is too small
        mask2 = ((self.tp['score'] >= min_score) &
                 (self.tp['iou'] < min_iou) &
                 (self.tp['class'] == cls))
        fp = len(self.tp[mask2])
        # false positivies, part 2: in the list of unmatches predictions
        mask3 = (self.fp['score'] >= min_score) & (self.fp['class'] == cls)
        fp += len(self.fp[mask3])
        # false negatives: number of ground truth boxes not detected
        # this is number of gt boxes minus number of true positives
        fn = self.gt_count[cls] - tp
        return tp, fp, fn

    def precision_recall(self, cls, min_score, min_iou=0.5):
        """Calculate precision & recall of a class for specific thresholds.
        """
        tp, fp, fn = self.counts(cls, min_score, min_iou)
        precision = tp/(tp+fp) if tp+fp > 0 else 1.
        recall = tp/(tp+fn) if tp+fn > 0 else 1.
        return precision, recall

    def _all_precision_recall(self, cls, min_iou=0.5):
        """Return all precision/recalls of a class for a threshold IoU value.
        """
        # start point: score = 0.
        p, r = self.precision_recall(cls, 0., min_iou)
        ret = [(p, r)]
        l_p = p
        l_r = r
        for sc in self._all_scores(cls, min_iou):
            p, r = self.precision_recall(cls, sc, min_iou)
            if p != l_p or r != l_r:
                ret.append((p, r))
            l_p = p
            l_r = r
        # final point: score = 1.
        p, r = self.precision_recall(cls, 1., min_iou)
        if p != l_p or r != l_r:
            ret.append((p, r))
        # convert into a numpy array
        ret = np.array(ret, dtype=float)
        return ret

    def average_precision_recall(self, cls, min_iou=0.5):
        """Calculate average precision & recall for a class.
        """
        # precision-recall values
        prec_rec = self._all_precision_recall(cls, min_iou)
        # sort by recall (for average precision)
        idxs = np.argsort(prec_rec[:, 1])
        avg_precision = 0.
        l_p = None
        l_r = None
        for p, r in prec_rec[idxs, :]:
            if l_p is None or l_r is None:
                l_p = p
                l_r = r
                continue
            # area of the shape: (average height) * width
            precision = (l_p + p) / 2.
            delta_recall = r - l_r
            avg_precision += precision * delta_recall
            # store previous values for loop
            l_p = p
            l_r = r
        # sort by precision (for average recall)
        idxs = np.argsort(prec_rec[:, 0])
        avg_recall = 0.
        l_p = None
        l_r = None
        for p, r in prec_rec[idxs, :]:
            if l_p is None or l_r is None:
                l_p = p
                l_r = r
                continue
            # area of the shape: (average height) * width
            recall = (r + l_r) / 2.
            delta_precision = p - l_p
            avg_recall += recall * delta_precision
            # store previous values for loop
            l_p = p
            l_r = r
        return avg_precision, avg_recall

    def mean_average_precision_recall(self, min_iou=0.5):
        """Calculate mean average precision & recall over all classes.
        """
        # add up all precision values
        precision_sum = 0.
        recall_sum = 0.
        for cls in range(1, self.num_classes):
            p, r = self.average_precision_recall(cls, min_iou)
            precision_sum += p
            recall_sum += r
        # n reduced by 1 since we ignored background
        n = self.num_classes - 1
        return precision_sum/n, recall_sum/n

    def plot_precision_recall_curves(
            self, cls: int, cname: str, ious: List[float], fname: str):
        """Plot the precision-recall curves for a specific class.
        """
        fig = plt.figure()
        for iou in ious:
            prec_recs = self._all_precision_recall(cls, min_iou=iou)
            # sort by recall (for nice plot)
            # only one point? don't draw anything
            if len(prec_recs) < 2:
                continue
            idxs = np.argsort(prec_recs[:, 1])
            prec = prec_recs[idxs, 0]
            recs = prec_recs[idxs, 1]
            plt.plot(recs, prec, label=f"IoU={iou:.0%}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall ({cname})")
        plt.legend()
        fig.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _classnames(self, class_names: List[str]):
        """Check the list of classnames and add 'background' if needed.
        """
        if len(class_names) < self.num_classes:
            return ["background"] + class_names
        else:
            return class_names

    def plot_precision(self, class_names: List[str], fname: str, min_iou=0.5):
        """Show a plot of detection precision for all classes.
        """
        fig = plt.figure()
        # adjust classnames if needed, and skip background
        class_names = self._classnames(class_names)[1:]
        prec = []
        for cls in range(1, self.num_classes):
            # calculate class-wise precision-recall values
            prec_recs = self.average_precision_recall(cls, min_iou)
            prec.append(prec_recs[0])
        # title of the plot
        title = f"Precision (min IoU={min_iou:.0%})"
        # sort by precision value
        idxs = np.argsort(prec)
        prec_sorted = [prec[i] for i in idxs]
        cls_sorted = [class_names[i] for i in idxs]
        # create bar plot
        ypos = range(self.num_classes - 1)
        plt.barh(ypos, prec_sorted)
        # set plot title
        plt.title(title)
        # add class names to the left
        plt.yticks(ypos, cls_sorted, size='small')
        # add label for x-axis
        plt.xlabel("Precision")
        # make figure larger
        fig.set_figwidth(fig.get_figwidth() + 1)
        fig.tight_layout()
        # save plot
        plt.savefig(fname)
        plt.close()

    def plot_recall(self, class_names: List[str], fname: str, min_iou=0.5):
        """Show a plot of detection recall for all classes.
        """
        fig = plt.figure()
        # adjust classnames if needed, and skip background
        class_names = self._classnames(class_names)[1:]
        recs = []
        for cls in range(1, self.num_classes):
            # calculate class-wise precision-recall values
            prec_recs = self.average_precision_recall(cls, min_iou)
            recs.append(prec_recs[1])
        # title of the plot
        title = f"Recall (min IoU={min_iou:.0%})"
        # sort by recall value
        idxs = np.argsort(recs)
        recs_sorted = [recs[i] for i in idxs]
        cls_sorted = [class_names[i] for i in idxs]
        # create bar plot
        ypos = range(self.num_classes - 1)
        plt.barh(ypos, recs_sorted)
        # set plot title
        plt.title(title)
        # add class names to the left
        plt.yticks(ypos, cls_sorted, size='small')
        # add label for x-axis
        plt.xlabel("Recall")
        # make figure larger
        fig.set_figwidth(fig.get_figwidth() + 1)
        fig.tight_layout()
        # save plot
        plt.savefig(fname)
        plt.close()
