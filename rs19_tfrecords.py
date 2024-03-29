#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file allows conversion of RailSem19 data to TensorFlow records.
"""

import os
import random
import argparse
import json

import numpy as np

from lib.tfr_utils import write_tfrecords, BBox

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2022, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def parse_json(json_data, classes):
    """Parse JSON file for objects.
    Args:
        json_data (object): Data from JSON-file for a record.
        classes (list): List of classes, starting with 'background'.
    Returns:
        metadata (dict): Metadata about the image.
        objects (list): Bounding boxes with label index
    """
    image_width = json_data["imgWidth"]
    image_height = json_data["imgHeight"]

    objects = []
    frame = json_data["frame"]
    for o in json_data["objects"]:
        label = o["label"]
        # we focus on object detection
        if "boundingbox" not in o:
            continue
        bb = o["boundingbox"]
        x0 = bb[0] / image_width
        y0 = bb[1] / image_height
        x1 = bb[2] / image_width
        y1 = bb[3] / image_height
        # check for plausibility
        is_ok = True
        if x0 >= x1 or y0 >= y1:
            print(f"Frame {frame} has empty bounding box for label {label}")
            is_ok = False
        if label not in classes:
            print(f"Frame {frame} contains unknown label {label}")
            is_ok = False
        if not is_ok:
            # skip this bounding box
            continue
        box = BBox(classes.index(label), label, x0, y0, x1, y1)
        objects.append(box)
    # no box found? add background box
    if len(objects) == 0:
        box = BBox(0, 'background', 0., 0., 1., 1.)
        objects.append(box)
    # prepare meta data
    meta = {
        'format': 'jpg',
        'width': image_width,
        'height': image_height,
        'name': frame
    }
    return meta, objects


def read_folder(root, det_classes):
    """Create data list from the root folder.
    Args:
        root (string): Root folder of RailSem19.
        det_classes (list[string]): List of detection class names.
    Returns:
        dataset (list(dict)): Data represented by {'image_path', 'objects'}.
    """
    jpegs_path = os.path.join(root, 'jpgs/rs19_val')
    jsons_path = os.path.join(root, 'jsons/rs19_val')
    masks_path = os.path.join(root, 'uint8/rs19_val')

    dataset = list()
    for f in os.listdir(jsons_path):
        # read JSON file
        with open(os.path.join(jsons_path, f), 'r') as json_file:
            json_data = json.loads(json_file.read())
        # parse objects
        metadata, objects = parse_json(json_data, det_classes)
        # get JPEG file path
        frame = json_data['frame']
        jpeg_path = os.path.join(jpegs_path, frame + ".jpg")
        mask_path = os.path.join(masks_path, frame + ".png")
        # append info to dataset
        dataset.append({
            'image_path': jpeg_path,
            'metadata': metadata,
            'mask': mask_path,
            'objects': objects})
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Make RailSem19 dataset TFRecords."
    )
    parser.add_argument(
        '--det-classes',
        type=str,
        help="File with class names for object detection.",
        required=True
    )
    parser.add_argument(
        '--source',
        type=str,
        help="Root directory of RailSem19 dataset.",
        required=True
    )
    parser.add_argument(
        '--tfrecords',
        type=str,
        help="Directory with output TFRecords.",
        required=True
    )
    parser.add_argument(
        '--test_split',
        type=float,
        help="Percentage of test data.",
        default=0.15
    )
    parser.add_argument(
        '--val_split',
        type=float,
        help="Percentage of validation data.",
        default=0.15
    )
    args = parser.parse_args()

    # Disable any CUDA devices - we don't need them here
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    # read class names
    with open(args.det_classes, 'r') as f:
        classes = f.read().splitlines()

    # Create data list from source folder
    data = read_folder(args.source, classes)
    # shuffle dataset
    n = len(data)
    shuffled_index = np.arange(n)
    random.seed(42)
    random.shuffle(shuffled_index)
    data = [data[i] for i in shuffled_index]
    # Split into training / validation / test data
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)
    n_train = n - n_test - n_val
    last_train = n_train
    last_val = n_train + n_val
    data_train = data[:last_train]
    data_val = data[last_train:last_val]
    data_test = data[last_val:]
    print(f"Training size {len(data_train)}, validation size {len(data_val)},"
          f"test size {len(data_test)}, total {n}")

    # make output folder (and path to it) if missing
    os.makedirs(args.tfrecords, exist_ok=True)
    # write TFRecords
    write_tfrecords(data_train, os.path.join(args.tfrecords, 'train.tfrec'), 5)
    write_tfrecords(data_val, os.path.join(args.tfrecords, 'val.tfrec'), 1)
    write_tfrecords(data_test, os.path.join(args.tfrecords, 'test.tfrec'), 1)


if __name__ == '__main__':
    main()
