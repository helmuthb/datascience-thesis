#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This package contains helper functions related to
Tensorflow records.
"""

import os
import random
import argparse
import json

import numpy as np

from lib.tfr_utils import write_tfrecords, parse_json
import lib.rs19_classes as rs19

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


def read_folder(root):
    """Create data list from the root folder.
    Args:
        root (string): Root folder of RailSem19.
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
        metadata, objects = parse_json(json_data, rs19.det_classes)
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

    # Create data list from source folder
    data = read_folder(args.source)
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
