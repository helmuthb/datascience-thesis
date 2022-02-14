#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file allows conversion of Pascal VOC data to TensorFlow records.
"""

import os
import random
import argparse

import numpy as np
import xml.etree.ElementTree as ET

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


def parse_xml(name, xml_data, classes):
    """Parse XML file for objects.
    Args:
        name (str): Name of image (without suffix).
        xml_data (object): Data from XML-file for a record.
        classes (list): List of classes, starting with 'background'.
    Returns:
        metadata (dict): Metadata about the image.
        objects (list): Bounding boxes with label index
    """
    size = xml_data.find("size")
    image_width = int(float(size.find("width").text))
    image_height = int(float(size.find("height").text))
    fname = xml_data.find("filename").text

    objects = []
    for o in xml_data.findall("object"):
        label = o.find("name").text
        bb = o.find("bndbox")
        x0 = int(float(bb.find("xmin").text)) / image_width
        y0 = int(float(bb.find("ymin").text)) / image_height
        x1 = int(float(bb.find("xmax").text)) / image_width
        y1 = int(float(bb.find("ymax").text)) / image_height
        # check for plausibility
        is_ok = True
        if x0 >= x1 or y0 >= y1:
            print(f"Frame {name} has empty bounding box for label {label}")
            is_ok = False
        if label not in classes:
            print(f"Frame {name} contains unknown label {label}")
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
        'name': name,
        'file': fname,
    }
    return meta, objects


def read_folder(root, year, det_classes):
    """Create data list from the root folder.
    Args:
        root (string): Root folder of Pascal data set.
        year (string): Year of Pascal data set.
        det_classes (list[string]): List of detection class names.
    Returns:
        dataset (list(dict)): Data represented by {'image_path', 'objects'}.
    """
    annotations = f"{root}/VOC{year}/Annotations"
    jpegs_path = f"{root}/VOC{year}/JPEGImages"
    png_path = f"{root}/VOC{year}/SegmentationClass"
    dataset = list()
    for f in os.listdir(annotations):
        # read XML file
        path = f"{annotations}/{f}"
        xml_data = ET.parse(path)
        name = os.path.splitext(f)[0]
        metadata, objects = parse_xml(name, xml_data, det_classes)
        # get JPEG file path
        jpeg_path = f"{jpegs_path}/{metadata['file']}"
        mask_path = f"{png_path}/{metadata['name']}.png"
        if not os.path.exists(mask_path):
            mask_path = None
        # append info to dataset
        dataset.append({
            'image_path': jpeg_path,
            'metadata': metadata,
            'mask': mask_path,
            'objects': objects})
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Make Pascal dataset TFRecords."
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
        help="Root directory of Pascal dataset.",
        required=True
    )
    parser.add_argument(
        '--year',
        type=str,
        help="Pascal year (2007 or 2012)",
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
    data = read_folder(args.source, args.year, classes)
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
