#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This package contains helper functions related to
Tensorflow records.
"""

from collections import namedtuple

import tensorflow as tf
from tqdm import tqdm

__author__ = 'Helmuth Breitenfellner'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.1.0'
__maintainer__ = 'Helmuth Breitenfellner'
__email__ = 'helmuth.breitenfellner@student.tuwien.ac.at'
__status__ = 'Experimental'


# metadata for a RailSem record bounding box
BBox = namedtuple('BBox', ['cl', 'lb', 'x0', 'y0', 'x1', 'y1'])

# Description of tf.train.Example
feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/segmentation/mask': tf.io.FixedLenFeature([], tf.string),
}


def wrap_in_list(value):
    """Wrap value - if not already a list - into a list."""
    if not isinstance(value, list):
        value = [value]
    return value


def int64_feature(value):
    """Convert integer value (or list of values) to TensorFlow Feature."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=wrap_in_list(value))
    )


def float_feature(value):
    """Convert float value (or list of values) to TensorFlow Feature."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=wrap_in_list(value))
    )


def bytes_feature(value):
    """Convert bytes/string value (or list of values) to TensorFlow Feature."""
    value = wrap_in_list(value)

    def str_wrap(v):
        return v.encode() if isinstance(v, str) else v

    value = [str_wrap(v) for v in value]
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def load_file(file_path):
    """Load a file into a string using TensorFlow GFile."""
    with tf.io.gfile.GFile(file_path, 'rb') as fp:
        file_data = fp.read()
    return file_data


def img_to_example(filename, metadata, objects, pngfile):
    """Create TensorFlow Example for a given record in the dataset.
    For a filename together with the list of objects in it, it
    will first load the image and then add the bounding boxes
    and classes as features.
    Args:
        filename (string): Full path to the image file.
        metadata (dict): Metadata like name, width and height of the image.
        objects (list): List of bounding boxes defined in the image.
        pngfile (string): Full path to the segmentation (8bit PNG).
    Returns:
        example (bytes): The serialized TensorFlow example.
    """
    image_string = load_file(filename)
    boxes_x0 = [b.x0 for b in objects]
    boxes_y0 = [b.y0 for b in objects]
    boxes_x1 = [b.x1 for b in objects]
    boxes_y1 = [b.y1 for b in objects]
    boxes_cl = [b.cl for b in objects]
    boxes_lb = [b.lb for b in objects]
    seg_string = load_file(pngfile)

    feature = {
        'image/height': int64_feature(metadata['height']),
        'image/width': int64_feature(metadata['width']),
        'image/filename': bytes_feature(metadata['name']),
        'image/source_id': bytes_feature(metadata['name']),
        'image/encoded': bytes_feature(image_string),
        'image/format': bytes_feature(metadata['format']),
        'image/object/bbox/xmin': float_feature(boxes_x0),
        'image/object/bbox/xmax': float_feature(boxes_x1),
        'image/object/bbox/ymin': float_feature(boxes_y0),
        'image/object/bbox/ymax': float_feature(boxes_y1),
        'image/object/class/text': bytes_feature(boxes_lb),
        'image/object/class/label': int64_feature(boxes_cl),
        'image/segmentation/mask': bytes_feature(seg_string),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    return ex.SerializeToString()


def img_from_example(example):
    """Parse a tf.train.Example into image and metadata.
    Args:
        example (tf.Tensor): One serialized example to be parsed.
    Returns:
        image (tf.Tensor): Pixel values of the image (HxBxC).
        bboxes (tf.Tensor): Bounding boxes of the image.
        classes (tf.Tensor): Numeric object classes of the image.
        mask (tf.Tensor): Segmentation mask of the image.
        name (tf.string): Name of the image.
    """
    # parse the image data
    data = tf.io.parse_single_example(example, feature_description)
    name = data['image/source_id']
    jpeg = data['image/encoded']
    image = tf.cast(tf.image.decode_jpeg(jpeg, channels=3), tf.float32)
    png = data['image/segmentation/mask']
    mask = tf.image.decode_png(png, channels=1)
    # create bounding boxes
    x0_part = tf.expand_dims(data['image/object/bbox/xmin'].values, 1)
    y0_part = tf.expand_dims(data['image/object/bbox/ymin'].values, 1)
    x1_part = tf.expand_dims(data['image/object/bbox/xmax'].values, 1)
    y1_part = tf.expand_dims(data['image/object/bbox/ymax'].values, 1)
    bboxes = tf.concat([x0_part, y0_part, x1_part, y1_part], 1)
    # fetch classes
    classes = data['image/object/class/label'].values
    # return image, classes & bboxes
    return image, bboxes, classes, mask, name


def write_tfrecords(rows, out_file, num_shards, verbose=True):
    """Write TFRecord file for a dataset.
    Args:
        rows (list(dict)): The data,
            represented by `{'image_path', 'metadata', 'objects', 'mask'}`.
        out_file (string): Filename of TFRecord, or start of file names.
        num_shards (int): Number of TFRecord files to create.
        verbose (boolean): Whether the progress shall be shown visually.
    """
    # calculate number of examples per shard
    samples_per_shard = (len(rows)+num_shards-1) // num_shards

    # get name of shard
    def get_shard_name(base, idx, num_shards):
        return f"{base}-{idx:05d}-of-{num_shards:05d}"
    # current shard index
    idx = 0
    # TFRecord writer (starts with None)
    writer = None
    # Show progress bar if verbose
    if verbose:
        rows = tqdm(rows)
    for i, x in enumerate(rows):
        # Check if we shall open a new TFrecord file
        if i % samples_per_shard == 0:
            # close old writer if exists
            if writer is not None:
                writer.close()
            shard_name = get_shard_name(out_file, idx, num_shards)
            writer = tf.io.TFRecordWriter(shard_name)
            # count to next index
            idx += 1
        example = img_to_example(
            x['image_path'],
            x['metadata'],
            x['objects'],
            x['mask'])
        writer.write(example)


def read_tfrecords(file):
    """Create a tf.data.Dataset from a file.
    The name provided is assumed to be the base name used for sharded
    write.

    Args:
        file (string): Name of the TFRecord file to be read.
    Returns:
        dataset (tf.data.Dataset): Dataset with image and metadata.
    """
    file_names = tf.data.Dataset.list_files(f"{file}-*")
    dataset_raw = tf.data.TFRecordDataset(file_names)
    return dataset_raw.map(img_from_example)


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
