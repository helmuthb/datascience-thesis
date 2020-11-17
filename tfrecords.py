import tensorflow as tf
from collections import namedtuple
from tqdm import tqdm


# metadata for a RailSem record bounding box
BBox = namedtuple('BBox', ['cl', 'x0', 'y0', 'x1', 'y1'])

# Description of tf.train.Example
feature_description = {
    'x0': tf.io.VarLenFeature(tf.float32),
    'y0': tf.io.VarLenFeature(tf.float32),
    'x1': tf.io.VarLenFeature(tf.float32),
    'y1': tf.io.VarLenFeature(tf.float32),
    'cl': tf.io.VarLenFeature(tf.int64),
    'im': tf.io.FixedLenFeature([], tf.string)
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
    """Convert bytes value (or list of values) to TensorFlow Feature."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=wrap_in_list(value))
    )


def load_file(file_path):
    """Load a file into a string using TensorFlow GFile."""
    with tf.gfile.GFile(file_path, 'rb') as fp:
        file_data = fp.read()
    return file_data


def to_example(filename, objects):
    """Create TensorFlow Example for a given record in the dataset.
    For a filename together with the list of objects in it, it
    will first load the image and then add the bounding boxes
    and classes as features.
    """
    image_string = load_file(filename)
    boxes_x0 = [b.x0 for b in objects]
    boxes_y0 = [b.y0 for b in objects]
    boxes_x1 = [b.x1 for b in objects]
    boxes_y1 = [b.y1 for b in objects]
    boxes_cl = [b.cl for b in objects]

    feature = {
        'x0': float_feature(boxes_x0),
        'y0': float_feature(boxes_y0),
        'x1': float_feature(boxes_x1),
        'y1': float_feature(boxes_y1),
        'cl': int64_feature(boxes_cl),
        'im': bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(rows, out_file):
    """Write TFRecord file for a dataset.
    Args:
        rows (list(dict)): The data,
            represented by {'image_path', 'objects'}.
        out_file (string): Filename of TFRecord.
    """
    # TFRecord writer
    writer = tf.io.TFRecordWriter(out_file)
    # Show progress bar
    for x in tqdm(rows):
        example = to_example(x['image_path'], x['objects'])
        writer.write(example.SerializeToString())


def from_example(example):
    """Parse a tf.train.Example into image and metadata.
    Args:
        example (tf.train.Example): One example to be parsed.
    Returns:
        image (tf.Tensor): Pixel values of the image.
        bboxes (tf.Tensor): Bounding boxes of the image.
        classes (tf.Tensor): Object classes of the image.
    """
    # parse the image data
    data = tf.io.parse_single_example(example, feature_description)
    jpeg = data['im']
    image = tf.cast(tf.image.decode_jpeg(jpeg, channels=3), tf.float32)
    # create bounding boxes
    x0_part = tf.expand_dims(data['x0'].values, 1)
    y0_part = tf.expand_dims(data['y0'].values, 1)
    x1_part = tf.expand_dims(data['x1'].values, 1)
    y1_part = tf.expand_dims(data['y1'].values, 1)
    bboxes = tf.concat([x0_part, y0_part, x1_part, y1_part], 1)
    # fetch classes
    classes = data['cl'].values
    # return image, classes & bboxes
    return image, bboxes, classes


def read_tfrecord(file):
    """Create a tf.data.Dataset from a file, with split features.

    Args:
        file (string): Name of the TFRecord file to be read.
    Returns:
        dataset (tf.data.Dataset): Dataset with image and metadata.
    """
    dataset_raw = tf.data.TFRecordDataset(file)
    return dataset_raw.map(from_example)
