import tensorflow as tf


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
    with tf.io.gfile.GFile(file_path, 'rb') as fp:
        file_data = fp.read()
    return file_data
