import tensorflow as tf
from tqdm import tqdm
from tfrecords import load_file, bytes_feature


# Description of tf.train.Example
feature_description = {
    'cl': tf.io.FixedLenFeature([], tf.string),
    'im': tf.io.FixedLenFeature([], tf.string)
}


def to_example(image_path, classes_path):
    """Create TensorFlow Example for a given record in the dataset.
    For a filename together with the classes encoded as pixels in it, it
    will first load the image and then encode the pixels as 8-bit PNG,
    and add it to the example.
    """
    image_string = load_file(image_path)
    classes_string = load_file(classes_path)
    feature = {
        'cl': bytes_feature(classes_string),
        'im': bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(rows, out_file):
    """Write TFRecord file for a dataset.
    Args:
        rows (list(dict)): The data,
            represented by {'image_path', 'classes_path'}.
        out_file (string): Filename of TFRecord.
    """
    # TFRecord writer
    writer = tf.io.TFRecordWriter(out_file)
    # Show progress bar
    for x in tqdm(rows):
        example = to_example(x['image_path'], x['classes_path'])
        writer.write(example.SerializeToString())


def from_example(example):
    """Parse a tf.train.Example into image and classes.
    Args:
        example (tf.train.Example): One example to be parsed.
    Returns:
        image (tf.Tensor): Pixel values of the image.
        classes (tf.Tensor): Pixel-wise classes.
    """
    # parse the example
    data = tf.io.parse_single_example(example, feature_description)
    # extract the image data
    jpeg = data['im']
    image = tf.cast(tf.image.decode_jpeg(jpeg, channels=3), tf.float32)
    # extract the classes data
    png = data['cl']
    classes = tf.image.decode_png(png, channels=1, dtype=tf.uint8)
    # return image & class pixels
    return image, classes


def read_tfrecord(file):
    """Create a tf.data.Dataset from a file, with split features.

    Args:
        file (string): Name of the TFRecord file to be read.
    Returns:
        dataset (tf.data.Dataset): Dataset with image and metadata.
    """
    dataset_raw = tf.data.TFRecordDataset(file)
    return dataset_raw.map(from_example)
