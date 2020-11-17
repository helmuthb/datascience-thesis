import os
import argparse
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from ssdlite import ssdlite, ssd_loss
from preprocess import preprocess_func
from tfrecords import read_tfrecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        help='Directory with TFrecords.',
        required=True
    )
    args = parser.parse_args()

    # build model
    model, anchors = ssdlite()

    # Compile
    model.compile(optimizer=optimizers.Adam(), loss=ssd_loss)

    # Load training data
    train_ds = read_tfrecord(os.path.join(args.target, 'det_train.tfrec'))

    # Preprocess data
    train_ds = train_ds.map(preprocess_func(size=(300, 300)))

    # Perform training
    model.fit()


if __name__ == '__main__':
    main()
