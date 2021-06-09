import os
import argparse

import tensorflow.keras.optimizers as optimizers
import tensorflow as tf

from lib.deeplab import deeplab
from lib.preprocess import preprocess_seg
from lib.tfr_utils import read_tfrecords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tfrecords',
        type=str,
        help='Directory with TFrecords.',
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Directory for model.',
        required=True
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs.',
        required=True
    )
    args = parser.parse_args()

    # build model
    model = deeplab((300, 300), 11)

    # Compile
    model.compile(
        optimizer=optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        run_eagerly=False,
    )

    # Load training & validation data
    train_ds_orig = read_tfrecords(os.path.join(args.model, 'det_train.tfrec'))
    val_ds_orig = read_tfrecords(os.path.join(args.model, 'det_val.tfrec'))

    # Preprocess data
    train_ds = train_ds_orig.map(preprocess_seg((300, 300), 11))
    val_ds = val_ds_orig.map(preprocess_seg((300, 300), 11))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=8)
    val_ds_batch = val_ds.batch(batch_size=8)

    # load model or train
    if os.path.isdir(f"{args.model}/model"):
        model = tf.keras.models.load_model(f"{args.model}/model")
    else:
        # Perform training
        model.fit(
                train_ds_batch,
                epochs=args.epochs,
                validation_data=val_ds_batch)
        # save model
        model.save(f"{args.model}/model")

    # perform inference on validation set
    preds = model.predict(val_ds_batch)
    print(preds.shape)


if __name__ == '__main__':
    main()
