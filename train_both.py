from contextlib import redirect_stdout
import os
import argparse

import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope, plot_model
from tensorflow.keras.callbacks import CSVLogger
from lib.augment import Augment

from lib.deeplab import add_deeplab_features
from lib.np_bbox_utils import BBoxUtils
from lib.preprocess import preprocess, filter_classes_bbox, filter_classes_mask
from lib.ssdlite import (
    add_ssdlite_features, detection_head, get_default_boxes_cwh,
    ssdlite_base_layers)
from lib.tfr_utils import read_tfrecords
from lib.mobilenet import mobilenetv2
from lib.losses import SSDLoss, DeeplabLoss
import lib.rs19_classes as rs19


def print_model(model, name, out_folder):
    """Print a model, together with its summary and a plot.

    Args:
        model (tf.keras.Model): The model which shall be printed.
        name (str): Name which shall be used for the files created.
        out_folder (str): Folder for the plots and info files.
    """
    # create a summary of the model
    summary_file = os.path.join(out_folder, name + '-summary.txt')
    with open(summary_file, mode='w') as f:
        with redirect_stdout(f):
            model.summary()
    # create a plot
    plot_file = os.path.join(out_folder, name + '.png')
    plot_model(model, to_file=plot_file)


def ssd_deeplab_model(size, n_det, n_seg):
    """
    """
    input_layer = tf.keras.layers.Input(
        shape=(size[0], size[1], 3))
    # base model
    base = mobilenetv2(input_layer)
    # add deeplab layers
    deeplab_output = add_deeplab_features(base, n_seg)
    # add SSDlite layers
    ext_base = add_ssdlite_features(base)
    l1, l2, l3, l4, l5, l6 = ssdlite_base_layers(ext_base)
    # add class and location predictions
    ssd_output = detection_head(n_det, l1, l2, l3, l4, l5, l6)
    # create models
    ssd_model = tf.keras.Model(
        inputs=input_layer,
        outputs=ssd_output
    )
    deeplab_model = tf.keras.Model(
        inputs=input_layer,
        outputs=deeplab_output
    )
    combined_model = tf.keras.Model(
        inputs=input_layer,
        outputs=[deeplab_output, ssd_output])
    # calculate default boxes
    default_boxes_cwh = get_default_boxes_cwh(l1, l2, l3, l4, l5, l6)
    # return combined model and the defaults
    return combined_model, default_boxes_cwh, base, deeplab_model, ssd_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tfrecords',
        type=str,
        help='Directory with TFrecords.',
        required=True
    )
    parser.add_argument(
        '--in-model',
        type=str,
        help='Folder for input model which will be further trained.'
    )
    parser.add_argument(
        '--out-model',
        type=str,
        help='Folder for output model after training.',
        required=True
    )
    parser.add_argument(
        '--plot',
        type=str,
        help='Folder for model plots if desired.'
    )
    parser.add_argument(
        '--plot-keras',
        action='store_true',
        help='Create plot / summary for Keras mobilenet as well.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs.',
        required=True
    )
    parser.add_argument(
        '--logs',
        type=str,
        help='Folder for storing training logs'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Perform augmentation.'
    )
    parser.add_argument(
        '--ssd-weight',
        type=float,
        default=1.0,
        help='Weight for SSD training (default = 1.0).'
    )
    parser.add_argument(
        '--deeplab-weight',
        type=float,
        default=1.0,
        help='Weight for DeepLab training (default = 1.0).'
    )
    parser.add_argument(
        '--batches-per-epoch',
        type=int,
        help='Number of batches per epoch (default = full dataset).'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Number of samples per batch (default=8).'
    )
    args = parser.parse_args()
    plot_dir = args.plot
    plot_keras = args.plot_keras
    tfrecdir = args.tfrecords
    in_model = args.in_model
    out_model = args.out_model
    num_epochs = args.epochs
    logs = args.logs
    augment = args.augment
    ssd_weight = args.ssd_weight
    deeplab_weight = args.deeplab_weight
    batches_per_epoch = args.batches_per_epoch
    batch_size = args.batch_size
    if logs and not os.path.exists(logs):
        os.makedirs(logs)

    # number of classes
    n_seg = len(rs19.seg_subset)
    n_det = len(rs19.det_subset)

    # build model
    models = ssd_deeplab_model((300, 300), n_det, n_seg)
    model, default_boxes_cwh, base, deeplab, ssd = models

    # load model if fine tuning
    if in_model:
        with custom_object_scope({
                    'SSDLoss': SSDLoss,
                    'DeeplabLoss': DeeplabLoss
                }):
            model = tf.keras.models.load_model(in_model)

    # Bounding box utility object
    bbox_util = BBoxUtils(n_det, default_boxes_cwh)

    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # create plots / summaries of models
        print_model(model, 'combined', plot_dir)
        print_model(base, 'base', plot_dir)
        print_model(deeplab, 'deeplab', plot_dir)
        print_model(ssd, 'ssd', plot_dir)
        if plot_keras:
            # Keras reference implementation
            mnet_keras = tf.keras.applications.MobileNetV2(
                input_shape=(300, 300, 3),
                include_top=False,
                weights=None,
                classes=n_det
            )
            # describe model
            print_model(mnet_keras, 'keras', plot_dir)

    # Loss functions
    losses = {
        "deeplab_output": DeeplabLoss(),
        "ssd_output": SSDLoss(n_det),
    }

    # loss weights
    lossWeights = {
        "deeplab_output": deeplab_weight,
        "ssd_output": ssd_weight
    }

    # compile combined model
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses,
        loss_weights=lossWeights,
        run_eagerly=False,
        metrics={
            "deeplab_output": DeeplabLoss(),
            "ssd_output": {
                "neg_cls_loss": SSDLoss(n_det, "neg_cls_loss"),
                "pos_cls_loss": SSDLoss(n_det, "pos_cls_loss"),
                "pos_loc_loss": SSDLoss(n_det, "pos_loc_loss")
            }
        }
    )

    # Load training & validation data
    train_ds_orig = read_tfrecords(
            os.path.join(tfrecdir, 'train.tfrec'))
    val_ds_orig = read_tfrecords(
            os.path.join(tfrecdir, 'val.tfrec'))

    # Augment data
    augmentor = Augment(1080, 1920)
    augmentor.crop_and_pad()
    augmentor.horizontal_flip()
    augmentor.hsv()
    augmentor.random_brightness_contrast()

    # generator with augmentation
    def train_aug_gen():
        """Get augmented data from training data.
        """
        for e in train_ds_orig.as_numpy_iterator():
            yield(augmentor(*e))

    if augment:
        train_ds_aug = tf.data.Dataset.from_generator(
            train_aug_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
    else:
        train_ds_aug = train_ds_orig

    # Filter for classes of interest
    train_ds_filtered_det = train_ds_aug.map(
        filter_classes_bbox(rs19.det_classes, rs19.det_subset)
    )
    val_ds_filtered_det = val_ds_orig.map(
        filter_classes_bbox(rs19.det_classes, rs19.det_subset)
    )
    train_ds_filtered = train_ds_filtered_det.map(
        filter_classes_mask(rs19.seg_classes, rs19.seg_subset)
    )
    val_ds_filtered = val_ds_filtered_det.map(
        filter_classes_mask(rs19.seg_classes, rs19.seg_subset)
    )

    # Preprocess data
    train_ds = train_ds_filtered.map(
        preprocess((300, 300), bbox_util, n_seg))
    val_ds = val_ds_filtered.map(
        preprocess((300, 300), bbox_util, n_seg))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=batch_size)
    val_ds_batch = val_ds.batch(batch_size=batch_size)

    # prepare callbacks
    callbacks = []
    if logs:
        callbacks.append(
            CSVLogger(logs + '/history.csv', append=True, separator=';'))

    # perform training
    model.fit(
        train_ds_batch,
        epochs=num_epochs,
        steps_per_epoch=batches_per_epoch,
        validation_data=val_ds_batch,
        callbacks=callbacks)

    # save resulting model
    model.save(out_model)


if __name__ == '__main__':
    main()
