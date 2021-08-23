from contextlib import redirect_stdout
import os
import argparse

import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope, plot_model

from lib.deeplab import add_deeplab_features
from lib.np_bbox_utils import BBoxUtils
from lib.preprocess import preprocess
from lib.ssdlite import (
    add_ssdlite_features, detection_head, get_anchor_boxes_cwh,
    ssdlite_base_layers)
from lib.tfr_utils import read_tfrecords
from lib.mobilenet import mobilenetv2
from lib.losses import SSDLoss


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


def ssd_deeplab_model(size, n_seg, n_det):
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
    # calculate anchor boxes
    anchor_boxes_cwh = get_anchor_boxes_cwh(l1, l2, l3, l4, l5, l6)
    # return combined model and the anchors
    return combined_model, anchor_boxes_cwh, base, deeplab_model, ssd_model


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
    args = parser.parse_args()
    plot_dir = args.plot
    plot_keras = args.plot_keras
    tfrecdir = args.tfrecords
    in_model = args.in_model
    out_model = args.out_model
    num_epochs = args.epochs

    # build model
    models = ssd_deeplab_model((300, 300), 11, 11)
    model, anchor_boxes_cwh, base, deeplab, ssd = models

    # Bounding box utility object
    bbox_util = BBoxUtils(11, anchor_boxes_cwh)

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
                classes=11
            )
            # describe model
            print_model(mnet_keras, 'keras', plot_dir)

    # Loss functions
    losses = {
        "deeplab_output":
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "ssd_output": SSDLoss(11),
    }
    # loss weights
    lossWeights = {
        "deeplab_output": 1.0,
        "ssd_output": 1.0,
    }
    # compile combined model
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses,
        loss_weights=lossWeights,
        run_eagerly=False,
    )

    # Load training & validation data
    train_ds_orig = read_tfrecords(
            os.path.join(tfrecdir, 'train.tfrec'))
    val_ds_orig = read_tfrecords(
            os.path.join(tfrecdir, 'val.tfrec'))

    # Preprocess data
    train_ds = train_ds_orig.map(
        preprocess((300, 300), bbox_util, 11))
    val_ds = val_ds_orig.map(
        preprocess((300, 300), bbox_util, 11))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=8)
    val_ds_batch = val_ds.batch(batch_size=8)

    # load model if fine tuning
    if in_model:
        with custom_object_scope({'SSDLoss': SSDLoss}):
            model = tf.keras.models.load_model(in_model)

    # perform training
    model.fit(train_ds_batch, epochs=num_epochs, validation_data=val_ds_batch)

    # save resulting model
    model.save(out_model)


if __name__ == '__main__':
    main()
