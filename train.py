import csv
import os
import time
import argparse
from numpy import infty

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from contextlib import redirect_stdout

from tqdm import tqdm

from lib.augment import Augment
from lib.tf_bbox_utils import BBoxUtils
from lib.preprocess import (
    filter_empty_samples, preprocess, filter_classes_bbox, filter_classes_mask)
from lib.tfr_utils import read_tfrecords
from lib.losses import SSDLosses, DeeplabLoss
from lib.combined import get_training_step, ssd_deeplab_model, loss_list
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
    plot_model(model, to_file=plot_file, show_dtype=True, show_shapes=True)


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
        '--epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs (if not stopped early).'
    )
    parser.add_argument(
        '--logs',
        type=str,
        help='Folder for storing training logs.'
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
        '--batch-size',
        type=int,
        default=8,
        help='Number of samples per batch (default=8).'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debugging: add more metrics for detailed analysis.'
    )
    parser.add_argument(
        '--subset-only',
        action='store_true',
        help='Use only subset of classes instead of all.'
    )
    parser.add_argument(
        '--freeze-base-epochs',
        type=int,
        default=1,
        help='Freeze base layers for number of epochs.'
    )
    parser.add_argument(
        '--freeze-ssd',
        action='store_true',
        help='Freeze SSD layers.'
    )
    parser.add_argument(
        '--freeze-deeplab',
        action='store_true',
        help='Freeze DeepLab layers.'
    )
    parser.add_argument(
        '--ssd-only',
        action='store_true',
        help='Only train SSDlite model.'
    )
    parser.add_argument(
        '--deeplab-only',
        action='store_true',
        help='Ony train DeepLab model.'
    )
    parser.add_argument(
        '--model-width',
        type=int,
        default=224,
        help='Specify image width for model.'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=1920,
        help='Specify original image width.'
    )
    parser.add_argument(
        '--image-height',
        type=int,
        default=1080,
        help='Specify original image height.'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=1,
        help='Warmup epochs for learning rate schedule.'
    )
    parser.add_argument(
        '--warmup-learning-rate',
        type=float,
        default=1e-4,
        help='Warmup learning rate.'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Peak learning rate after warmup.'
    )
    parser.add_argument(
        '--decay-factor',
        type=float,
        default=0.9,
        help='Factor to reduce learning rate after non-improvement.'
    )
    parser.add_argument(
        '--stop-after',
        type=int,
        default=5,
        help='Stop after N epochs without improvement.'
    )
    args = parser.parse_args()
    plot_dir = args.plot
    tfrecdir = args.tfrecords
    in_model = args.in_model
    out_model = args.out_model
    num_epochs = args.epochs
    logs = args.logs
    augment = args.augment
    ssd_weight = args.ssd_weight
    deeplab_weight = args.deeplab_weight
    batch_size = args.batch_size
    subset_only = args.subset_only
    freeze_base_epochs = args.freeze_base_epochs
    freeze_ssd = args.freeze_ssd
    freeze_deeplab = args.freeze_deeplab
    ssd_only = args.ssd_only
    deeplab_only = args.deeplab_only
    model_width = args.model_width
    image_width = args.image_width
    image_height = args.image_height
    warmup_epochs = args.warmup_epochs
    warmup_learning_rate = args.warmup_learning_rate
    learning_rate = args.learning_rate
    decay_factor = args.decay_factor
    stop_after = args.stop_after
    # checks for consistency
    if ssd_only and deeplab_only:
        print("Only one of ssd-only and deeplab-only can be specified.")
        return
    # create folder for log files
    if logs and not os.path.exists(logs):
        os.makedirs(logs)

    # number of classes
    if subset_only:
        n_seg = len(rs19.seg_subset)
        n_det = len(rs19.det_subset)
    else:
        n_seg = len(rs19.seg_classes)
        n_det = len(rs19.det_classes)

    # build model
    models = ssd_deeplab_model((model_width, model_width), n_det, n_seg)
    model, default_boxes_cw, base, deeplab, ssd = models

    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # create plots / summaries of models
        print_model(model, 'combined', plot_dir)
        print_model(base, 'base', plot_dir)
        print_model(deeplab, 'deeplab', plot_dir)
        print_model(ssd, 'ssd', plot_dir)

    # find SSD & Deeplab layers
    base_names = {layer.name for layer in base.layers}
    deeplab_names = {layer.name for layer in deeplab.layers} - base_names
    ssd_names = {layer.name for layer in ssd.layers} - base_names

    # SSD-only, DeepLab-only?
    if ssd_only:
        model = ssd
    elif deeplab_only:
        model = deeplab

    # load model if provided
    if in_model:
        model = tf.keras.models.load_model(in_model)

    # Loss functions
    losses = loss_list(SSDLosses(3), DeeplabLoss(), ssd_only, deeplab_only)
    # weights for losses
    if ssd_only:
        loss_weights = (ssd_weight, ssd_weight)
    elif deeplab_only:
        loss_weights = (deeplab_weight, )
    else:
        loss_weights = (ssd_weight, ssd_weight, deeplab_weight)

    # Freeze weights as requested
    if freeze_ssd:
        for l_name in ssd_names:
            model.get_layer(l_name).trainable = False
    if freeze_deeplab:
        for l_name in deeplab_names:
            model.get_layer(l_name).trainable = False
    if freeze_base_epochs != 0:
        for layer in model.layers:
            if layer.name not in ssd_names and layer.name not in deeplab_names:
                layer.trainable = False

    # Bounding box utility object
    bbox_util = None if deeplab_only else BBoxUtils(n_det, default_boxes_cw)

    # Number of classes for segmentation
    if ssd_only:
        n_seg = 0

    # Load training & validation data
    train_ds = read_tfrecords(
            os.path.join(tfrecdir, 'train.tfrec'))
    val_ds = read_tfrecords(
            os.path.join(tfrecdir, 'val.tfrec'))

    # Augment data
    if augment:
        augmentor = Augment(image_height, image_width)
        augmentor.crop_and_pad()
        augmentor.horizontal_flip()
        augmentor.hsv()
        augmentor.random_brightness_contrast()
        train_ds = train_ds.map(augmentor.tf_wrap())

    # Filter for classes of interest
    if subset_only:
        ssd_filter = filter_classes_bbox(rs19.det_classes, rs19.det_subset)
        deeplab_filter = filter_classes_mask(rs19.seg_classes, rs19.seg_subset)
        if ssd_only:
            train_ds = train_ds.map(ssd_filter)
            val_ds = val_ds.map(ssd_filter)
        elif deeplab_only:
            train_ds = train_ds.map(deeplab_filter)
            val_ds = val_ds.map(deeplab_filter)
        else:
            train_ds = train_ds.map(ssd_filter).map(deeplab_filter)
            val_ds = val_ds.map(ssd_filter).map(deeplab_filter)

    # Filter out empty samples - if SSD requested
    if not deeplab_only:
        train_ds = train_ds.filter(filter_empty_samples)
        val_ds = val_ds.filter(filter_empty_samples)

    # Preprocess data
    train_ds = train_ds.map(
        preprocess((model_width, model_width), bbox_util, n_seg))
    val_ds = val_ds.map(
        preprocess((model_width, model_width), bbox_util, n_seg))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=batch_size)
    val_ds_batch = val_ds.batch(batch_size=batch_size)

    # learning rate
    if warmup_epochs > 0:
        lr = warmup_learning_rate
    else:
        lr = learning_rate
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lambda: lr)
    # training step
    training_step = get_training_step(model, losses, loss_weights,
                                      optimizer, ssd_only, deeplab_only)

    # open logfile
    if logs:
        ts = time.strftime("%Y%m%d-%H%M%S")
        csv_file = open(f"{logs}/history-{ts}.csv", "a", newline="")
        csv_writer = csv.writer(csv_file)

    # minimum loss so far
    min_loss = infty

    # number of non-improvements
    non_improved = 0

    # perform training
    for epoch in range(num_epochs):
        # end of warmup?
        if epoch == warmup_epochs:
            lr = learning_rate
        # unfreeze base?
        if epoch == freeze_base_epochs:
            for layer in model.layers:
                if (layer.name not in ssd_names and
                        layer.name not in deeplab_names):
                    layer.trainable = True

        train_conf_loss = 0.0
        train_locs_loss = 0.0
        train_segs_loss = 0.0
        train_loss = 0.0
        train_num = 0
        start_time = time.time()
        for batch in tqdm(train_ds_batch):
            img, gt = batch
            ll = training_step(img, gt)
            if ssd_only:
                train_conf_loss += ll[1].numpy()
                train_locs_loss += ll[2].numpy()
            elif deeplab_only:
                train_segs_loss += ll[1].numpy()
            else:
                train_conf_loss += ll[1].numpy()
                train_locs_loss += ll[2].numpy()
                train_segs_loss += ll[3].numpy()
            train_loss += sum([li*wi for li, wi in zip(ll[1:], loss_weights)])
            train_num += 1
        train_time = time.time() - start_time
        train_conf_loss /= train_num
        train_locs_loss /= train_num
        train_segs_loss /= train_num
        train_loss /= train_num
        out = [epoch+1, lr, train_time, train_loss.numpy()]
        if not deeplab_only:
            out += [train_conf_loss, train_locs_loss]
        if not ssd_only:
            out += [train_segs_loss]
        print(f"Epoch {epoch}: lr={lr}, time={train_time}, loss={train_loss}")
        # validation run
        val_conf_loss = 0.0
        val_locs_loss = 0.0
        val_segs_loss = 0.0
        val_loss = 0.0
        val_num = 0
        start_time = time.time()
        for batch in val_ds_batch:
            img, gt = batch
            pr = model(img)
            ll = losses(gt, pr)
            if ssd_only:
                val_conf_loss += ll[0].numpy()
                val_locs_loss += ll[1].numpy()
            elif deeplab_only:
                val_segs_loss += ll[0].numpy()
            else:
                val_conf_loss += ll[0].numpy()
                val_locs_loss += ll[1].numpy()
                val_segs_loss += ll[2].numpy()
            val_loss += sum([li*wi for li, wi in zip(ll, loss_weights)])
            val_num += 1
        val_time = time.time() - start_time
        val_conf_loss /= val_num
        val_locs_loss /= val_num
        val_segs_loss /= val_num
        val_loss /= val_num
        out += [val_time, val_loss.numpy()]
        if not deeplab_only:
            out += [val_conf_loss, val_locs_loss]
        if not ssd_only:
            out += [val_segs_loss]
        print(f"Epoch {epoch}: val. time={val_time}, val. loss={val_loss}")
        if logs:
            csv_writer.writerow(out)
            csv_file.flush()
        # best model so far?
        if (val_loss < min_loss) and out_model:
            print(f"New minimum loss {val_loss} - saving model")
            min_loss = val_loss
            model.save(out_model)
            non_improved = 0
        else:
            lr *= decay_factor
            non_improved += 1
        if non_improved >= stop_after:
            # end of training
            print(f"No improvement after {non_improved} epochs - ending.")
            break

    # close log file
    if logs:
        csv_file.close()


if __name__ == '__main__':
    main()
