import csv
import os
import time
import argparse
from numpy import infty

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tqdm import tqdm
from contextlib import redirect_stdout

from lib.augment import Augment
from lib.tf_bbox_utils import BBoxUtils
from lib.preprocess import (
    filter_empty_samples, preprocess, filter_classes_bbox, filter_classes_mask)
from lib.tfr_utils import read_tfrecords
from lib.losses import SSDLosses, DeeplabLoss
from lib.combined import (
    training_step, ssd_deeplab_model, combined_losses, CustomSchedule)
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
        '--batch-size',
        type=int,
        default=8,
        help='Number of samples per batch (default=8).'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debugging: add more metrics for detailed analysis'
    )
    parser.add_argument(
        '--all-classes',
        action='store_true',
        help='Use all classes instead of a subset'
    )
    parser.add_argument(
        '--freeze-base',
        action='store_true',
        help='Freeze base layers'
    )
    parser.add_argument(
        '--freeze-ssd',
        action='store_true',
        help='Freeze SSD layers'
    )
    parser.add_argument(
        '--freeze-deeplab',
        action='store_true',
        help='Freeze DeepLab layers'
    )
    parser.add_argument(
        '--model-width',
        type=int,
        default=224,
        help='Specify image width for model'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=1920,
        help='Specify original image width'
    )
    parser.add_argument(
        '--image-height',
        type=int,
        default=1080,
        help='Specify original image height'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=8000,
        help='Warmup steps for learning rate schedule'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Peek learning rate'
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
    all_classes = args.all_classes
    freeze_base = args.freeze_base
    freeze_ssd = args.freeze_ssd
    freeze_deeplab = args.freeze_deeplab
    model_width = args.model_width
    image_width = args.image_width
    image_height = args.image_height
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate
    if logs and not os.path.exists(logs):
        os.makedirs(logs)

    # number of classes
    if all_classes:
        n_seg = len(rs19.seg_classes)
        n_det = len(rs19.det_classes)
    else:
        n_seg = len(rs19.seg_subset)
        n_det = len(rs19.det_subset)

    # build model
    models = ssd_deeplab_model((model_width, model_width), n_det, n_seg)
    model, default_boxes_cwh, base, deeplab, ssd = models

    # find SSD & Deeplab layers
    base_names = {layer.name for layer in base.layers}
    deeplab_names = {layer.name for layer in deeplab.layers} - base_names
    ssd_names = {layer.name for layer in ssd.layers} - base_names

    # epoch counter
    epoch = 0

    # load model if fine tuning
    if in_model:
        model = tf.keras.models.load_model(in_model)
        # read epoch file
        try:
            with open(f"{in_model}/epoch.txt") as f:
                epoch = int(f.readline())
        except OSError:
            pass

    # Freeze weights as requested
    if freeze_ssd:
        for l_name in ssd_names:
            model.get_layer(l_name).trainable = False
    if freeze_deeplab:
        for l_name in deeplab_names:
            model.get_layer(l_name).trainable = False
    if freeze_base:
        for layer in model.layers:
            if layer.name not in ssd_names and layer.name not in deeplab_names:
                layer.trainable = False
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

    # Load training & validation data
    train_ds_orig = read_tfrecords(
            os.path.join(tfrecdir, 'train.tfrec'))
    val_ds_orig = read_tfrecords(
            os.path.join(tfrecdir, 'val.tfrec'))

    # Augment data
    augmentor = Augment(image_height, image_width)
    augmentor.crop_and_pad()
    augmentor.horizontal_flip()
    augmentor.hsv()
    augmentor.random_brightness_contrast()

    if augment:
        train_ds_aug = train_ds_orig.map(augmentor.tf_wrap())
    else:
        train_ds_aug = train_ds_orig

    # Filter for classes of interest
    if all_classes:
        train_ds_filtered = train_ds_aug
        val_ds_filtered = val_ds_orig
    else:
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

    # Filter out empty samples
    train_ds_filtered = train_ds_filtered.filter(filter_empty_samples)
    val_ds_filtered = val_ds_filtered.filter(filter_empty_samples)

    # Preprocess data
    train_ds = train_ds_filtered.map(
        preprocess((model_width, model_width), bbox_util, n_seg))
    val_ds = val_ds_filtered.map(
        preprocess((model_width, model_width), bbox_util, n_seg))

    # Create batches
    train_ds_batch = train_ds.batch(batch_size=batch_size)
    val_ds_batch = val_ds.batch(batch_size=batch_size)

    # Loss functions
    losses = combined_losses(SSDLosses(3), DeeplabLoss())
    loss_weights = (ssd_weight, deeplab_weight)
    # learning rate
    lr_schedule = CustomSchedule(learning_rate, warmup_steps)
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.Adam()

    # open logfile
    if logs:
        csv_file = open(f"{logs}/history.csv", "a", newline="")
        csv_writer = csv.writer(csv_file)

    # minimum loss so far
    min_loss = infty

    # perform training
    for epoch in tqdm(range(epoch, epoch + num_epochs)):
        out = [epoch+1]
        train_conf_loss = 0.0
        train_locs_loss = 0.0
        train_segs_loss = 0.0
        train_loss = 0.0
        train_lr = 0.0
        train_num = 0
        start_time = time.time()
        for i, batch in enumerate(train_ds_batch):
            img, gt = batch
            l, c_l, l_l, s_l = training_step(
                img, gt, model, losses, optimizer, loss_weights)
            train_conf_loss += c_l.numpy()
            train_locs_loss += l_l.numpy()
            train_segs_loss += s_l.numpy()
            train_loss += l.numpy()
            train_lr += optimizer._decayed_lr(tf.float32).numpy()
            train_num += 1
        train_time = time.time() - start_time
        train_conf_loss /= train_num
        train_locs_loss /= train_num
        train_segs_loss /= train_num
        train_loss /= train_num
        train_lr /= train_num
        out += [train_time, train_loss, train_conf_loss, train_locs_loss,
                train_segs_loss, train_lr]
        # validation run
        val_conf_loss = 0.0
        val_locs_loss = 0.0
        val_segs_loss = 0.0
        val_loss = 0.0
        val_num = 0
        start_time = time.time()
        for i, batch in enumerate(val_ds_batch):
            img, gt = batch
            pr = model(img)
            l_list = losses(gt, pr)
            c_l, l_l, s_l = l_list
            val_conf_loss += c_l.numpy()
            val_locs_loss += l_l.numpy()
            val_segs_loss += s_l.numpy()
            val_num += 1
        val_time = time.time() - start_time
        val_conf_loss /= val_num
        val_locs_loss /= val_num
        val_segs_loss /= val_num
        val_loss = ((val_conf_loss + val_locs_loss) * loss_weights[0] +
                    val_segs_loss * loss_weights[1])
        out += [val_time, val_loss, val_conf_loss, val_locs_loss,
                val_segs_loss]
        if logs:
            csv_writer.writerow(out)
            csv_file.flush()
        # best model so far?
        if (val_loss < min_loss) and out_model:
            print(f"New minimum loss {val_loss} - saving model")
            min_loss = val_loss
            model.save(out_model)
            # write epoch file
            with open(f"{out_model}/epoch.txt", "w") as f:
                f.write(str(epoch + 1))

    # close log file
    if logs:
        csv_file.close()

    # save model
    if out_model:
        model.save(out_model)
        # write epoch file
        with open(f"{out_model}/epoch.txt", "w") as f:
            f.write(str(epoch + 1))


if __name__ == '__main__':
    main()
