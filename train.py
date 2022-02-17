import csv
from inspect import getsourcefile
import math
import os
import time
import argparse
from numpy import infty
import yaml

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from contextlib import redirect_stdout

from tqdm import tqdm

from lib.augment import Augment
from lib.np_bbox_utils import BBoxUtils as BBoxUtilsNp
from lib.tf_bbox_utils import BBoxUtils as BBoxUtilsTf
from lib.preprocess import (
    filter_empty_samples, filter_no_mask, preprocess_np, preprocess_tf)
from lib.tfr_utils import read_tfrecords
from lib.losses import SSDLosses, DeeplabLoss
from lib.combined import get_training_step, ssd_deeplab_model, loss_list


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
        '--det-weight',
        type=float,
        default=1.0,
        help='Weight for object detection training (default = 1.0).'
    )
    parser.add_argument(
        '--seg-weight',
        type=float,
        default=1.0,
        help='Weight for segmentation training (default = 1.0).'
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
        '--freeze-base-epochs',
        type=int,
        default=5,
        help='Freeze base layers for number of epochs.'
    )
    parser.add_argument(
        '--freeze-det',
        action='store_true',
        help='Freeze object detection layers.'
    )
    parser.add_argument(
        '--freeze-seg',
        action='store_true',
        help='Freeze segmentation layers.'
    )
    parser.add_argument(
        '--det-num-classes',
        type=int,
        default=0,
        help='Number of classes for object detection (0 = segmentation only).'
    )
    parser.add_argument(
        '--seg-num-classes',
        type=int,
        default=0,
        help='Number of classes for segmentation (0 = object detection only).'
    )
    parser.add_argument(
        '--use-numpy',
        action='store_true',
        help='Use (slower) numpy for encoding of ground truth.'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        help='Specify configuration yaml file for model.',
        required=True
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
        default=2,
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
        default=1e-2,
        help='Peak learning rate after warmup.'
    )
    parser.add_argument(
        '--l2-weight',
        type=float,
        default=5e-5,
        help='L2 normalization weight.'
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
        default=50,
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
    det_weight = args.det_weight
    seg_weight = args.seg_weight
    batch_size = args.batch_size
    freeze_base_epochs = args.freeze_base_epochs
    freeze_det = args.freeze_det
    freeze_seg = args.freeze_seg
    n_seg = args.seg_num_classes
    n_det = args.det_num_classes
    use_numpy = args.use_numpy
    model_config = args.model_config
    image_width = args.image_width
    image_height = args.image_height
    warmup_epochs = args.warmup_epochs
    warmup_learning_rate = args.warmup_learning_rate
    learning_rate = args.learning_rate
    l2_weight = args.l2_weight
    decay_factor = args.decay_factor
    stop_after = args.stop_after
    # checks for consistency
    if n_det == 0 and n_seg == 0:
        print("Number of classes is 0 for all - no training at all.")
        return
    # create folder for log files
    if logs and not os.path.exists(logs):
        os.makedirs(logs)

    # read model config
    if not os.path.exists(model_config):
        # current script folder ...
        folder = os.path.dirname(getsourcefile(main))
        model_config = f"{folder}/config/{model_config}.cfg"
    with open(model_config, 'r') as cf:
        config = yaml.safe_load(cf)

    # load model width from config
    model_width = config['width']

    # build model
    models = ssd_deeplab_model(n_det, n_seg, config)
    model, base, deeplab, ssd, default_boxes_cw, prep = models

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

    # No object detection or segmentation?
    if n_seg == 0:
        model = ssd
    elif n_det == 0:
        model = deeplab

    # load model if provided
    if in_model:
        model = tf.keras.models.load_model(in_model)

    # Loss functions
    losses = loss_list(SSDLosses(3), DeeplabLoss(), n_det, n_seg)
    # weights for losses
    if n_seg == 0:
        loss_weights = (det_weight, det_weight)
    elif n_det == 0:
        loss_weights = (seg_weight, )
    else:
        loss_weights = (det_weight, det_weight, seg_weight)

    # Freeze weights as requested
    if freeze_det:
        for l_name in ssd_names:
            model.get_layer(l_name).trainable = False
    if freeze_seg:
        for l_name in deeplab_names:
            model.get_layer(l_name).trainable = False
    if freeze_base_epochs != 0:
        for layer in model.layers:
            if layer.name not in ssd_names and layer.name not in deeplab_names:
                layer.trainable = False

    # Bounding box utility object
    if use_numpy:
        BBoxUtils = BBoxUtilsNp
    else:
        BBoxUtils = BBoxUtilsTf
    bbox_util = None if n_det == 0 else BBoxUtils(n_det, default_boxes_cw)

    # Load training & validation data
    train_ds = read_tfrecords(f"{tfrecdir}/train.tfrec", shuffle=True)
    val_ds = read_tfrecords(f"{tfrecdir}/val.tfrec")

    # Augment data
    if augment:
        augmentor = Augment(image_height, image_width)
        augmentor.crop_and_pad()
        augmentor.horizontal_flip()
        augmentor.hsv()
        augmentor.random_brightness_contrast()
        train_ds = train_ds.map(
            augmentor.tf_wrap(),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Filter out empty samples - if object detection requested
    if n_det > 0:
        train_ds = train_ds.filter(filter_empty_samples)
        val_ds = val_ds.filter(filter_empty_samples)

    # Filter out missing masks - if segmentation requested
    if n_seg > 0:
        train_ds = train_ds.filter(filter_no_mask)
        val_ds = val_ds.filter(filter_no_mask)

    # Count elements
    epoch_size = sum(1 for _ in train_ds)
    num_batches = math.ceil(epoch_size / batch_size)

    # Preprocess data
    if use_numpy:
        train_ds = train_ds.map(
            preprocess_np(prep, (model_width, model_width), bbox_util, n_seg),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            preprocess_np(prep, (model_width, model_width), bbox_util, n_seg),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        train_ds = train_ds.map(
            preprocess_tf(prep, (model_width, model_width), bbox_util, n_seg),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            preprocess_tf(prep, (model_width, model_width), bbox_util, n_seg),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Shuffle & create batches
    train_ds_batch = (
        train_ds.shuffle(100)
                .prefetch(tf.data.AUTOTUNE)
                .batch(batch_size=batch_size)
    )
    val_ds_batch = (
        val_ds.prefetch(tf.data.AUTOTUNE)
              .cache()
              .batch(batch_size=batch_size)
    )

    # learning rate
    if warmup_epochs > 0:
        lr = warmup_learning_rate
    else:
        lr = learning_rate
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lambda: lr)
    # training step
    training_step = get_training_step(model, losses, loss_weights,
                                      optimizer, n_det, n_seg, l2_weight)

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
        for batch in tqdm(
                iterable=train_ds_batch,
                desc=f"Epoch {epoch+1}",
                unit='bt',
                total=num_batches):
            img, gt = batch
            ll = training_step(img, gt)
            if n_seg == 0:
                train_conf_loss += ll[1].numpy()
                train_locs_loss += ll[2].numpy()
            elif n_det == 0:
                train_segs_loss += ll[1].numpy()
            else:
                train_conf_loss += ll[1].numpy()
                train_locs_loss += ll[2].numpy()
                train_segs_loss += ll[3].numpy()
            train_loss += sum([li*wi for li, wi in zip(ll[1:], loss_weights)])
            train_loss = train_loss.numpy()
            # break in case of NaN
            if train_loss != train_loss:
                break
            train_num += 1
        # break in case of NaN
        if train_loss != train_loss:
            print("NaN detected - ending training")
            break
        train_time = time.time() - start_time
        train_conf_loss /= train_num
        train_locs_loss /= train_num
        train_segs_loss /= train_num
        train_loss /= train_num
        out = [epoch+1, lr, train_time, train_loss]
        if n_det > 0:
            out += [train_conf_loss, train_locs_loss]
        if n_seg > 0:
            out += [train_segs_loss]
        print(f"Epoch {epoch+1}: lr={lr}, time={train_time}, "
              f"loss={train_loss}")
        # validation run
        val_conf_loss = 0.0
        val_locs_loss = 0.0
        val_segs_loss = 0.0
        val_loss = 0.0
        val_num = 0
        start_time = time.time()
        for batch in val_ds_batch:
            img, gt = batch
            pr = model(img, training=False)
            ll = losses(gt, pr)
            if n_seg == 0:
                val_conf_loss += ll[0].numpy()
                val_locs_loss += ll[1].numpy()
            elif n_det == 0:
                val_segs_loss += ll[0].numpy()
            else:
                val_conf_loss += ll[0].numpy()
                val_locs_loss += ll[1].numpy()
                val_segs_loss += ll[2].numpy()
            val_loss += sum([li*wi for li, wi in zip(ll, loss_weights)])
            val_loss = val_loss.numpy()
            val_num += 1
        val_time = time.time() - start_time
        val_conf_loss /= val_num
        val_locs_loss /= val_num
        val_segs_loss /= val_num
        val_loss /= val_num
        out += [val_time, val_loss]
        if n_det > 0:
            out += [val_conf_loss, val_locs_loss]
        if n_seg > 0:
            out += [val_segs_loss]
        print(f"Epoch {epoch+1}: val. time={val_time}, val. loss={val_loss}")
        if logs:
            csv_writer.writerow(out)
            csv_file.flush()
        # best model so far?
        if (val_loss < min_loss) and out_model:
            print(f"New minimum loss {val_loss} - saving model")
            min_loss = val_loss
            model.save(out_model)
            non_improved = 0
        elif epoch > warmup_epochs:
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
