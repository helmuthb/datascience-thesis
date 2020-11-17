#! /bin/sh

# Create TF-records for Object Detection
python make_tfrecords_det.py \
    --source /raid/media/master/railsem19 \
    --output_folder ${TFRECORD_FOLDER} \
    --val_split 0.2

# Create TF-records for Semantic Segmentation