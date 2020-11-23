#! /bin/sh

# Create TF-records for Object Detection
python3 make_tfrecords_det.py \
    --source ../railsem19 \
    --target ../tf_records \
    --val_split 0.2

# Create TF-records for Semantic Segmentation
