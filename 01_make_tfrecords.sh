#! /bin/sh

# Create TF-records for Object Detection
python3 make_tfrecords_det.py \
    --source ../railsem19 \
    --target ../tf_records

# Create TF-records for Semantic Segmentation
python3 make_tfrecords_seg.py \
    --source ../railsem19 \
    --target ../tf_records
