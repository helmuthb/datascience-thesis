import numpy as np
import os
import random
import argparse
import json
from tfrecords import BBox, write_tfrecord


# Object classes (for bounding boxes) which are used in RailSem
CLASSES = ["background", "buffer-stop", "crossing", "switch-indicator",
           "switch-left", "switch-right", "switch-static", "switch-unknown",
           "track-signal-back", "track-signal-front", "track-sign-front"]


def parse_json(json_data):
    """Parse JSON file for objects.
    Args:
        json_data (object): Data from JSON-file for a record.
    Returns:
        objects (list): Bounding boxes with label index
    """
    image_width = json_data["imgWidth"]
    image_height = json_data["imgHeight"]

    objects = []
    frame = json_data["frame"]
    for o in json_data["objects"]:
        label = o["label"]
        # we focus on object detection
        if "boundingbox" not in o:
            continue
        bb = o["boundingbox"]
        x0 = (bb[0]-1) / image_width
        y0 = (bb[1]-1) / image_height
        x1 = (bb[2]-1) / image_width
        y1 = (bb[3]-1) / image_height
        # check for plausibility
        is_ok = True
        if x0 >= x1 or y0 >= y1:
            print(f"Frame {frame} has empty bounding box for label {label}")
            is_ok = False
        if label not in CLASSES:
            print(f"Frame {frame} contains unknown label {label}")
            is_ok = False
        if not is_ok:
            # skip this bounding box
            continue
        box = BBox(CLASSES.index(label), x0, y0, x1, y1)
        objects.append(box)
    # no box found? add background box
    if len(objects) == 0:
        box = BBox(0, 0., 0., 1., 1.)
        objects.append(box)
    return objects


def read_folder(root):
    """Create data list from the root folder.
    Args:
        root (string): Root folder of RailSem19.
    Returns:
        dataset (list(dict)): Data represented by {'image_path', 'objects'}.
    """
    jpegs_path = os.path.join(root, 'jpgs/rs19_val')
    jsons_path = os.path.join(root, 'jsons/rs19_val')

    dataset = list()
    for f in os.listdir(jsons_path):
        # read JSON file
        with open(os.path.join(jsons_path, f), 'r') as json_file:
            json_data = json.loads(json_file.read())
        # parse objects
        objects = parse_json(json_data)
        # get JPEG file path
        frame = json_data['frame']
        jpeg_path = os.path.join(jpegs_path, frame + ".jpg")
        # append info to dataset
        dataset.append({'image_path': jpeg_path, 'objects': objects})
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Make RailSem19 dataset TFRecords."
    )
    parser.add_argument(
        '--source',
        type=str,
        help="Root directory of RailSem19 dataset.",
        required=True
    )
    parser.add_argument(
        '--target',
        type=str,
        help="Where to save TFRecords.",
        required=True
    )
    parser.add_argument(
        '--test_split',
        type=float,
        help="Percentage of test data.",
        default=0.15
    )
    parser.add_argument(
        '--val_split',
        type=float,
        help="Percentage of validation data.",
        default=0.15
    )
    args = parser.parse_args()

    # Disable any CUDA devices - we don't need them here
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    # Create data list from source folder
    data = read_folder(args.source)
    # shuffle dataset
    n = len(data)
    shuffled_index = np.arange(n)
    random.seed(42)
    random.shuffle(shuffled_index)
    data = [data[i] for i in shuffled_index]
    # Split into training / validation / test data
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)
    n_train = n - n_test - n_val
    last_train = n_train
    last_val = n_train + n_val
    data_train = data[:last_train]
    data_val = data[last_train:last_val]
    data_test = data[last_val:]
    print(f"Training size {len(data_train)}, validation size {len(data_val)},"
          f"test size {len(data_test)}, total {n}")

    # make output folder (and path to it) if missing
    os.makedirs(args.output_folder, exist_ok=True)
    # write TFRecords
    write_tfrecord(data_train, os.path.join(args.target, 'det_train.tfrec'))
    write_tfrecord(data_val, os.path.join(args.target, 'det_val.tfrec'))
    write_tfrecord(data_test, os.path.join(args.target, 'det_test.tfrec'))


if __name__ == '__main__':
    main()
