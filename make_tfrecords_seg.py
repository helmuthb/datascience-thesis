import os
import random
import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm
from tfrecords_seg import write_tfrecord


# Object classes which are used in RailSem
CLASSES = ["background", "buffer-stop", "crossing", "guard-rail", "train-car",
           "platform", "rail", "switch-indicator", "switch-left",
           "switch-unknown", "switch-static", "track-sign-front",
           "track-signal-front", "track-signal-back", "person-group",
           "car", "fence", "person", "pole", "rail-occluder", "truck"]


def draw_json(json_data):
    """Parse a JSON file for objects, drawing them on a numpy array.
    Args:
        json_data (object): Data from JSON-file for a record.
    Returns:
        img (ndarray, 8bit): Resulting image as numpy array.
    """
    image_width = json_data["imgWidth"]
    image_height = json_data["imgHeight"]
    # create target image
    img = np.zeros((image_width, image_height), np.uint8)
    # go through objects in JSON
    for o in json_data["objects"]:
        # we focus on segmentation
        if "boundingbox" in o:
            continue
        # get label & label-id
        label = o["label"]
        label_id = CLASSES.index(label)
        if "polygon" in o:
            # get points in polygon
            poly_points = np.array(o["polygon"])
            # round to next integer
            png_draw = np.around(poly_points).astype(np.int32)
            # draw on target image
            cv2.fillPoly(img, [png_draw], color=label_id)
        if "polyline" in o:
            # get points in polyline
            poly_points = np.array(o["polyline"])
            # round to next integer
            png_draw = np.around(poly_points).astype(np.int32)
            # draw on target image
            cv2.polylines(
                img,
                [png_draw],
                isClosed=False,
                color=label_id,
                thickness=2)
        if "polyline-pair" in o:
            # get left/right part of pair
            poly_points_left = np.array(o["polyline-pair"][0])
            poly_points_right = np.array(o["polyline-pair"][1])
            # round to next integer
            png_draw_left = np.around(poly_points_left).astype(np.int32)
            png_draw_right = np.around(poly_points_right).astype(np.int32)
            # draw on target image
            cv2.fillPoly(
                img,
                [png_draw_left, png_draw_right],
                color=label_id)
    # return resulting image (numpy array)
    return img


def read_folder(root, png_folder):
    """Create data list from the root folder.
    Args:
        root (string): Root folder of RailSem19.
        png_folder (string): Folder for PNG files to be created.
    Returns:
        dataset (list(dict)): Data represented by {'image_path', 'classes_path'}.
    """
    jpegs_path = os.path.join(root, 'jpgs/rs19_val')
    jsons_path = os.path.join(root, 'jsons/rs19_val')

    dataset = list()
    for f in tqdm(os.listdir(jsons_path)):
        # read JSON file
        with open(os.path.join(jsons_path, f), 'r') as json_file:
            json_data = json.loads(json_file.read())
        # draw objects as numpy array
        objects = draw_json(json_data)
        # get JPEG & PNG file path
        frame = json_data['frame']
        jpeg_path = os.path.join(jpegs_path, frame + ".jpg")
        png_path = os.path.join(png_folder, frame + ".png")
        # save objects to PNG
        cv2.imwrite(png_path, objects)
        # append info to dataset
        dataset.append({'image_path': jpeg_path, 'classes_path': png_path})
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Make RailSem19 dataset TFRecords for segmentation."
    )
    parser.add_argument(
        '--source',
        type=str,
        help="Root directory of RailSem19 dataset.",
        required=True
    )
    parser.add_argument(
        '--png',
        type=str,
        help="Where to save generated PNGs.",
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

    # make PNG folder (and path to it) if missing
    os.makedirs(args.png, exist_ok=True)
    # Create data list from source folder
    # and corresponding PNG files
    print("Creating PNG versions of segmentation annotation...")
    data = read_folder(args.source, args.png)
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
    os.makedirs(args.target, exist_ok=True)
    # write TFRecords
    write_tfrecord(data_train, os.path.join(args.target, 'seg_train.tfrec'))
    write_tfrecord(data_val, os.path.join(args.target, 'seg_val.tfrec'))
    write_tfrecord(data_test, os.path.join(args.target, 'seg_test.tfrec'))


if __name__ == '__main__':
    main()
