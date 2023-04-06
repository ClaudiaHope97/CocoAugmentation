import argparse
import os
import json
import sys

import cv2
import numpy as np
import yaml

from augementationTools import Rotator, VerticalShifter, NoiseAdder, HorizontalFlipper, VerticalFlipper, \
    HorizontalShifter


def augment_dataset(images_dir, annotation_dir, config_file_dir, output_dir):

    with open(annotation_dir) as ann_file:
        full_annotations = json.load(ann_file)

    with open(config_file_dir) as conf_file:
        configs = yaml.safe_load(conf_file)

    # generate tools for augmentation
    rotator = Rotator(configs['rotation_lb'], configs['rotation_ub'])
    h_shifter = HorizontalShifter(configs['h_shift_ratio'])
    v_shifter = VerticalShifter(configs['v_shift_ratio'])
    noise_adder = NoiseAdder(configs['noise_intensity'])
    h_flipper = HorizontalFlipper()
    v_flipper = VerticalFlipper()

    new_full_annotation = []
    for file_name in os.listdir(images_dir):
        image = cv2.imread(os.path.join(images_dir, file_name))
        image_id = [annot for annot in full_annotations['images'] if annot['file_name'] == file_name][0]['id']
        annotations = [annot for annot in full_annotations['annotations'] if annot['image_id'] == image_id]

        new_image, new_annotation = image, annotations
        # apply all transformations with the given probability
        if np.random.rand() < configs['rotation_prob']:
            new_image, new_annotation = rotator.modify(new_image, new_annotation)
        if np.random.rand() < configs['v_shift_prob']:
            new_image, new_annotation = h_shifter.modify(new_image, new_annotation)
        if np.random.rand() < configs['h_shift_prob']:
            new_image, new_annotation = v_shifter.modify(new_image, new_annotation)
        if np.random.rand() < configs['noise_prob']:
            new_image, new_annotation = noise_adder.modify(new_image, new_annotation)
        if np.random.rand() < configs['h_flip_prob']:
            new_image, new_annotation = h_flipper.modify(new_image, new_annotation)
        if np.random.rand() < configs['v_flip_prob']:
            new_image, new_annotation = v_flipper.modify(new_image, new_annotation)

        new_full_annotation += new_annotation
        # save image to the output directory
        cv2.imwrite(os.path.join(output_dir, file_name), new_image)
        print(f"Image {file_name} modified and saved")

    # save annotation file
    full_annotations['annotations'] = new_full_annotation
    with open(os.path.join(output_dir, 'augmented_annotations.json'), 'w') as f:
        json.dump(full_annotations, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='', help='original images dir')
    parser.add_argument('--annotation_dir', type=str, default='', help='original annotation file dir')
    parser.add_argument('--config_file_dir', type=str, default='configs.yaml', help='configuration file dir')
    parser.add_argument('--output_dir', type=str, default='', help='output dataset dir')
    opt = parser.parse_args()

    # check inputs
    if not os.path.exists(opt.images_dir):
        print("Invalid image directory")
        sys.exit(-1)
    if not os.path.exists(opt.annotation_dir):
        print("Invalid annotation directory")
        sys.exit(-1)
    if not os.path.exists(opt.config_file_dir):
        print("Invalid configuration file directory")
        sys.exit(-1)
    if not os.path.exists(opt.output_dir):
        if len(opt.output_dir) == 0:
            opt.output_dir = "augmented_dataset"
            os.mkdir(opt.output_dir)
        else:
            print("Invalid output directory")
            sys.exit(-1)

    try:
        augment_dataset(opt.images_dir, opt.annotation_dir, opt.config_file_dir, opt.output_dir)
    except Exception as e:
        print(e)
        sys.exit(-1)

    print("Dataset augmentation completed")

