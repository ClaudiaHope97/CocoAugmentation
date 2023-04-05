import argparse
import os
import json
import sys
import yaml


def augment_dataset(images_dir, annotation_dir, config_file_dir, output_dir):
    with open(annotation_dir) as ann_file:
        annotations = json.load(ann_file)

    with open(config_file_dir) as conf_file:
        configs = yaml.safe_load(conf_file)

    for file_name in os.listdir(images_dir):
        continue


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

