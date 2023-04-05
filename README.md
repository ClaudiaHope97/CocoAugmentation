# CocoAugmentation

This is a tool to generate a new set of images and annotations through clever transformation of the available ones.
It is applied to the COCO dataset

## Installing

NOTE: this repo has been tested using python 3.8

Cloning the repo and installing required libraries

```
git clone https://github.com/ClaudiaHope97/CocoAugmentation.git
cd CocoAugmentation
pip install -r requirements.txt
```

In case you need to download the COCO dataset and annotation
```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```
otherwise you can specify the images folder and annotation file in the run script (please note that the annotations need to be in COCO format)

## Quick start