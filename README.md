# CocoAugmentation

This is a tool to generate a new set of images and annotations through clever transformation of the available ones.
It is applied to the COCO dataset

## Installing

Cloning the repo and installing required libraries
```
git clone https://github.com/ClaudiaHope97/CocoAugmentation.git
cd CocoAugmentation
pip install -r requirements.txt
```

Downloading COCO dataset and annotation

```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

## Quick start