# CocoAugmentation

This is a tool to generate a new set of images and corresponding annotations through clever transformation of the available ones. It uses only numpy and OpenCV libraries. It can be applied to any dataset with annotations in COCO format. \
The available transformations are the following: 
* ROTATION : random rotation of the image
* SHIFT : vertical and horizontal shift of the image
* NOISE ADDITION : adding random gaussian noise to the image colors
* FLIP : horizontal and vertical flip of the image


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

You can run the *run.sh* script specifying the following parameters:
* --images_dir : path to the images folder
* --annotation_dir : path to the annotation json file
* --config_file_dir : path to the yaml configuration file
* --output_dir : path to the output directory

as a result a set of modified images will be created in th output folder along with a json fild with the new annotation (only bounding boxes are updated in this code). The output set has the same size of the input one, from each input image we generate a modified one using specific transformations with defined probability.

### Config File
It is possible to configure some parameters for the data augmentation tool using the *config.yaml* file.\
In particular, you can define the probability of a certain transformation to happen as a number between 0 and 1.\
For example, if you never want to apply a specific transformation to your dataset you can set the corresponding probability to 0.
Moreover, you can select the range or values specific to each type of transformation.\
Please refer to the config file example for further details.

### Visual testing
You can also test the visual output of each transformation using the python test file *AugmentationToolsTest.py*, where you can define a specific image and run different tests related to different augmentations.