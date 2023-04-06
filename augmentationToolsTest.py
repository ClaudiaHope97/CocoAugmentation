import json
import unittest
import cv2

from augmentationTools import Rotator, HorizontalShifter, VerticalShifter, NoiseAdder, VerticalFlipper, \
    HorizontalFlipper

'''
This is a test script to visualize the output of the different augmentation tools.
You need to specify the image path, the image id and the annotation file path.
The original and modified image will be displayed, press "0" to exit.
'''


class AugmentationToolsTest(unittest.TestCase):
    image_path = "val2017/000000000139.jpg"
    image_id = 139
    annotations_path = "annotations/instances_val2017.json"

    def __init__(self, *args, **kwargs):
        super(AugmentationToolsTest, self).__init__(*args, **kwargs)
        self.image = cv2.imread(self.image_path)
        with open(self.annotations_path) as f:
            annotation_dict = json.load(f)
        self.annotations = [annot for annot in annotation_dict['annotations'] if annot['image_id'] == self.image_id]

    def show(self, modified_im, modified_annot):
        # draw bounding boxes on original image
        image_original = self.image
        for annot in self.annotations:
            bbx = annot['bbox']
            cv2.rectangle(image_original, (int(bbx[0]), int(bbx[1])), (int(bbx[0] + bbx[2]), int(bbx[1] + bbx[3])), (0, 255, 0), 2)
        cv2.imshow('Original image', image_original)

        # draw bounding boxes on modified image
        for annot in modified_annot:
            bbx = annot['bbox']
            cv2.rectangle(modified_im, (int(bbx[0]), int(bbx[1])), (int(bbx[0] + bbx[2]), int(bbx[1] + bbx[3])), (0, 255, 0), 2)
        cv2.imshow('Modified image', modified_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_rotation(self):
        rotator = Rotator(-30, 30)
        rotated_im, rotated_annot = rotator.modify(self.image, self.annotations)
        self.show(modified_im=rotated_im, modified_annot=rotated_annot)

    def test_h_shift(self):
        shifter = HorizontalShifter(0.2)
        shifted_im, shifted_annot = shifter.modify(self.image, self.annotations)
        self.show(modified_im=shifted_im, modified_annot=shifted_annot)

    def test_v_shift(self):
        shifter = VerticalShifter(0.2)
        shifted_im, shifted_annot = shifter.modify(self.image, self.annotations)
        self.show(modified_im=shifted_im, modified_annot=shifted_annot)

    def test_noise(self):
        color_changer = NoiseAdder(0.7)
        changed_im, changed_annot = color_changer.modify(self.image, self.annotations)
        self.show(modified_im=changed_im, modified_annot=changed_annot)

    def test_h_flip(self):
        flipper = HorizontalFlipper()
        flipped_im, flipped_annot = flipper.modify(self.image, self.annotations)
        self.show(modified_im=flipped_im, modified_annot=flipped_annot)

    def test_v_flip(self):
        flipper = VerticalFlipper()
        flipped_im, flipped_annot = flipper.modify(self.image, self.annotations)
        self.show(modified_im=flipped_im, modified_annot=flipped_annot)


if __name__ == '__main__':
    unittest.main()
