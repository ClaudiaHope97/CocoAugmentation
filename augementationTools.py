import numpy as np
import cv2
import math
import random


class BaseAugmentor:
    def modify(self, image, annotations: list):
        # modifies the image and all the related annotations accordingly
        pass

    def crop_bbx(self, image_w: int, image_h: int, bbx: list):
        # crop the bbx so that it is contained in the image boundaries
        if bbx[0] > image_w or bbx[1] > image_h:
            return [0, 0, 0, 0]
        if bbx[0] < 0:
            bbx[2] += bbx[0]
            bbx[0] = 0
        if bbx[1] < 0:
            bbx[3] += bbx[1]
            bbx[1] = 0
        bbx[2] = np.clip(bbx[2], 0, image_w - bbx[0])
        bbx[3] = np.clip(bbx[3], 0, image_h - bbx[1])
        return bbx

    def check_annotations(self, annotation: dict):
        # check if the annotation is valid
        return annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0


class Rotator(BaseAugmentor):

    def __init__(self, rot_min, rot_max) -> None:
        super().__init__()
        if rot_max < rot_min:
            raise Exception("Maximum rotation should be higher than the minimum one")
        self.rot_max = rot_max
        self.rot_min = rot_min

    def modify(self, image, annotations: list):
        # randomly select an angle in the range
        angle = random.randint(self.rot_min, self.rot_max)

        h, w = image.shape[0], image.shape[1]
        (cX, cY) = (w // 2, h // 2)  # original image center
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # 2 by 3 rotation matrix
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the dimensions of the rotated image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation of the new centre
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotated_im = cv2.warpAffine(image, M, (nW, nH))

        rotated_annot = []
        for annotation in annotations:
            new_annot = annotation.copy()
            # get the four rotated corners of the bounding box
            bbx = new_annot['bbox']
            vec1 = np.matmul(M, np.array([bbx[0], bbx[1], 1], dtype=np.float64))  # top left corner transformed
            vec2 = np.matmul(M, np.array([bbx[0]+bbx[2], bbx[1], 1], dtype=np.float64))  # top right corner transformed
            vec3 = np.matmul(M, np.array([bbx[0], bbx[1]+bbx[3], 1], dtype=np.float64))  # bottom left corner transformed
            vec4 = np.matmul(M, np.array([bbx[0]+bbx[2], bbx[1]+bbx[3], 1], dtype=np.float64))  # bottom right corner transformed
            x_vals = [vec1[0], vec2[0], vec3[0], vec4[0]]
            y_vals = [vec1[1], vec2[1], vec3[1], vec4[1]]
            x_min = math.ceil(np.min(x_vals))
            x_max = math.floor(np.max(x_vals))
            y_min = math.ceil(np.min(y_vals))
            y_max = math.floor(np.max(y_vals))
            bbx = [x_min, y_min, x_max-x_min, y_max-y_min]
            new_annot['bbox'] = bbx
            rotated_annot.append(new_annot)

        return rotated_im, rotated_annot


class HorizontalShifter(BaseAugmentor):

    def __init__(self, h_shift_ratio) -> None:
        super().__init__()
        if h_shift_ratio > 1 or h_shift_ratio < 0:
            raise Exception("The shift ratio should be between 0 and 1")
        self.h_shift_ratio = h_shift_ratio

    def modify(self, image, annotations: list):
        h, w = image.shape[0], image.shape[1]
        to_shift = w * random.uniform(-self.h_shift_ratio, self.h_shift_ratio)
        M = np.matrix([[1, 0, to_shift], [0, 1, 0]], dtype=np.float64)
        image_shifted = cv2.warpAffine(image, M, (w, h))

        annotations_shifted = []
        for annotation in annotations:
            new_annot = annotation.copy()
            bbx = new_annot['bbox']
            new_annot['bbox'] = self.crop_bbx(w, h, [bbx[0]+to_shift, bbx[1], bbx[2], bbx[3]])
            if self.check_annotations(new_annot):
                annotations_shifted.append(new_annot)

        return image_shifted, annotations_shifted


class VerticalShifter(BaseAugmentor):

    def __init__(self, v_shift_ratio) -> None:
        super().__init__()
        if v_shift_ratio > 1 or v_shift_ratio < 0:
            raise Exception("The shift ratio should be between 0 and 1")
        self.v_shift_ratio = v_shift_ratio

    def modify(self, image, annotations: list):
        h, w = image.shape[0], image.shape[1]
        to_shift = w * random.uniform(-self.v_shift_ratio, self.v_shift_ratio)
        M = np.matrix([[1, 0, 0], [0, 1, to_shift]], dtype=np.float64)
        image_shifted = cv2.warpAffine(image, M, (w, h))

        annotations_shifted = []
        for annotation in annotations:
            new_annot = annotation.copy()
            bbx = new_annot['bbox']
            new_annot['bbox'] = self.crop_bbx(w, h, [bbx[0], bbx[1]+to_shift, bbx[2], bbx[3]])
            if self.check_annotations(new_annot):
                annotations_shifted.append(new_annot)

        return image_shifted, annotations_shifted


class NoiseAdder(BaseAugmentor):

    def __init__(self, noise_intensity) -> None:
        super().__init__()
        if noise_intensity < 0:
            raise Exception("The noise intensity should be bigger than 0")
        self.noise_intensity = noise_intensity

    def modify(self, image, annotations: list):
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(0, 0.5, image.size)
        gaussian_noise = gaussian_noise.reshape((image.shape[0], image.shape[1], 3)).astype('uint8')
        # Add the Gaussian noise to the image
        image_with_noise = cv2.add(image, gaussian_noise)

        return image_with_noise, annotations


class HorizontalFlipper(BaseAugmentor):

    def modify(self, image, annotations: list):
        h, w = image.shape[0], image.shape[1]

        # flip image
        flipped_image = cv2.flip(image, 1)
        # flip bbx
        annotations_flipped = []
        for annotation in annotations:
            new_annot = annotation.copy()
            bbx = new_annot['bbox']
            new_annot['bbox'] = self.crop_bbx(w, h, [w-bbx[0]-bbx[2], bbx[1], bbx[2], bbx[3]])
            if self.check_annotations(new_annot):
                annotations_flipped.append(new_annot)

        return flipped_image, annotations_flipped


class VerticalFlipper(BaseAugmentor):

    def modify(self, image, annotations: list):
        h, w = image.shape[0], image.shape[1]

        # flip image
        flipped_image = cv2.flip(image, 0)
        # flip bbx
        annotations_flipped = []
        for annotation in annotations:
            new_annot = annotation.copy()
            bbx = new_annot['bbox']
            new_annot['bbox'] = self.crop_bbx(w, h, [bbx[0], h - bbx[1] - bbx[3], bbx[2], bbx[3]])
            if self.check_annotations(new_annot):
                annotations_flipped.append(new_annot)

        return flipped_image, annotations_flipped
