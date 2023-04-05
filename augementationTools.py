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
        bbx[2] = np.clip(bbx[2], 0, image_w - bbx[0])
        bbx[3] = np.clip(bbx[3], 0, image_h - bbx[1])
        return bbx

    @staticmethod
    def is_visible(old_bbx: list, new_bbx: list):
        # check if the new bbx is visible compared to the original one
        # (it is visible if the new area is more than 10% of the old one)
        old_area = old_bbx[2] * old_bbx[3]
        new_area = new_bbx[2] * new_bbx[3]
        if new_area / old_area > 0.1:
            return True
        return False


class Rotator(BaseAugmentor):

    def __init__(self, rot_min, rot_max) -> None:
        super().__init__()
        self.rot_max = rot_max
        self.rot_min = rot_min

    def

    def modify(self, image, annotations: list):

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

        rotated_annot = annotations
        for annotation in rotated_annot:
            # get the four rotated corners of the bounding box
            bbx = annotation['bbox']
            vec1 = np.matmul(M, np.array([bbx[0], bbx[1], 1], dtype=np.float64))  # top left corner transformed
            vec2 = np.matmul(M, np.array([bbx[2], bbx[1], 1], dtype=np.float64))  # top right corner transformed
            vec3 = np.matmul(M, np.array([bbx[0], bbx[3], 1], dtype=np.float64))  # bottom left corner transformed
            vec4 = np.matmul(M, np.array([bbx[2], bbx[3], 1], dtype=np.float64))  # bottom right corner transformed
            x_vals = [vec1[0], vec2[0], vec3[0], vec4[0]]
            y_vals = [vec1[1], vec2[1], vec3[1], vec4[1]]
            x_min = math.ceil(np.min(x_vals))
            x_max = math.floor(np.max(x_vals))
            y_min = math.ceil(np.min(y_vals))
            y_max = math.floor(np.max(y_vals))
            bbx = [x_min, y_min, x_max, y_max]
            annotation['bbox'] = bbx

        rotated_im, rotated_annot = resizeImageAndBoxes(rotated_im, w, h, rotated_annot)

        return rotated_im, rotated_annot
