# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import re


# Requires python-qt4 to be installed
# Used to display correctly images
plt.switch_backend('Qt4Agg')

IMG_DIR_PATH = "train_images/"
IMG_EXT = (r'.*\.jpg',)
IMG_EXT_RE = [re.compile(pattern) for pattern in IMG_EXT]


class Image(object):
    """
    The image class manage image processing
    """

    def __init__(self, filename):
        self.filename = IMG_DIR_PATH + filename
        self.img = cv2.imread(self.filename)
        self.img_gray = cv2.imread(self.filename, 0)

    def process(self):
        self.laplacian = cv2.Laplacian(self.img_gray, cv2.CV_64F)
        self.edges = cv2.Canny(self.img_gray, 150, 300)
        self.hist = cv2.calcHist([self.img_gray], [0], None, [256], [0, 256])

    def show(self, img=None):
        if img:
            plt.imshow(img)
        else:
            plt.imshow(self.img)
        plt.title(self.filename)
        plt.show()

    def output(self, filename):
        # write the features extracted from the image on the filename
        pass


def walk_directory():
    img_files = []
    files = os.listdir(IMG_DIR_PATH)
    for file in files:
        for pattern in IMG_EXT_RE:
            if re.match(pattern, file):
                img_files.append(file)
                break
    return img_files
