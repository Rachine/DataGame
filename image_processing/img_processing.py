# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import re


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

    def process(self):
        pass

    def show(self):
        # Requires python-qt4 to be installed
        # Used to display correctly images
        plt.switch_backend('Qt4Agg')
        plt.imshow(self.img)
        plt.title(self.filename)
        plt.show()

    def output(self):
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
