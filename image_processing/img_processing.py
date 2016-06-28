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

    @property
    def size(self):
        return self.img.shape

    def process(self):
        self.laplacian = cv2.Laplacian(self.img_gray, cv2.CV_64F)
        self.edges = cv2.Canny(self.img_gray, 150, 250)
        self.hist = cv2.calcHist([self.img_gray], [0], None, [256], [0, 256])

    def cut_edges(self, sample=3, f=0.1):
        """
        Counts the number of times edges are crossed along the both axis
        """
        height = self.size[0]
        width = self.size[1]
        width_cuts = 0
        height_cuts = 0
        for i in list(np.random.randint(int(height/2*(1-f)), int(height/2*(1+f)), sample)):
            width_cuts += np.sum(self.edges[i, :]/255)
        for j in list(np.random.randint(int(width/2*(1-f)), int(width/2*(1+f)), sample)):
            height_cuts += np.sum(self.edges[:, j]/255)
        return (width_cuts/sample, height_cuts/sample)

    def show(self, img=None, **kwargs):
        if img is None:
            img = self.img
        plt.imshow(img, **kwargs)
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
