# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:48:42 2016

@author: ubuntu
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import re
import time 

# Requires python-qt4 to be installed
# Used to display correctly images
plt.switch_backend('Qt4Agg')

IMG_DIR_PATH = "img/"
IMG_EXT = (r'.*\.jpg',)
IMG_EXT_RE = [re.compile(pattern) for pattern in IMG_EXT]
IMG_DIST_GRAD ="grad_img/"
IMG_DIST_EDGE = "edge_img/"
IMG_DIST_ABS ="abs_img/"
IMG_DIST_PHASE = "phase_img/"

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
    
    def fourier(self):
        FS = np.fft.fft2(self.img_gray)
        self.log_shift_abs = np.log(np.abs(np.fft.fftshift(FS))**2)
        self.shift_phase = np.angle(np.fft.fftshift(FS), deg=True)
        self.imag = np.imag(np.fft.fftshift(FS))
        self.real = np.real(np.fft.fftshift(FS))


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
    
if __name__ == '__main__':
    start_time = time.time()
    img_files = walk_directory()
    for imgName in img_files:
        img = Image(imgName)
        img.process()
        img.fourier()
        cv2.imwrite(IMG_DIST_GRAD+ "grad_" + imgName, img.laplacian)
        cv2.imwrite(IMG_DIST_EDGE+ "edge_" + imgName, img.edges)
        cv2.imwrite(IMG_DIST_ABS+ "abs_" + imgName, img.log_shift_abs)
        cv2.imwrite(IMG_DIST_PHASE+ "phase_" + imgName, img.shift_phase)
    elapsed_time = time.time() - start_time
    print elapsed_time

