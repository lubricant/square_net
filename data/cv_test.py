import struct
import math

import cv2

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from data import CasiaFile


def gabor_filters():
    filters = []
    ksize = [7, 9, 11, 13, 15, 17]  # gabor尺度，6个
    lamda = np.pi / 2.0  # 波长
    for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向，0°，45°，90°，135°，共四个
        for K in range(6):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def proc(image, filters):
    res = [] #滤波结果
    for i in range(len(filters)):
        res1 = process(image, filters[i])
        res.append(np.asarray(res1))

    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(4,6,temp+1)
        plt.imshow(res[temp], cmap='gray' )
    plt.show()


def binary(image):
    a = cv2.threshold(img, np.mean(image), 255, cv2.THRESH_TOZERO)
    b = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(a[1], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(b[1], cmap='gray')
    plt.show()


for ch, img in CasiaFile('1001-c.gnt'):
    # print(ch)
    # proc(img, gabor_filters())
    # binary(img)
    break


