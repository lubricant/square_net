import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


class Deform(object):

    (DT_LEFT, DT_RIGHT, DT_HORIZON_EXPAND, DT_HORIZON_SHRINK) = 0x5, 0x6, 0x7, 0x4

    (DT_TOP, DT_BOTTOM, DT_VERTICAL_EXPAND, DT_VERTICAL_SHRINK) = (
        DT_LEFT << 3, DT_RIGHT << 3, DT_HORIZON_EXPAND << 3, DT_HORIZON_SHRINK << 3)

    # print(bin(DT_LEFT), bin(DT_RIGHT), bin(DT_HORIZON_EXPAND), bin(DT_HORIZON_SHRINK))
    # print(bin(DT_TOP), bin(DT_BOTTOM), bin(DT_VERTICAL_EXPAND), bin(DT_VERTICAL_SHRINK))

    def __init__(self, eta=1.):
        def deform_trans(x, a, b):
            affine = (b - a) * x + a
            return math.sin(affine) * math.cos(affine) - math.sin(b) * math.cos(b)

        self.__left = self.__top = lambda x: x + eta * deform_trans(x, 0., .5)
        self.__right = self.__bottom = lambda x: x + eta * deform_trans(x, .5, 1.)
        self.__expand = lambda x: x + eta * deform_trans(x, 0., 1.)
        self.__shrink = lambda x: x - eta * deform_trans(x, 0., 1.)

    @staticmethod
    def example():
        offset = 5
        example = np.ones((100, 100))
        example[:, :] = 255
        example[offset:100:offset] = 0
        example[:, offset:100:offset] = 0
        return example

    def transform(self, image, flag):

        print(bin(0x7), bin(0x38))

        horizon, vertical = flag & 0x7, flag & 0x38
        horizon_dt, vertical_dt = None, None

        if horizon == Deform.DT_LEFT:
            horizon_dt = self.__left
        elif horizon == Deform.DT_RIGHT:
            horizon_dt = self.__right
        elif horizon == Deform.DT_HORIZON_EXPAND:
            horizon_dt = self.__expand
            print('HE')
        elif horizon == Deform.DT_HORIZON_SHRINK:
            horizon_dt = self.__shrink
            print('HS')

        if vertical == Deform.DT_TOP:
            vertical_dt = self.__top
        elif vertical == Deform.DT_BOTTOM:
            vertical_dt = self.__bottom
        elif vertical == Deform.DT_VERTICAL_EXPAND:
            vertical_dt = self.__expand
            print('VE')
        elif vertical == Deform.DT_VERTICAL_SHRINK:
            vertical_dt = self.__shrink
            print('VS')

        image_dt = np.zeros(image.shape)

        rows, cols = image.shape
        for y in np.arange(0, rows):
            y_dt = (vertical_dt(y/rows) * rows) if vertical_dt else y
            y_dt = np.minimum(np.maximum(round(y_dt), 0), rows-1)
            for x in np.arange(0, cols):
                x_dt = (horizon_dt(x/cols) * cols) if horizon_dt else x
                x_dt = np.minimum(np.maximum(round(x_dt), 0), cols-1)
                image_dt[y, x] = image[y_dt, x_dt]

        return image_dt


def deform(eta=1, image=Deform.example()):
    trans = Deform(eta)
    plt.subplot(2, 4, 1)
    plt.imshow(trans.transform(image, Deform.DT_LEFT), cmap='gray')
    plt.subplot(2, 4, 2)
    plt.imshow(trans.transform(image, Deform.DT_RIGHT), cmap='gray')
    plt.subplot(2, 4, 3)
    plt.imshow(trans.transform(image, Deform.DT_HORIZON_EXPAND), cmap='gray')
    plt.subplot(2, 4, 4)
    plt.imshow(trans.transform(image, Deform.DT_HORIZON_SHRINK), cmap='gray')

    plt.subplot(2, 4, 5)
    plt.imshow(trans.transform(image, Deform.DT_TOP), cmap='gray')
    plt.subplot(2, 4, 6)
    plt.imshow(trans.transform(image, Deform.DT_BOTTOM), cmap='gray')
    plt.subplot(2, 4, 7)
    plt.imshow(trans.transform(image, Deform.DT_VERTICAL_EXPAND), cmap='gray')
    plt.subplot(2, 4, 8)
    plt.imshow(trans.transform(image, Deform.DT_VERTICAL_SHRINK), cmap='gray')


fig, ax = plt.subplots()
slid = Slider(ax, 'eta', 0.1, 2.0, valinit=1)

def update(val):
    eta = slid.val
    deform(eta)
    fig.canvas.draw_idle()

slid.on_changed(update)

plt.show()
