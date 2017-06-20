import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from data import CasiaFile


class Distort(object):

    def __init__(self, aa, k=0.3):

        def distort_trans(a, t):
            assert 0 <= t <= 1
            a = a if t <= .5 else -a
            b = 1 if t <= .5 else 0
            t = 2 * (t if t <= .5 else t - .5)
            return .5 * (b + (1-math.exp(-a*t))/(1-math.exp(-a)))

        self.__test = lambda x: k * x + distort_trans(aa, x)

    def transform(self, image):

        rows, cols = image.shape
        image_dt = np.zeros(image.shape)
        for y in np.arange(0., rows):
            y_dt = y
            for x in np.arange(0., cols):
                x_dt = (self.__test(x/cols) * cols)
                x_dt = np.minimum(np.maximum(round(x_dt), 0), cols-1)
                image_dt[y, x] = image[y_dt, x_dt]

        return image_dt


class Deform(object):

    (DT_LEFT, DT_RIGHT, DT_HORIZON_EXPAND, DT_HORIZON_SHRINK) = 0x5, 0x6, 0x7, 0x4

    (DT_TOP, DT_BOTTOM, DT_VERTICAL_EXPAND, DT_VERTICAL_SHRINK) = (
        DT_LEFT << 3, DT_RIGHT << 3, DT_HORIZON_EXPAND << 3, DT_HORIZON_SHRINK << 3)

    # print(bin(DT_LEFT), bin(DT_RIGHT), bin(DT_HORIZON_EXPAND), bin(DT_HORIZON_SHRINK))
    # print(bin(DT_TOP), bin(DT_BOTTOM), bin(DT_VERTICAL_EXPAND), bin(DT_VERTICAL_SHRINK))

    def __init__(self, eta=1.):
        def deform_trans(x, a, b):
            affine = (b - a) * x + a
            return eta * x * (
                math.sin(affine) * math.cos(affine) - math.sin(b) * math.cos(b))

        self.__left = self.__top = lambda x: x + deform_trans(x, 0., .5)
        self.__right = self.__bottom = lambda x: x + deform_trans(x, .5, 1.)
        self.__expand = lambda x: x + deform_trans(x, 0., 1.)
        self.__shrink = lambda x: x - deform_trans(x, 0., 1.)

    def transform(self, image, flag):

        horizon, vertical = flag & 0x7, flag & 0x38
        horizon_dt, vertical_dt = None, None

        if horizon == Deform.DT_LEFT:
            horizon_dt = self.__left
        elif horizon == Deform.DT_RIGHT:
            horizon_dt = self.__right
        elif horizon == Deform.DT_HORIZON_EXPAND:
            horizon_dt = self.__expand
        elif horizon == Deform.DT_HORIZON_SHRINK:
            horizon_dt = self.__shrink

        if vertical == Deform.DT_TOP:
            vertical_dt = self.__top
        elif vertical == Deform.DT_BOTTOM:
            vertical_dt = self.__bottom
        elif vertical == Deform.DT_VERTICAL_EXPAND:
            vertical_dt = self.__expand
        elif vertical == Deform.DT_VERTICAL_SHRINK:
            vertical_dt = self.__shrink

        def prepare(i, size, dt_func):
            i = (dt_func(i / size) * size) if dt_func else i
            return int(round(np.minimum(np.maximum(i, 0), size - 1)))

        rows, cols = image.shape
        y_dt = [prepare(y, rows, vertical_dt) for y in range(0, rows)]
        x_dt = [prepare(x, cols, horizon_dt) for x in range(0, cols)]
        image_dt = np.zeros(image.shape)
        for y in range(0, rows):
            for x in range(0, cols):
                image_dt[y, x] = image[y_dt[y], x_dt[x]]

        return image_dt


if __name__ == '__main__':

    def example_img():
        offset = 5
        example = np.ones((100, 100))
        example[:, :] = 255
        example[offset:100:offset] = 0
        example[:, offset:100:offset] = 0
        return example

    _, img = next(iter(CasiaFile('1001-c.gnt')))


    def update_distort(eta):
        pass

    def update_deform(eta):
        trans = Deform(eta)
        plt.subplot(3, 4, 1)
        plt.title('LEFT')
        plt.imshow(trans.transform(img, Deform.DT_LEFT), cmap='gray')
        plt.subplot(3, 4, 2)
        plt.title('RIGHT')
        plt.imshow(trans.transform(img, Deform.DT_RIGHT), cmap='gray')
        plt.subplot(3, 4, 3)
        plt.title('H_EXPAND')
        plt.imshow(trans.transform(img, Deform.DT_HORIZON_EXPAND), cmap='gray')
        plt.subplot(3, 4, 4)
        plt.title('H_SHRINK')
        plt.imshow(trans.transform(img, Deform.DT_HORIZON_SHRINK), cmap='gray')

        plt.subplot(3, 4, 5)
        plt.title('TOP')
        plt.imshow(trans.transform(img, Deform.DT_TOP), cmap='gray')
        plt.subplot(3, 4, 6)
        plt.title('BOTTOM')
        plt.imshow(trans.transform(img, Deform.DT_BOTTOM), cmap='gray')
        plt.subplot(3, 4, 7)
        plt.title('V_EXPAND')
        plt.imshow(trans.transform(img, Deform.DT_VERTICAL_EXPAND), cmap='gray')
        plt.subplot(3, 4, 8)
        plt.title('V_SHRINK')
        plt.imshow(trans.transform(img, Deform.DT_VERTICAL_SHRINK), cmap='gray')

    update = update_deform

    update(1.)
    slid = Slider(plt.subplot(3, 1, 3), 'eta', 0.001, 5.0, valinit=1)
    slid.on_changed(update)
    plt.show()




