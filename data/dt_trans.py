import math

import numpy as np


class DT(object):

    (DT_LEFT, DT_RIGHT, DT_HORIZON_EXPAND, DT_HORIZON_SHRINK) = 0x5, 0x6, 0x7, 0x4
    (DT_TOP, DT_BOTTOM, DT_VERTICAL_EXPAND, DT_VERTICAL_SHRINK) = (
        DT_LEFT << 3, DT_RIGHT << 3, DT_HORIZON_EXPAND << 3, DT_HORIZON_SHRINK << 3)

    _left, _right = None, None
    _top, _bottom = None, None
    _shrink, _expand = None, None

    def transform(self, image, flag, out=None):

        horizon, vertical = flag & 0x7, flag & 0x38
        horizon_dt, vertical_dt = None, None

        if horizon == DT.DT_LEFT:
            horizon_dt = self._left
        elif horizon == DT.DT_RIGHT:
            horizon_dt = self._right
        elif horizon == DT.DT_HORIZON_EXPAND:
            horizon_dt = self._expand
        elif horizon == DT.DT_HORIZON_SHRINK:
            horizon_dt = self._shrink

        if vertical == DT.DT_TOP:
            vertical_dt = self._top
        elif vertical == DT.DT_BOTTOM:
            vertical_dt = self._bottom
        elif vertical == DT.DT_VERTICAL_EXPAND:
            vertical_dt = self._expand
        elif vertical == DT.DT_VERTICAL_SHRINK:
            vertical_dt = self._shrink

        def prepare(i, size, dt_func):
            i = (dt_func(i / size) * size) if dt_func else i
            return int(round(np.minimum(np.maximum(i, 0), size - 1)))

        rows, cols = image.shape
        y_dt = [prepare(y, rows, vertical_dt) for y in range(0, rows)]
        x_dt = [prepare(x, cols, horizon_dt) for x in range(0, cols)]

        if out is None:
            image_dt = np.zeros(image.shape, dtype=image.dtype)
        else:
            assert out.shape == image.shape
            assert out.dtype == image.dtype
            image_dt = out

        for y in range(0, rows):
            for x in range(0, cols):
                image_dt[y, x] = image[y_dt[y], x_dt[x]]

        return image_dt


class Distort(DT):

    def __init__(self, k=0., a=1.):

        def w1(a, t):
            assert 0 <= t <= 1
            return (1 - math.exp(-a * t)) / (1 - math.exp(-a))

        def w2(a, t):
            assert 0 <= t <= 1
            a = a if t <= .5 else -a
            b = 0 if t <= .5 else 1
            t = 2 * (t if t <= .5 else t - .5)
            return .5 * (b + (1-math.exp(-a*t))/(1-math.exp(-a)))  # w1

        self._left = self._top = lambda x: k * x + w1(a, x)
        self._right= self._bottom = lambda x: k * x + w1(-a, x)
        self._shrink = lambda x: k * x + w2(-a, x)
        self._expand = lambda x: k * x + w2(a, x)


class Deform(DT):

    def __init__(self, eta=1.):
        def deform(x, a, b):
            affine = (b - a) * x + a
            return eta * x * (
                math.sin(affine) * math.cos(affine) - math.sin(b) * math.cos(b))

        self._left = self._top = lambda x: x + deform(x, .5, 1.)
        self._right = self._bottom = lambda x: x + deform(x, 0., .5)
        self._shrink = lambda x: x + deform(x, 0., 1.)
        self._expand = lambda x: x - deform(x, 0., 1.)


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from matplotlib.widgets import Slider

    from data.fmt_file import CasiaFile

    def example_img():
        offset = 5
        example = np.ones((100, 100))
        example[:, :] = 255
        example[offset:100:offset] = 0
        example[:, offset:100:offset] = 0
        return example


    f = CasiaFile.list_file(use_db_v10=True, get_train_set=True)[0]
    _, img = next(iter(CasiaFile(f)))
    slid = None

    def update_distort(factor=1):
        trans = Distort(a=factor)  # Distort(k=factor)

        plt.subplot(3, 4, 1)
        plt.title('LEFT')
        plt.imshow(trans.transform(img, Distort.DT_LEFT), cmap='gray')
        plt.subplot(3, 4, 2)
        plt.title('RIGHT')
        plt.imshow(trans.transform(img, Distort.DT_RIGHT), cmap='gray')
        plt.subplot(3, 4, 3)
        plt.title('H_EXPAND')
        plt.imshow(trans.transform(img, Distort.DT_HORIZON_EXPAND), cmap='gray')
        plt.subplot(3, 4, 4)
        plt.title('H_SHRINK')
        plt.imshow(trans.transform(img, Distort.DT_HORIZON_SHRINK), cmap='gray')

        plt.subplot(3, 4, 5)
        plt.title('TOP')
        plt.imshow(trans.transform(img, Distort.DT_TOP), cmap='gray')
        plt.subplot(3, 4, 6)
        plt.title('BOTTOM')
        plt.imshow(trans.transform(img, Distort.DT_BOTTOM), cmap='gray')
        plt.subplot(3, 4, 7)
        plt.title('V_EXPAND')
        plt.imshow(trans.transform(img, Distort.DT_VERTICAL_EXPAND), cmap='gray')
        plt.subplot(3, 4, 8)
        plt.title('V_SHRINK')
        plt.imshow(trans.transform(img, Distort.DT_VERTICAL_SHRINK), cmap='gray')


    def update_deform(eta=1):
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


    if True:
        update = update_deform  # 1.85
        slid = Slider(plt.subplot(3, 1, 3), 'eta', 0.001, 5.0, valinit=1)
    else:
        update = update_distort  # a = 1.3, k = 0
        slid = Slider(plt.subplot(3, 1, 3), 'a', -2, 2, valinit=1)

    update()
    slid.on_changed(update)
    plt.show()




