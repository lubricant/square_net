
import cv2

import numpy as np


class BinaryFilter(object):

    def __init__(self, inverse=False, otsu=False):
        self.__inverse = inverse
        self.__otsu_thresh = otsu

    def filter(self, image):
        if self.__otsu_thresh:
            return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | (
                cv2.THRESH_BINARY_INV if self.__inverse else cv2.THRESH_BINARY))[1]
        else:
            return cv2.threshold(image, np.mean(image), 255, (
                cv2.THRESH_BINARY_INV if self.__inverse else cv2.THRESH_BINARY))[1]


class AlignFilter(object):

    def __init__(self, size, mode='constant', **args):
        assert mode and len(size) == 2
        self.__size = size
        self.__mode = mode
        self.__args = args

    def filter(self, image):
        rows, cols = self.__size
        i_rows, i_cols = image.shape

        align_ratio, i_ratio = rows/cols, i_rows/i_cols
        if i_ratio != align_ratio:
            align_rows, align_cols = i_cols * align_ratio, i_rows / align_ratio
            assert align_rows > i_rows or align_cols > i_cols

            if align_rows > i_rows:
                pad_rows = int(round((align_rows - i_rows) / 2))
                padding = ((pad_rows, pad_rows), (0, 0))
            else:
                pad_cols = int(round((align_cols - i_cols) / 2))
                padding = ((0, 0), (pad_cols, pad_cols))

            image = np.pad(image, padding, self.__mode, **self.__args)

        return cv2.resize(image, (cols, rows))


class GaborFilter(object):

    def __init__(self, size):
        assert len(size) == 2

        aspect_ratio = size[0]/size[1]
        wavelength = 4 * np.sqrt(2)
        self.__kernels = [
            cv2.getGaborKernel(size, 1, orient, wavelength, aspect_ratio, ktype=cv2.CV_32F)
            for orient in np.arange(0, np.pi, np.pi / 8)]

    def filter(self, image):
        features = np.zeros((8,) + image.shape)
        for i in range(8):
            features[i] = cv2.filter2D(image, cv2.CV_8UC3, self.__kernels[i])
        return features

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from data.fmt_file import CasiaFile

    def gabor_feature(image):
        res = GaborFilter(image.shape).filter(image)
        plt.subplot(3, 1, 1)
        plt.imshow(image, cmap='gray')
        for i in range(len(res)):
            plt.subplot(3, 4, 5 + i)
            plt.title(i * 180./8.)
            plt.imshow(res[i], cmap='gray')
        plt.show()


    def binary(image):
        plt.subplot(3, 1, 1)
        plt.imshow(image, cmap='gray')
        plt.subplot(3, 2, 3)
        plt.imshow(BinaryFilter().filter(image), cmap='gray')
        plt.subplot(3, 2, 4)
        plt.imshow(BinaryFilter(inverse=True).filter(image), cmap='gray')
        plt.subplot(3, 2, 5)
        plt.imshow(BinaryFilter(otsu=True).filter(image), cmap='gray')
        plt.subplot(3, 2, 6)
        plt.imshow(BinaryFilter(otsu=True, inverse=True).filter(image), cmap='gray')
        plt.show()

    def resize(image):
        plt.subplot(2, 1, 1)
        plt.imshow(image, cmap='gray')
        plt.subplot(2, 3, 4)
        plt.imshow(AlignFilter((150, 150), constant_values=255).filter(image), cmap='gray')
        plt.subplot(2, 3, 5)
        plt.imshow(AlignFilter((40, 50), constant_values=255).filter(image), cmap='gray')
        plt.subplot(2, 3, 6)
        plt.imshow(AlignFilter((50, 150), constant_values=255).filter(image), cmap='gray')
        plt.show()


    for ch, img in CasiaFile('1001-c.gnt'):
        # gabor_feature(img)
        resize(img)
        break



