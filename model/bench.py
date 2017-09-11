
import data
import data.fmt_file as fmt
import data.cv_filter as flt

import model.network as net

import cv2
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from skimage import measure, morphology

from collections import namedtuple

Rect = namedtuple('Rect', 'top, left, bottom, right, height, width')
Region = namedtuple('Region', 'top, left, bottom, right, ratio, label')


def merge_region(labels):
    top, left = labels.shape
    bottom, right = 0, 0

    found = False
    regions = []
    for props in measure.regionprops(labels):
        if props.eccentricity < .95:
            t, l, b, r = props.bbox
            top, left = min(t, top), min(l, left)
            right, bottom = max(r, right), max(b, bottom)
            regions.append(Region(t, l, b, r, abs((r-l) / float(b-t) - 1.), (props.label,)))
            found = True

    merged, prev = [], None
    for cur in sorted(regions, key=lambda it: it.left):
        if not prev:
            prev = cur
        elif cur.left <= prev.right:
            t, l, b, r = min(prev.top, cur.top), prev.left, max(prev.bottom, cur.bottom), max(prev.right, cur.right)
            prev = Region(t, l, b, r, abs((r-l) / float(b-t) - 1.), prev.label + cur.label)
        else:
            merged.append(prev)
            prev = cur
    merged.append(prev)

    final = [Rect(top, left, bottom, right, bottom-top, right-left)]
    i, prev = 1, None
    for cur in merged:
        if not prev:
            prev = cur
        else:
            t, l, b, r = min(prev.top, cur.top), prev.left, max(prev.bottom, cur.bottom), max(prev.right, cur.right)
            merged_ratio = abs((r-l) / float(b-t) - 1.)
            if merged_ratio < prev.ratio and merged_ratio < cur.ratio:
                prev = Region(t, l, b, r, merged_ratio, prev.label + cur.label)
            else:
                t, l, b, r = prev.top-top, prev.left-left, prev.bottom-top, prev.right-left
                final.append(Rect(t, l, b, r, b-t, r-l))
                labels[np.isin(labels, prev.label)] = i
                i, prev = i+1, cur

    if prev is not None:
        t, l, b, r = prev.top - top, prev.left - left, prev.bottom - top, prev.right - left
        final.append(Rect(t, l, b, r, b - t, r - l))
        labels[np.isin(labels, prev.label)] = i

    return final if found else None


def reduce_region(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    equalized = cv2.equalizeHist(blurred)
    edged = cv2.Canny(equalized, 50, 150)

    labels = measure.label(edged, connectivity=2)
    morphology.remove_small_objects(labels, min_size=32, connectivity=2, in_place=True)

    boxes = merge_region(labels)

    if boxes:
        top, left, bottom, right, _, _ = boxes[0]
        return gray[top:bottom, left:right], boxes


if __name__ == '__main__':

    # load dict
    indices, chinese = data.label_dict()

    # load network
    network = net.HCCR_GoogLeNet(is_training=False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    network.set_model_params(sess, data.model_param())

    # standardize image
    def pre_process(gray_img):
        gray_img = flt.AlignFilter((data.IMG_SIZE, data.IMG_SIZE), 'maximum').filter(gray_img)
        gray_img = flt.BinaryFilter(inverse=True, otsu=True).filter(gray_img)
        gray_img = np.pad(gray_img, ((4, 4), (4, 4)), 'constant')
        gray_img = gray_img.reshape(gray_img.shape + (1,)).astype(np.float32)
        gray_img -= np.mean(gray_img)
        gray_img /= np.std(gray_img)
        return gray_img

    # batch predict image
    def guess_ch(img_list):
        img_list = [pre_process(i) for i in img_list]

        # for i, w in enumerate(img_list):
        #     plt.subplot(1, len(img_list), i + 1)
        #     plt.imshow(w.reshape(w.shape[:-1]), cmap='gray')
        # plt.show()

        images = np.stack(img_list)
        logits = sess.run(network.logits, {network.images: images, network.keep_prob: 1.})
        labels = np.argmax(logits, axis=-1)
        return [indices[i] for i in labels]

    # total, wrong = 0, 0
    # for f in fmt.HITFile.list_file():
    #     for ch, img in fmt.HITFile(*f):
    #         if ch in chinese:
    #             lab = guess_ch([img])[0]
    #             total += 1
    #             wrong += 0 if lab == ch else 1
    #             if not total % 1000:
    #                 print((total - wrong) / total)

    # init camera
    camera = cv2.VideoCapture(0)

    # loop for frame
    while True:

        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        result = reduce_region(frame)
        if result is not None:
            gray, boxes = result
            frame = gray
            line_box, word_boxes = boxes[0], boxes[1:]
            if word_boxes:
                # rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # rect = [np.array([[l, t], [r, t], [r, b], [l, b]]) for t, l, b, r, _, _ in word_boxes]
                # cv2.drawContours(rgb, rect, -1, (0, 0, 255), 4)
                # frame = rgb
                # [gray[wb.top:wb.bottom+1, wb.left:wb.right+1] for wb in word_boxes]
                words = [gray[:, l:r] for t, l, b, r, _, _ in word_boxes]
                print(guess_ch(words))

        cv2.imshow('X', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("Q"):
            break

    # release camera
    camera.release()
    cv2.destroyAllWindows()