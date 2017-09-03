
import data
import data.fmt_file as fmt
import data.cv_filter as flt

import model.network as net

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

indices, chinese = data.label_dict()

network = net.HCCR_GoogLeNet(is_training=False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
network.set_model_params(sess, data.model_param())


def pre_process(gray_img, reverse=False):
    gray_img = flt.AlignFilter((data.IMG_SIZE, data.IMG_SIZE)).filter(gray_img)
    if reverse:
        np.subtract(255, gray_img, gray_img)
    gray_img = np.pad(gray_img, ((4, 4), (4, 4)), 'constant')
    gray_img = gray_img.reshape(gray_img.shape + (1,)).astype(np.float32)
    gray_img -= np.mean(gray_img)
    gray_img /= np.std(gray_img)
    return gray_img


def guess_ch(img_list):
    img_list = [pre_process(i) for i in img_list]
    images = np.stack(img_list)
    logits = sess.run(network.logits, {network.images: images, network.keep_prob: 1.})
    labels = np.argmax(logits, axis=-1)
    return [indices[i] for i in labels]

total, wrong = 0, 0
for f in fmt.HITFile.list_file():
    for ch, img in fmt.HITFile(*f):
        if ch in chinese:
            lab = guess_ch([img])[0]
            total += 1
            wrong += 0 if lab == ch else 1
            if not total % 1000:
                print((total-wrong)/total)

