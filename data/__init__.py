import inspect
import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow.python import gfile as gf

import data as this

PWD = os.path.abspath(os.path.join(inspect.getfile(this), os.pardir)).replace('\\', '/')
RECORD_ROOT = 'G:'
TEMP_ROOT = 'G:'

NUM_CLASSES = 3755
IMG_SIZE, IMG_CHANNEL = 112, 1
(DS_TRAIN, DS_TEST, DS_ALL) = ('TRAIN', 'TEST', 'ALL')
(TRAIN_SET_PREFIX, TEST_SET_PREFIX) = ('training_set', 'test_set')


def label_dict(dict_name='labels_dict.npy'):
    assert os.path.exists(PWD + '/blob/' + dict_name)
    return np.load(PWD + '/blob/' + dict_name)


def model_param(model_file='model_params.npy'):
    assert os.path.exists(PWD + '/blob/' + model_file)
    return np.load(PWD + '/blob/' + model_file)[0]


def data_queue(data_set, batch_size, thread_num=1, epoch_num=None):

    assert batch_size > 0 and thread_num > 0
    data_repo = {
        DS_TRAIN: RECORD_ROOT + '/record/**/' + TRAIN_SET_PREFIX + '*',
        DS_TEST: RECORD_ROOT + '/record/**/' + TEST_SET_PREFIX + '*'}

    assert data_set and data_set.upper() in (DS_TRAIN, DS_TEST, DS_ALL)

    with tf.name_scope('Queue'):
        if DS_ALL == data_set.upper():
            data_set_file = glob.glob(data_repo[DS_TRAIN]) + \
                            glob.glob(data_repo[DS_TEST])
        else:
            data_set_file = glob.glob(data_repo[data_set.upper()])

        filename_queue = tf.train.string_input_producer(
            list(filter(lambda f: os.path.isfile(f), set(data_set_file))),
            num_epochs=epoch_num)

        _, serialized_example = tf.TFRecordReader().read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'index': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)})

        pad = (120 - IMG_SIZE) // 2
        assert pad > 0 and (pad*2) + IMG_SIZE == 120
        padding = [[pad, pad], [pad, pad], [0, 0]]

        labels = tf.cast(features['index'], tf.int32)
        images = tf.cast(tf.pad(tf.reshape(
                    tf.decode_raw(features['image'], tf.uint8),
                    [IMG_SIZE, IMG_SIZE, IMG_CHANNEL]), padding), tf.float32)

        rand_data_queue = tf.train.shuffle_batch([images, labels],
                                                 batch_size=batch_size,
                                                 capacity=batch_size * 5,
                                                 min_after_dequeue=batch_size,
                                                 num_threads=thread_num)
        return rand_data_queue


if __name__ == '__main__':

    from PIL import Image
    from matplotlib import pyplot as plt
    from data.cv_filter import *
    from data.dt_trans import *

    np.set_printoptions(threshold=np.NaN, linewidth=np.NaN)

    ch_dict, _ = label_dict()

    def try_queue():

        filename_queue = tf.train.string_input_producer([RECORD_ROOT + '/record/train/training_set_0.tfr'])
        _, serialized_example = tf.TFRecordReader().read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'index': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.string)})

        label, index = features['label'], features['index']
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # 使用 eval 执行时，数据之间是乱序的
            for i in range(5000):
                print(label.eval().decode('UTF8'), ch_dict[index.eval()])

            coord.request_stop()
            coord.join(threads)

    def try_rand_queue():

        def save_image(images, labels, dir_path='D:/picture/'):
            for a, b in zip(images, labels):
                prefix, suffix = dir_path+ch_dict[b], ''
                while os.path.exists(prefix+suffix+'.png'):
                    suffix += '_'
                a = a.astype(np.uint8).reshape(a.shape[:-1])
                Image.fromarray(a).save(prefix+suffix+'.jpg')

        image_batch, label_batch = data_queue(data_set=DS_TRAIN, batch_size=100, epoch_num=1)
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())
        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(50):
                image, label = sess.run([image_batch, label_batch])
                # gabor_filter = GaborFilter((100, 100))
                # gabor_part = np.array()

                save_image(image, label)

                # img, idx = image[0], label[0]
                # mean, stddev = np.mean(img), np.std(img)
                # img -= mean
                # img /= stddev
                # print(ch_dict[idx], np.mean(img), np.std(img))

                # plt.imshow(img.reshape(img.shape[:-1]), cmap='gray')
                # plt.show()

            coord.request_stop()
            coord.join(threads)

    # try_queue()
    # try_rand_queue()

