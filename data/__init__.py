import inspect
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python import gfile as gf

import data as this

PWD = os.path.abspath(os.path.join(inspect.getfile(this), os.pardir)).replace('\\', '/')
RECORD_ROOT = 'F:'
TEMP_ROOT = PWD

NUM_CLASSES = 10 + 3755
IMG_SIZE, IMG_CHANNEL = 112, 1
(DS_TRAIN, DS_TEST, DS_ALL) = ('TRAIN', 'TEST', 'ALL')


def label_dict(dict_name='labels_dict.npy'):
    assert gf.Exists(PWD + '/blob/' + dict_name)
    return np.load(PWD + '/blob/' + dict_name)


def data_queue(data_set, batch_size, thread_num=1, epoch_num=None):

    assert batch_size > 0 and thread_num > 0

    data_repo = {
        DS_TRAIN: ['F:/record/train/'],
        DS_TEST: [],
        DS_ALL: []}

    assert data_set and data_set.upper() in data_repo

    with tf.name_scope('Queue'):
        data_set_file = []
        data_set_list = data_repo[data_set.upper()]

        assert data_set_list
        for data_set in data_set_list:
            assert gf.Exists(data_set) and \
                   gf.IsDirectory(data_set)
            data_set_file += [data_set + f for f in gf.ListDirectory(data_set)]

        filename_queue = tf.train.string_input_producer(
            list(filter(lambda f: os.path.isfile(f), set(data_set_file))),
            num_epochs=epoch_num)

        _, serialized_example = tf.TFRecordReader().read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'index': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)})

        labels = tf.cast(features['index'], tf.int32)
        images = tf.reshape((tf.cast(tf.decode_raw(features['image'], tf.uint8), tf.float32) / 255. - .5),
                            [IMG_SIZE, IMG_SIZE, IMG_CHANNEL])

        rand_data_queue = tf.train.shuffle_batch([images, labels],
                                                 batch_size=batch_size,
                                                 capacity=batch_size * 5,
                                                 min_after_dequeue=batch_size,
                                                 num_threads=thread_num)
        return rand_data_queue


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from data.cv_filter import *
    from data.dt_trans import *

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
        image_batch, label_batch = data_queue(data_set=DS_TRAIN, batch_size=100, epoch_num=1)
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())
        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(50):
                image, label = sess.run([image_batch, label_batch])
                print([(i, ch_dict[i]) for i in label])
                print(image.shape, type(image.shape[0]))
                # gabor_filter = GaborFilter((100, 100))
                # gabor_part = np.array()

                # img = image[0]
                # plt.imshow(img.reshape(img.shape[:-1]), cmap='gray')
                # plt.show()

            coord.request_stop()
            coord.join(threads)


    try_rand_queue()
