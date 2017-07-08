import inspect
import os
import shutil
import struct

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import data as this

__data_path = os.path.abspath(os.path.join(inspect.getfile(this), os.pardir))


def get_path(path):
    path = path.replace('/', os.path.sep)
    if not path.startswith(os.path.sep):
        path = os.path.sep + path
    return __data_path + path


def get_size(path):
    filename = get_path(path)
    assert os.path.isfile(filename)
    return os.path.getsize(filename)


def list_file(path):
    filename = get_path(path)
    assert os.path.isdir(filename)
    return os.listdir(filename)


def exist_path(path):
    return os.path.exists(get_path(path))


def ensure_path(path):
    if exist_path(path):
        shutil.rmtree(get_path(path))
    os.mkdir(get_path(path))


def tf_queue(shuffle_queue=True):

    flags = tf.app.flags.FLAGS

    data_dir = flags.data_dir
    thread_num, epoch_num = flags.thread_num, flags.epoch_num
    batch_size, image_size = flags.batch_size, flags.image_size

    filename_queue = tf.train.string_input_producer([
        get_path(data_dir + '/' + f) for f in list_file(data_dir)], num_epochs=epoch_num)

    _, serialized_example = tf.TFRecordReader().read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'index': tf.FixedLenFeature([], tf.int64),
            'shape': tf.FixedLenFeature([2], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)})

    if not shuffle_queue:  # only for unit test !
        idx = tf.cast(features['index'], tf.int32)
        img = tf.reshape(tf.decode_raw(features['image'], tf.uint8),
                         tf.cast(features['shape'], tf.int32))
        return img, idx

    labels = tf.cast(features['index'], tf.int32)
    images = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [image_size, image_size])
    rand_data_queue = tf.train.shuffle_batch([images, labels],
                                             batch_size=batch_size,
                                             capacity=batch_size * 5,
                                             min_after_dequeue=batch_size,
                                             num_threads=thread_num)
    return rand_data_queue



if __name__ == '__main__':

    ch_dict, _ = ch_dict()

    image, label = tf_queue(shuffle_queue=False)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1):
            print('image label:' + ch_dict[label.eval()])
            plt.imshow(image.eval(), cmap='gray')
            plt.show()

        coord.request_stop()
        coord.join(threads)




