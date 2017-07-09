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


def label_dict(dict_path='labels_dict.npy'):
    return np.load(get_path(dict_path))


def data_queue(exec_mode, epoch_num, batch_size, image_size, thread_num=1):

    assert epoch_num > 0 and batch_size > 0 and image_size > 0

    data_set = {
        'MIXING': 'mixing_set.tfr',
        'TRAINING': 'training_set.tfr',
        'TEST': 'test_set.tfr'}

    assert exec_mode and exec_mode.upper() in data_set

    # filename_queue = tf.train.string_input_producer([
    #     get_path(data_dir + '/' + f) for f in list_file(data_dir)], num_epochs=epoch_num)

    filename_queue = tf.train.string_input_producer(
        [data_set[exec_mode.upper()]], num_epochs=epoch_num)

    _, serialized_example = tf.TFRecordReader().read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'index': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)})

    labels = tf.cast(features['index'], tf.int32)
    images = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [image_size, image_size])
    rand_data_queue = tf.train.shuffle_batch([images, labels],
                                             batch_size=batch_size,
                                             capacity=batch_size * 5,
                                             min_after_dequeue=batch_size,
                                             num_threads=thread_num)
    return rand_data_queue


if __name__ == '__main__':

    ch_dict, _ = label_dict()

    image_batch, label_batch = data_queue(exec_mode='test', batch_size=100, epoch_num=10, image_size=100)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image, label = sess.run([image_batch, label_batch])
        print(image)
        print(label)
        # print('image label:' + ch_dict[])
        # plt.imshow(image.eval(), cmap='gray')
        # plt.show()


        coord.request_stop()
        coord.join(threads)




