import inspect
import os
import shutil

import tensorflow as tf

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


def data_queue(exec_mode, batch_size, image_size, thread_num=1, epoch_num=None):

    assert batch_size > 0 and image_size > 0

    data_set = {
        'TRAINING': 'record/train',
        'TEST': 'record/test',
        'MIXING': 'record/all'}

    assert exec_mode and exec_mode.upper() in data_set

    data_set_dir = data_set[exec_mode.upper()]
    filename_queue = tf.train.string_input_producer([
        get_path(data_set_dir + '/' + f) for f in list_file(data_set_dir)], num_epochs=epoch_num)

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

    from data.cv_filter import *
    from data.dt_trans import *

    ch_dict, _ = label_dict()

    def test_queue():
        filename_queue = tf.train.string_input_producer([get_path('record/test/test_set_0.tfr')])
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

    def test_rand_queue():
        image_batch, label_batch = data_queue(exec_mode='test', batch_size=100, image_size=100, epoch_num=1)
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())
        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(50):
                image, label = sess.run([image_batch, label_batch])
                batch, each_shape = image.shape[0], image.shape[1:]
                print([(i, ch_dict[i]) for i in label])
                print(image.shape, type(image.shape[0]))
                # gabor_filter = GaborFilter((100, 100))
                # gabor_part = np.array()

            coord.request_stop()
            coord.join(threads)


    test_rand_queue()


