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


class MnistFile(object):

    '''
    MNIST 手写数字数据集
    '''

    def __init__(self, image_filename, label_filename):
        assert exist_path('mnist/' + image_filename) and \
               exist_path('mnist/' + label_filename)
        self.__img_filepath = 'mnist/' + image_filename
        self.__lab_filepath = 'mnist/' + label_filename

    def __iter__(self):
        img_file = open(get_path(self.__img_filepath), 'rb')
        lab_file = open(get_path(self.__lab_filepath), 'rb')
        img_buf, lab_buf = img_file.read(), lab_file.read()

        def read_img():
            head = struct.unpack_from('>4I', img_buf, 0)
            magic_num, img_num, img_row, img_col = head

            offset = struct.calcsize('>4I')
            pixels = img_num * img_row * img_col

            images = struct.unpack_from('>%dB' % pixels, img_buf, offset)
            assert len(images) / (img_row * img_col) == img_num

            return np.array(images, dtype=np.uint8).reshape(img_num, img_row, img_col)

        def read_lab():
            head = struct.unpack_from('>2I', lab_buf, 0)
            magic_num, lab_num = head

            offset = struct.calcsize('>2I')
            labels = struct.unpack_from('>%dB' % lab_num, lab_buf, offset)

            return np.array(labels, dtype=np.uint8)

        for ch, img in zip(read_lab(), read_img()):
            yield ch, img


class CasiaFile(object):

    '''
    Casia 手写中文数据集
        采用 GBK 编码
        总共包含 3755 个汉字
        每个汉字 240 个样本
    '''

    def __init__(self, filename):
        assert exist_path('casia/' + filename)
        self.__filepath = 'casia/' + filename

    def __iter__(self):

        size = get_size(self.__filepath)
        file = open(get_path(self.__filepath), 'rb')

        while file.tell() < size:
            length, = struct.unpack('<I', file.read(4))
            code = file.read(2)
            col, row = struct.unpack('<2H', file.read(4))
            data_len = length - 10
            data_tag = code.decode('gbk')
            data_img = np.fromstring(file.read(data_len), np.uint8).reshape((row, col))
            yield (data_tag, data_img)


class TFRecordFile(object):

    '''
    TFRecord 转换器
        保存大小对齐后的图片以及汉字对应的索引
    '''

    def __init__(self, filename, dict_map=None):
        assert filename
        self.__dict_map = dict_map
        self.__filepath = 'tfrecord/' + filename
        self.__tfwriter = None

    def __iter__(self):
        assert exist_path(self.__filepath)
        for serialized_example in tf.python_io.tf_record_iterator(get_path(self.__filepath)):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)

            rows, cols = example.features.feature['shape'].int64_list.value
            image, = example.features.feature['image'].bytes_list.value
            label, = example.features.feature['label'].bytes_list.value
            yield label.decode(), np.fromstring(image, np.uint8).reshape((rows, cols))

    def write(self, ch, img):
        assert isinstance(ch, str) and \
               isinstance(img, np.ndarray)

        dict_map = self.__dict_map
        assert dict_map and (ch in dict_map) and (
            isinstance(dict_map[ch], int))

        if not self.__tfwriter:
            if not exist_path('tfrecord'):
                ensure_path('tfrecord')
            self.__tfwriter = tf.python_io.TFRecordWriter(get_path(self.__filepath))

        writer = self.__tfwriter
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ch.encode()])),
            'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[dict_map[ch]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[*img.shape]))
        }))

        writer.write(example.SerializeToString())

    def close(self):
        if self.__tfwriter:
            self.__tfwriter.close()
            self.__tfwriter = None

    @staticmethod
    def ch_dict(refresh=False):

        def generate_dict():
            ch_mapping, idx_mapping, ch_idx = {}, {}, 0
            for name in list_file('casia'):
                for ch, _ in CasiaFile(name):
                    if ch not in ch_map:
                        ch_mapping[ch_idx] = ch
                        idx_mapping[ch] = ch_idx
                        ch_idx += 1
            return [ch_mapping, idx_mapping]

        dict_file = 'tmp/ch_dict.npy'
        if refresh or not exist_path(dict_file):
            np.save(get_path(dict_file), generate_dict())

        ch_map, idx_map = np.load(dict_file)
        # g_ch_map, g_idx_map = generate_dict()
        # assert not any(True for k in ch_map if (
        #     k not in g_ch_map or ch_map[k] != g_ch_map[k]))
        # assert not any(True for k in idx_map if (
        #     k not in g_idx_map or idx_map[k] != g_idx_map[k]))
        return ch_map, idx_map

    @staticmethod
    def tf_queue():
        filename_queue = tf.train.string_input_producer([
            get_path('tfrecord/' + f) for f in list_file('tfrecord')])

        _, serialized_example = tf.TFRecordReader().read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'index': tf.FixedLenFeature([], tf.int64),
                'shape': tf.FixedLenFeature([2], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)})

        idx = tf.cast(features['index'], tf.int32)
        img = tf.reshape(tf.decode_raw(features['image'], tf.uint8),
                         tf.cast(features['shape'], tf.int32))

        return img, idx


if __name__ == '__main__':

    ch_dict, _ = TFRecordFile.ch_dict()

    image, label = TFRecordFile.tf_queue()
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




