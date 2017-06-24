import inspect
import os
import shutil
import struct

import numpy as np
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

    def __init__(self, filename):
        assert filename
        self.__filepath = 'record/' + filename
        self.__tfwriter = None

    def __del__(self):
        self.close()
        if exist_path(self.__filepath):
            os.remove(get_path(self.__filepath))

    def write(self, ch, img):
        if not self.__tfwriter:
            self.__tfwriter = tf.python_io.TFRecordWriter(get_path(self.__filepath))

        writer = self.__tfwriter
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[ch])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
        }))

        writer.write(example.SerializeToString())

    def close(self):
        if self.__tfwriter:
            self.__tfwriter.close()
            self.__tfwriter = None


if __name__ == '__main__':
    ch_cnt = {}
    ch_idx = []
    for name in list_file('casia'):
        for ch, _ in CasiaFile(name):
            if ch not in ch_cnt:
                ch_cnt[ch] = 1
                ch_idx.append(ch)
            else:
                ch_cnt[ch] += 1

    print(len(ch_cnt.keys()))
    print(ch_cnt)




