import inspect
import os
import shutil
import struct

import numpy as np

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


def exist_path(path):
    return os.path.exists(get_path(path))


def ensure_path(path):
    if exist_path(path):
        shutil.rmtree(get_path(path))
    os.mkdir(get_path(path))


class CasiaFile(object):

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
