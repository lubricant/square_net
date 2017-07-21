import struct

import numpy as np

from data import *


class File(object):

    def __init__(self, filename):
        assert filename
        self._filename = filename

    name = property(lambda self: self._filename, None, None)


class MnistFile(File):

    '''
    MNIST 手写数字数据集
        包含 0-9 的数字图片，每张图片大小 28x28
        训练集：每个数字约 6000 张
        测试集：每个数字约 1000 张
    '''

    def __init__(self, image_filename, label_filename, reverse_pixel=False):
        super().__init__((image_filename, label_filename))
        assert exist_path('mnist/' + image_filename) and \
               exist_path('mnist/' + label_filename)
        self.__img_filepath = 'mnist/' + image_filename
        self.__lab_filepath = 'mnist/' + label_filename
        self.__reverse = reverse_pixel

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
            if self.__reverse:
                np.subtract(255, img, img)
            yield str(ch), img


class CasiaFile(File):

    '''
    Casia 手写中文数据集
        采用 GBK 编码，总共包含 3755 个汉字，每张图片的边长在 50 ~ 150 之间不等
        训练集：每个汉字约 240 个样本
        训练集：每个汉字约 60 个样本
    '''

    def __init__(self, filename, reverse_pixel=True):
        super().__init__(filename)
        assert exist_path('casia/' + filename)
        self.__filepath = 'casia/' + filename
        self.__reverse = reverse_pixel

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
            if self.__reverse:
                np.subtract(255, data_img, data_img)
            yield (data_tag, data_img)


class TFRecordFile(File):

    '''
    TFRecord 转换器
        保存大小对齐后的图片以及汉字对应的索引
    '''

    def __init__(self, filename, dict_map=None):
        super().__init__(filename)

        self.__dict_map = dict_map
        self.__filepath = 'record/' + filename
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
            if not exist_path('record'):
                ensure_path('record')
            self.__tfwriter = tf.python_io.TFRecordWriter(get_path(self.__filepath))

        writer = self.__tfwriter
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ch.encode()])),
            'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[dict_map[ch]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
            # 'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[*img.shape]))
        }))

        writer.write(example.SerializeToString())

    def close(self):
        if self.__tfwriter:
            self.__tfwriter.close()
            self.__tfwriter = None


if __name__ == '__main__':

    def mnist_test():
        cnt = 0
        ch_map = {}
        # for ch, img in MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'):
        for ch, img in MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte'):
            cnt += 1
            if ch not in ch_map:
                ch_map[ch] = 0
            ch_map[ch] += 1
            print(ch, img.shape, np.max(img), np.min(img))
            # plt.imshow(img, cmap='gray')
            # plt.show()
        print(cnt)
        print(str(ch_map))

    def casi_test():
        cnt = 0
        ch_map = {}
        for name in list_file('casia/test'):
            for ch, img in CasiaFile('test/'+name):
                cnt += 1
                if ch not in ch_map:
                    ch_map[ch] = 0
                ch_map[ch] += 1
                print(ch, img.shape)
                # plt.imshow(img, cmap='gray')
                # plt.show()
        print(cnt)
        print(str(ch_map))

    # mnist_test()
    # casi_test()
