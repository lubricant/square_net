from collections import namedtuple
import os.path as path
import struct

import numpy as np
from tensorflow.python import gfile as gf

from data import *

LabelWithLoc = namedtuple('LabelWithLoc', 'label, top, left, height, width')


class File(object):

    def __init__(self, filename):
        assert filename
        self._filename = filename

    name = property(lambda self: self._filename, None, None)

    @staticmethod
    def list_file(full_path=True):
        raise NotImplementedError


class MnistFile(File):

    '''
    MNIST 手写数字数据集
        包含 0-9 的数字图片，每张图片大小 28x28
        训练集：每个数字约 6000 张
        测试集：每个数字约 1000 张
    '''

    @staticmethod
    def list_file(full_path=True, get_train_set=True, get_test_set=False):
        train_filename = PWD + '/mnist/train-images.idx3-ubyte', PWD + '/mnist/train-labels.idx1-ubyte'
        test_filename = PWD + '/mnist/t10k-images.idx3-ubyte', PWD + '/mnist/t10k-labels.idx1-ubyte'

        file_list = []
        if get_train_set:
            assert path.exists(train_filename[0]) and path.exists(train_filename[1])
            file_list.append(train_filename if full_path else path.basename(train_filename))
        if get_test_set:
            assert path.exists(test_filename[0]) and path.exists(test_filename[1])
            file_list.append(test_filename if full_path else path.basename(test_filename))

        return file_list

    def __init__(self, image_filename, label_filename, reverse_pixel=False):
        super().__init__((path.basename(image_filename),
                          path.basename(label_filename)))
        assert path.exists(image_filename) and path.exists(label_filename)
        self.__img_path = image_filename
        self.__lab_path = label_filename
        self.__reverse = reverse_pixel

    def __iter__(self):

        with open(self.__img_path, 'rb') as img_file, open(self.__lab_path, 'rb') as lab_file:

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

    @staticmethod
    def list_file(full_path=True, use_db_v10=False, use_db_v11=False, get_train_set=False, get_test_set=False):

        db_v11 = ['1%03d-c.gnt' % i for i in range(1, 301)]

        file_list = []
        if get_train_set:
            if use_db_v11:
                v11_train_files = [PWD + '/casia/' + f for f in db_v11[:240]]
                assert all(path.exists(p) for p in v11_train_files if path.isfile(p))
                file_list += v11_train_files
            if use_db_v10:
                assert path.exists(PWD + '/casia/1.0train-gb1.gnt')
                file_list.append(PWD + '/casia/1.0train-gb1.gnt')

        if get_test_set:
            if use_db_v11:
                v11_test_files = [PWD + '/casia/' + f for f in db_v11[240:300]]
                assert all(path.exists(p) for p in v11_test_files if path.isfile(p))
                file_list += v11_test_files
            if use_db_v10:
                assert path.exists(PWD + '/casia/1.0test-gb1.gnt')
                file_list.append(PWD + '/casia/1.0test-gb1.gnt')

        return file_list if full_path else [path.basename(f) for f in file_list]

    def __init__(self, filename, reverse_pixel=True):
        super().__init__(path.basename(filename))
        assert path.exists(filename)
        self.__path = filename
        self.__reverse = reverse_pixel

    def __iter__(self):

        assert not path.isdir(self.__path)
        size = path.getsize(self.__path)

        with open(self.__path, 'rb') as file:
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


class HITFile(File):
    '''
    HIT-ORC 手写中文数据集
        采用 GBK 编码，总共包含 6825 个常用字符与汉字，每类 122 个样本，每张图片大小为 128x128
    '''

    @staticmethod
    def list_file(full_path=True):

        images_files = ['%d_images' % i for i in range(1, 123)]
        images_paths = [PWD + '/hit/' + f for f in images_files]
        assert all(path.exists(p) for p in images_paths if path.isfile(p))

        labels_file = ['labels'] * 122
        labels_path = [PWD + '/hit/labels'] * 122
        assert path.exists(labels_path[0])

        return [*zip(images_paths, labels_path)] if full_path else [*zip(images_files, labels_file)]

    def __init__(self, image_filename, label_filename, reverse_pixel=True):
        super().__init__((path.basename(image_filename),
                          path.basename(label_filename)))
        assert path.exists(image_filename) and path.exists(label_filename)
        self.__img_path = image_filename
        self.__lab_path = label_filename
        self.__reverse = reverse_pixel

    def __iter__(self):

        with open(self.__img_path, 'rb') as img_file, open(self.__lab_path, 'rb') as lab_file:

            img_num, row, col = struct.unpack('<IBB', img_file.read(6))
            lab_num, ch_bytes = struct.unpack('<HB', lab_file.read(3))
            assert img_num == lab_num

            labels = lab_file.read(lab_num * ch_bytes).decode('GBK')
            assert len(labels) == lab_num

            for ch in labels:
                image = np.fromfile(img_file, np.uint8, row * col).reshape((row, col))
                if self.__reverse:
                    np.subtract(255, image, image)
                yield ch, image


class HitDocFile(File):

    @staticmethod
    def list_file(full_path=True):

        assert full_path

        dir_list = [PWD + '/hit_doc/%d' % (i+1) for i in range(1)]
        image_paths = [('%s/1_images' % d, '%s/2_images' % d) for d in dir_list]
        label_paths = ['%s/labels' % d for d in dir_list]

        assert all(path.exists(i1) and path.exists(i2) for i1, i2 in image_paths)
        assert all(path.exists(i) for i in label_paths)
        return [*zip(image_paths, label_paths)]

    def __init__(self, image_filename, label_filename, reverse_pixel=True):
        image_filename_1, image_filename_2 = image_filename
        super().__init__((path.basename(image_filename_1),
                          path.basename(image_filename_2),
                          path.basename(label_filename)))

        assert path.exists(image_filename_1) and \
               path.exists(image_filename_2) and \
               path.exists(label_filename)

        self.__img_path = image_filename_1, image_filename_2
        self.__lab_path = label_filename
        self.__reverse = reverse_pixel

    def __iter__(self):
        img_path_1 , img_path_2 = self.__img_path
        with open(self.__lab_path, 'rb') as lab_file, \
                open(img_path_1, 'rb') as img_file_1, \
                open(img_path_2, 'rb') as img_file_2:

            lab_num, ch_bytes = struct.unpack('<HB', lab_file.read(3))
            img_num_1, row_1, col_1 = struct.unpack('<IBB', img_file_1.read(6))
            img_num_2, row_2, col_2 = struct.unpack('<IBB', img_file_2.read(6))

            labels = lab_file.read(lab_num * ch_bytes).decode('GBK')
            assert len(labels) == lab_num

            for ch in labels:
                image = np.fromfile(img_file_1, np.uint8, row_1 * col_1).reshape((row_1, col_1))
                if self.__reverse:
                    np.subtract(255, image, image)
                yield ch, image

            for ch in labels:
                image = np.fromfile(img_file_2, np.uint8, row_2 * col_2).reshape((row_2, col_2))
                if self.__reverse:
                    np.subtract(255, image, image)
                yield ch, image


class TFRecordFile(File):

    '''
    TFRecord 转换器
        保存大小对齐后的图片以及汉字对应的索引
    '''

    @staticmethod
    def list_file(full_path=True, get_train_set=False, get_test_set=False):

        file_list = []
        if get_train_set:
            train_files = [p for p in gf.ListDirectory(RECORD_ROOT + '/record/train/') if not gf.IsDirectory(p)]
            file_list += train_files
        if get_test_set:
            test_files = [p for p in gf.ListDirectory(RECORD_ROOT + '/record/test/') if not gf.IsDirectory(p)]
            file_list += test_files

        return file_list if full_path else [path.basename(f) for f in file_list]

    def __init__(self, filename, dict_map=None):
        super().__init__(filename)
        assert gf.Exists(RECORD_ROOT + '/record/')
        self.__dict_map = dict_map
        self.__file_path = RECORD_ROOT + '/record/' + filename
        self.__tf_writer = None

    def __iter__(self):
        assert gf.Exists(self.__file_path)
        for serialized_example in tf.python_io.tf_record_iterator(self.__file_path):
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

        if not self.__tf_writer:
            assert gf.Exists(RECORD_ROOT + '/record/')
            self.__tf_writer = tf.python_io.TFRecordWriter(self.__file_path)

        writer = self.__tf_writer
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ch.encode()])),
            'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[dict_map[ch]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
            # 'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[*img.shape]))
        }))

        writer.write(example.SerializeToString())

    def close(self):
        if self.__tf_writer:
            self.__tf_writer.close()
            self.__tf_writer = None


class CasiaDocFile(File):

    @staticmethod
    def list_file(full_path=True):
        doc_paths = glob.glob(PWD + '/casia_doc/*.dgr')
        doc_files = [path.basename(p) for p in doc_paths]
        return doc_paths if full_path else doc_files

    def __init__(self, filename, reverse_pixel=True, full_doc=False):
        super().__init__(path.basename(filename))
        assert path.exists(filename)
        self.__path = filename
        self.__reverse = reverse_pixel
        self.__full = full_doc

    def __iter__(self):
        assert not path.isdir(self.__path)

        with open(self.__path, 'rb') as file:

            head_size, = struct.unpack('<I', file.read(4))
            fmt_code = struct.unpack('<8B', file.read(8))
            illus_txt = struct.unpack('<%dB' % (head_size-36), file.read(head_size-36))
            code_type = file.read(20).decode('ASCII').replace('\0', '')
            code_len, pix_bits = struct.unpack('<2H', file.read(4))
            img_row, img_col, line_num = struct.unpack('<3I', file.read(12))

            code_type = 'GBK' if code_type == 'GB' else code_type

            assert pix_bits == 8
            assert code_type in ('ASCII', 'GBK')

            if self.__full:
                doc_img = np.zeros((img_row, img_col), np.uint8) + 255
                doc_lab = []

                for i in range(line_num):
                    word_num, = struct.unpack('<I', file.read(4))
                    for j in range(word_num):
                        label = file.read(code_len).decode(code_type)
                        top, left, height, width = struct.unpack('<4H', file.read(8))
                        doc_lab.append(LabelWithLoc(label, top, left, height, width))
                        word = np.fromfile(file, np.uint8, height * width).reshape((height, width))
                        doc_img[top:top + height, left:left + width] = word
                yield doc_lab, doc_img
            else:
                pass
                # for i in range(line_num):
                #     word_num, = struct.unpack('<I', file.read(4))
                #     img_line, lab_line = [], []
                #
                #     for j in range(word_num):
                #         label = file.read(code_len).decode(code_type)
                #         top, left, height, width = struct.unpack('<4H', file.read(8))
                #         lab_line.append(LabelWithLoc(label, top, left, height, width))
                #         word = np.fromfile(file, np.uint8, height * width).reshape((height, width))
                #         img_line.append(word)
                #
                #     yield doc_lab, doc_img


if __name__ == '__main__':

    import cv2
    import matplotlib.pyplot as plt

    def try_mist():
        cnt = 0
        ch_map = {}
        test_files = MnistFile.list_file(get_train_set=False, get_test_set=True)[0]
        for ch, img in MnistFile(*test_files):
            cnt += 1
            if ch not in ch_map:
                ch_map[ch] = 0
            ch_map[ch] += 1
            print(ch, img.shape, np.max(img), np.min(img))
            # plt.imshow(img, cmap='gray')
            # plt.show()
        print(cnt)
        print(str(ch_map))

    def try_casi():

        cnt = 0
        ch_map = {}
        for name in CasiaFile.list_file(use_db_v10=True, use_db_v11=True, get_train_set=True, get_test_set=False):
            for ch, img in CasiaFile(name):
                cnt += 1
                if ch not in ch_map:
                    ch_map[ch] = 0
                ch_map[ch] += 1
                # print(ch, img.shape)
                # plt.subplot(2, 1, 1)
                # plt.imshow(img, cmap='gray')
                # plt.subplot(2, 1, 2)
                # img = np.subtract(255, img)
                # plt.imshow(img, cmap='gray')
                # plt.show()
        print(cnt)
        print(str(ch_map))
        print(len(ch_map))
        print(sorted(ch_map.keys()))
        print(CasiaFile.list_file(full_path=False, use_db_v10=True, use_db_v11=True, get_train_set=False,
                                  get_test_set=True))

    def try_hit():
        cnt = 0
        ch_map = {}
        for name in HITFile.list_file():
            for ch, img in HITFile(*name):
                cnt += 1
                if ch not in ch_map:
                    ch_map[ch] = 0
                ch_map[ch] += 1
                print(ch, img.shape)
                # plt.imshow(img, cmap='gray')
                # plt.show()
        print(cnt)
        print(str(ch_map))
        print(len(ch_map))
        print(sorted(ch_map.keys()))
        print(HITFile.list_file())

    # try_mist()
    # try_casi()
    # try_hit()


    def try_casia_doc():
        for f in CasiaDocFile.list_file():
            for label, image in CasiaDocFile(f, full_doc=True):
                img_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                for ch, top, left, height, width in label:
                    cv2.rectangle(img_copy, (left, top), (left + width, top + height), (255, 0, 0), thickness=5)
                plt.imshow(img_copy)
                plt.show()

    try_casia_doc()
