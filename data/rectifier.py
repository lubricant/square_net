import queue
import logging

from data.mult_io import *
from data.dt_trans import *
from data.fmt_file import *

'''
使用函数前，按照如下目录结构放置好文件：
data
|----casia
|   |----train：[CASIA 训练集文件 (.gnt)]
|   |----test： [CASIA 测试集文件 (.gnt)]
|----mnist：    [MNIST 所有文件]

RECORD_ROOT     [任意一个磁盘目录，在 __init__ 中定义]
|----record:
|   |----train：[TFRecord 格式训练集文件 (.tfr)]
|   |----test： [TFRecord 格式测试集文件 (.tfr)]
|   |----all：  [TFRecord 格式数据集文件 (.tfr)]

TEMP_ROOT       [任意一个磁盘目录，在 __init__ 中定义]
|----summary    [存放训练过程中生成的 Summary]
|----checkpoint [存放训练过程中生成的 Checkpoint]
'''


def prepare_label_dict(dict_path='tmp/labels_dict.npy'):
    '''
    构建 label 字典文件并以 npy 格式保存
    生成的字典文件暂时存放在 tmp 目录
    '''

    def build_label_mapping(casia_files):

        chinese = []
        for file in casia_files:
            for cn, _ in file:
                chinese.append(cn)

        chinese = sorted(set(chinese), key=lambda x: int(x.encode('GBK').hex(), base=16))
        classes = [str(digit) for digit in range(10)] + chinese

        id_2_ch, ch_2_id = {}, {}
        for id in range(len(classes)):
            ch = classes[id]
            id_2_ch[id] = ch
            ch_2_id[ch] = id

        return id_2_ch, ch_2_id

    logging.info('Start preparing label dict ......')

    ch_mapping, id_mapping = build_label_mapping(
        [CasiaFile('test/' + name) for name in gf.ListDirectory(PWD + '/casia/test')])

    np.save(PWD + '/' + dict_path, (ch_mapping, id_mapping))

    logging.info('Finish preparing label dict.')


def prepare_image_files(file_name, data_set, img_per_file=250000, dict_path=None, img_filter=lambda _: _):
    '''
    将原始图片文件转换为 TFRecord 格式的文件
    '''

    class TFWriter(object):

        def __init__(self, file_dir, img_queue):
            self.file_dir = file_dir
            self.img_queue = img_queue
            self.img_num, self.img_writer = 0, []
            _, self.forward_dict = np.load((PWD + '/blob/labels_dict.npy') if not dict_path else dict_path)

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            if len(self.img_writer):
                self.img_writer[-1].close()
                self.img_writer = None
            del self.forward_dict
            self.forward_dict = None
            logging.info('Total image : %d', self.img_num)

        def __call__(self, data_batch):
            img_writer = self.img_writer
            writer_id = self.img_num // img_per_file
            if writer_id >= len(img_writer):
                if writer_id:
                    img_writer[writer_id - 1].close()
                img_writer.append(TFRecordFile((self.file_dir + '/' + file_name) % writer_id, self.forward_dict))
                assert writer_id == len(img_writer) - 1

            for ch, im in data_batch:
                img_writer[writer_id].write(ch, img_filter(im))
            self.img_num += len(data_batch)
            # print(self.img_queue.qsize())

    class MNISTReader(object):

        def __init__(self, file_list, fetch_num=None):
            assert len(file_list)
            self.file_list = file_list
            self.fetch_num = fetch_num
            self.total_num = 0
            self.digits_cnt = {}
            for n in range(10):
                self.digits_cnt[str(n)] = 0

        def __enter__(self):
            if self.fetch_num is not None:
                logging.info('Expect MNIST image: {}[{}x{}]'.format(
                    self.fetch_num * 10, 10, self.fetch_num))

        def __exit__(self, exc_type, exc_val, exc_tb):
            if len(self.digits_cnt) and self.fetch_num is not None:
                logging.error('Not enough MNIST image: {}'.format(['%d:%d' % (
                    n, self.fetch_num - self.digits_cnt[n] if n in self.digits_cnt else self.fetch_num
                ) for n in range(10)]))
            else:
                logging.info('Total MNIST image: {}'.format(self.total_num))

        def __iter__(self):
            digits_cnt, fetch_num = self.digits_cnt, self.fetch_num
            if fetch_num is None:
                for file in self.file_list:
                    for digit, img in file:
                        self.total_num += 1
                        yield [(digit, img)]
            else:
                for file in self.file_list:
                    for digit, img in file:

                        if not len(digits_cnt):
                            logging.info('Enough MNIST image')
                            return

                        if digit not in digits_cnt:
                            continue

                        if digits_cnt[digit] == self.fetch_num:
                            logging.info('fetched enough digit {}'.format(digit))
                            del digits_cnt[digit]
                            continue

                        digits_cnt[digit] += 1
                        yield [(digit, img)]

    class CASIAReader(object):

        def __init__(self, file_list, img_transfer=None):
            assert len(file_list)
            self.file_list = file_list
            self.img_transfer = img_transfer
            self.total_img_num = 0

        def __enter__(self):
            file_num = len(self.file_list)
            extra_num = 1 if not self.img_transfer else len(self.img_transfer)
            logging.info('Expect CASIA image: {}[{}x{}x{}]'.format(
                3755 * file_num * extra_num, 3755, file_num, extra_num))

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def __iter__(self):

            img_transfer = self.img_transfer
            for file in self.file_list:
                img_num = 0
                for ch, img in file:
                    batch = [(ch, img)]
                    if img_transfer:
                        batch += [(ch, trans(img)) for trans in img_transfer]
                    yield batch
                    img_num += len(batch)

                self.total_img_num += img_num
                logging.info('got {} from file {}'.format(img_num, file.name))

    assert data_set.upper() in ('TRAINING', 'TEST', 'ALL')

    factor_33 = [DT.DT_TOP, DT.DT_LEFT, DT.DT_TOP | DT.DT_LEFT]

    factor_19 = [DT.DT_BOTTOM,  DT.DT_RIGHT, DT.DT_TOP | DT.DT_RIGHT,
                 DT.DT_BOTTOM | DT.DT_LEFT, DT.DT_BOTTOM | DT.DT_RIGHT,
                 DT.DT_HORIZON_EXPAND, DT.DT_HORIZON_SHRINK,
                 DT.DT_VERTICAL_EXPAND, DT.DT_VERTICAL_SHRINK]

    factor_15 = [DT.DT_HORIZON_EXPAND | DT.DT_VERTICAL_EXPAND,
                 DT.DT_HORIZON_EXPAND | DT.DT_VERTICAL_SHRINK,
                 DT.DT_HORIZON_SHRINK | DT.DT_VERTICAL_EXPAND,
                 DT.DT_HORIZON_SHRINK | DT.DT_VERTICAL_SHRINK]

    trans_33, trans_19, trans_15 = Deform(3.3), Deform(1.95), Deform(1.6)

    transfer_list = ([lambda im, f=flag: trans_33.transform(im, f) for flag in factor_33] +
                     [lambda im, f=flag: trans_19.transform(im, f) for flag in factor_19] +
                     [lambda im, f=flag: trans_15.transform(im, f) for flag in factor_15])

    def start_reader(mist_files, casi_files, dir_prefix,
                     casi_img_transfer=None, casi_reader_num=3, queue_writer_num=3,
                     need_all_mist=False):

        logging.info('Start preparing image file ......')

        assert casi_files or mist_files

        if casi_files:
            n = max(1, len(casi_files) // casi_reader_num)
            casi_files_batch = [casi_files[i: min(i+n, len(casi_files))] for i in range(0, len(casi_files), n)]

            casi_file_num, casi_boost_num = len(casi_files), len(casi_img_transfer) if casi_img_transfer else 0
            mist_fetch_num = None if need_all_mist else casi_file_num * (1 + casi_boost_num)
        else:
            mist_fetch_num = None
            casi_files_batch = []

        img_queue = queue.Queue(maxsize=100)
        mist_reader = [Producer(MNISTReader(mist_files, mist_fetch_num), img_queue)]
        casi_reader = [Producer(CASIAReader(batch, transfer_list), img_queue) for batch in casi_files_batch]

        tf_writer_seq = [dir_prefix] if queue_writer_num < 2 else (
                        [dir_prefix + '_' + str(i) for i in range(queue_writer_num)])
        tf_writer = [Consumer(mist_reader + casi_reader, TFWriter(dir_with_seq, img_queue), img_queue)
                     for dir_with_seq in tf_writer_seq]

        [ensure_dir('/record/' + dir_with_seq) for dir_with_seq in tf_writer_seq]

        [reader.start() for reader in (mist_reader + casi_reader)]
        [writer.start() for writer in tf_writer]
        [writer.join() for writer in tf_writer]

        total_casi_num = sum([reader.data_src.total_img_num for reader in casi_reader])
        logging.info('Total CASIA image: {}'.format(total_casi_num))

        logging.info('Finish preparing image file')

    def ensure_dir(dirname):
        if gf.Exists(RECORD_ROOT + dirname):
            gf.DeleteRecursively(RECORD_ROOT + dirname)
        gf.MakeDirs(RECORD_ROOT + dirname)

    if data_set.upper() == 'TRAINING':
        mist_file_list = [MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')]
        casi_file_list = [CasiaFile('train/' + name) for name in gf.ListDirectory(PWD + '/casia/train') if
                          not gf.IsDirectory(PWD + '/casia/train/' + name)]
        start_reader(mist_file_list, casi_file_list, 'train', transfer_list)

    if data_set.upper() == 'TEST':
        mist_file_list = [MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')]
        casi_file_list = [CasiaFile('test/' + name) for name in gf.ListDirectory(PWD + '/casia/test') if
                          not gf.IsDirectory(PWD + '/casia/test/' + name)]
        start_reader(mist_file_list, casi_file_list, 'test', queue_writer_num=1)

    if data_set.upper() == 'ALL':
        mist_file_list = [MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'),
                          MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')]
        casi_file_list = ([CasiaFile('train/' + name) for name in gf.ListDirectory(PWD + '/casia/train') if
                           not gf.IsDirectory(PWD + '/casia/train/' + name)] +
                          [CasiaFile('test/' + name) for name in gf.ListDirectory(PWD + '/casia/test') if
                           not gf.IsDirectory(PWD + '/casia/test/' + name)])
        start_reader(mist_file_list, casi_file_list, 'all', transfer_list)


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s][%(threadName)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)

    from data.cv_filter import *
    from data import IMG_SIZE

    filter = AlignFilter(size=(IMG_SIZE, IMG_SIZE), constant_values=0)
    resize = lambda x: filter.filter(x)

    # prepare_label_dict()
    # prepare_image_files('training_set_%d.tfr', data_set='TRAINING', img_filter=resize)
    # prepare_image_files('test_set_%d.tfr', data_set='TEST', img_filter=resize)
    # prepare_image_files('mixing_set_%d.tfr', data_set='ALL', img_filter=resize)

