import queue
import logging

from data.mult_io import *
from data.dt_trans import *
from data.fmt_file import *

'''
使用函数前，按照如下目录结构放置好文件：

PWD/data
|----casia：    [CASIA 所有文件]
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


def prepare_image_files(file_name, data_set, img_per_file=250000, ch2idx_dict=None, img_filter=lambda _: _):
    '''
    将原始图片文件转换为 TFRecord 格式的文件
    '''

    assert data_set.upper() in ('TRAINING', 'TEST', 'ALL')

    def start_reader(dir_prefix, casi_files_10, casi_files_11,
                     casi_img_transfer=None, queue_writer_num=3):

        logging.info('Start preparing image file ......')

        assert casi_files_10 or casi_files_11

        img_queue = queue.Queue(maxsize=100)

        casi_reader = []
        if casi_files_10:
            casi_reader += [Producer(CASIAReader(casi_files_10, casi_img_transfer), img_queue)]
        if casi_files_11:
            casi_reader += [Producer(CASIAReader(casi_files_11, casi_img_transfer), img_queue)]

        tf_writer_seq = [dir_prefix] if queue_writer_num < 2 else (
                        [dir_prefix + '_' + str(i) for i in range(queue_writer_num)])
        tf_writer = [Consumer(casi_reader, TFWriter(dir_with_seq, img_queue,
                                                    forward_dict=ch2idx_dict,
                                                    filename_template=file_name,
                                                    img_per_file=img_per_file,
                                                    img_filter=img_filter), img_queue)
                     for dir_with_seq in tf_writer_seq]

        [ensure_dir(dir_with_seq) for dir_with_seq in tf_writer_seq]
        [reader.start() for reader in casi_reader]
        [writer.start() for writer in tf_writer]
        [writer.join() for writer in tf_writer]

        total_casi_num = sum([reader.data_src.total_img_num for reader in casi_reader])
        logging.info('Total CASIA image: {}'.format(total_casi_num))

        logging.info('Finish preparing image file')

    def ensure_dir(dirname):
        if gf.Exists(RECORD_ROOT + '/record/' + dirname):
            gf.DeleteRecursively(RECORD_ROOT + '/record/' + dirname)
        gf.MakeDirs(RECORD_ROOT + '/record/' + dirname)

    if data_set.upper() == 'TRAINING':
        casi_db_10 = CasiaFile.list_file(get_train_set=True, get_test_set=False, use_db_v10=True)
        casi_db_11 = CasiaFile.list_file(get_train_set=True, get_test_set=False, use_db_v11=True)
        start_reader('train', casi_db_10, casi_db_11)

    if data_set.upper() == 'TEST':
        casi_db_10 = CasiaFile.list_file(get_train_set=False, get_test_set=True, use_db_v10=True)
        casi_db_11 = CasiaFile.list_file(get_train_set=False, get_test_set=True, use_db_v11=True)
        start_reader('test', casi_db_10, casi_db_11, queue_writer_num=1)

    if data_set.upper() == 'ALL':
        casi_db_10 = CasiaFile.list_file(get_train_set=True, get_test_set=True, use_db_v10=True)
        casi_db_11 = CasiaFile.list_file(get_train_set=True, get_test_set=True, use_db_v11=True)
        start_reader('all', casi_db_10, casi_db_11, queue_writer_num=5)


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

