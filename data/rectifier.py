
import logging

from data.fmt_file import *

'''
使用函数前，按照如下目录结构放置好文件：
data
|----casia
|   |----train：[CASIA 训练集文件 (.gnt)]
|   |----test： [CASIA 测试集文件 (.gnt)]
|----mnist：    [MNIST 所有文件] 
|----record:    [留空]
|   |----train：[TFRecord 格式训练集文件 (.tfr)]
|   |----test： [TFRecord 格式测试集文件 (.tfr)]
|   |----mix：  [TFRecord 格式数据集文件 (.tfr)]
|----tmp:       [留空]
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
        [CasiaFile('test/' + name) for name in list_file('casia/test')])

    np.save(get_path(dict_path), (ch_mapping, id_mapping))

    logging.info('Finish preparing label dict.')


def prepare_image_files(file_name, data_set, img_per_file=100000, dict_path='labels_dict.npy', img_filter=lambda _: _):
    '''
    将原始图片文件转换为 TFRecord 格式的文件
    '''

    logging.info('Start preparing image file ......')

    _, id_mapping = np.load(dict_path)

    img_cnt, img_writer = [0], []

    def flush_image(ch_str, img_arr, file_dir):

        writer_id = img_cnt[0] // img_per_file
        if not img_cnt[0] % img_per_file:
            img_writer.append(TFRecordFile((file_dir + '/' + file_name) % writer_id, id_mapping))
            assert writer_id == len(img_writer) - 1

        img_writer[writer_id].write(ch_str, img_filter(img_arr))
        img_cnt[0] += 1

    def flush_mnist_image(file_list, file_dir, fetch_num=300):
        for file in file_list:
            for record in file:
                if not fetch_num:
                    return
                fetch_num -= 1
                flush_image(*record, file_dir)

    def flush_casia_image(file_list, file_dir):
        for file in file_list:
            for record in file:
                flush_image(*record, file_dir)

    assert data_set.upper() in ('TRAINING', 'TEST', 'ALL')

    if data_set.upper() == 'TRAINING':
        ensure_path('record/train')
        flush_mnist_image([MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')], 'train')
        flush_casia_image([CasiaFile('train/' + name) for name in list_file('casia/train')], 'train')

    if data_set.upper() == 'TEST':
        ensure_path('record/test')
        flush_mnist_image([MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')], 'test')
        flush_casia_image([CasiaFile('test/' + name) for name in list_file('casia/test')], 'test')

    if data_set.upper() == 'ALL':
        ensure_path('record/all')
        flush_mnist_image([MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'),
                           MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')], 'all')
        flush_casia_image([CasiaFile('train/' + name) for name in list_file('casia/train')] +
                          [CasiaFile('test/' + name) for name in list_file('casia/test')], 'all')

    [w.close() for w in img_writer]

    logging.info('Finish preparing image file.')


if __name__ == '__main__':

    from data.cv_filter import *
    from data import IMG_SIZE

    filter = AlignFilter(size=(IMG_SIZE, IMG_SIZE), constant_values=255)
    resize = lambda x: filter.filter(x)

    # prepare_label_dict()
    # prepare_image_files('training_set_%d.tfr', data_set='TRAINING', img_filter=resize)
    # prepare_image_files('test_set_%d.tfr', data_set='TEST', img_filter=resize)
    # prepare_image_files('mixing_set_%d.tfr', data_set='ALL', img_filter=resize)

