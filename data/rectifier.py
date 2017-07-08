
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
'''


def prepare_label_dict(dict_path='labels_dict.npy'):
    '''
    构建 label 字典文件并以 npy 格式保存
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


def prepare_image_files(file_name, data_set, dict_path='labels_dict.npy', img_filter=lambda _: _):
    '''
    将原始图片文件转换为 TFRecord 格式的文件
    '''

    logging.info('Start preparing image file ......')

    _, id_mapping = np.load(dict_path)
    tf_writer = TFRecordFile(file_name, id_mapping)

    def flush_mnist_image(file_list, fetch_num=300):
        for file in file_list:
            for ch, img in file:
                if not fetch_num:
                    return
                fetch_num -= 1
                tf_writer.write(ch, img_filter(img))

    def flush_casia_image(file_list):
        for file in file_list:
            for ch, img in file:
                tf_writer.write(ch, img_filter(img))

    assert data_set.upper() in ('TRAINING', 'TEST', 'ALL')

    if data_set.upper() == 'TRAINING':
        flush_mnist_image([MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')])
        flush_casia_image([CasiaFile('train/' + name) for name in list_file('casia/train')])

    if data_set.upper() == 'TEST':
        flush_mnist_image([MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')])
        flush_casia_image([CasiaFile('test/' + name) for name in list_file('casia/test')])

    if data_set.upper() == 'ALL':
        flush_mnist_image([MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'),
                           MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')])
        flush_casia_image([CasiaFile('train/' + name) for name in list_file('casia/train')] +
                          [CasiaFile('test/' + name) for name in list_file('casia/test')])

    tf_writer.close()

    logging.info('Finish preparing image file.')


if __name__ == '__main__':
    # prepare_label_dict()
    # prepare_image_files('training_set.tfr', data_set='TRAINING')
    # prepare_image_files('test_set.tfr', data_set='TEST')
    prepare_image_files('data_set.tfr', data_set='ALL')

