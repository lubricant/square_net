
import logging

from data.dt_trans import *
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


def prepare_image_files(file_name, data_set, img_per_file=250000, dict_path='labels_dict.npy', img_filter=lambda _: _):
    '''
    将原始图片文件转换为 TFRecord 格式的文件
    '''

    logging.info('Start preparing image file ......')

    _, id_mapping = np.load(dict_path)

    img_cnt, img_writer, digits_cnt = [0], [], {}

    for i in range(10):
        digits_cnt[str(i)] = 0

    def flush_image(ch_str, img_arr, file_dir):

        writer_id = img_cnt[0] // img_per_file
        if not img_cnt[0] % img_per_file:
            if writer_id:
                img_writer[writer_id-1].close()
            img_writer.append(TFRecordFile((file_dir + '/' + file_name) % writer_id, id_mapping))
            assert writer_id == len(img_writer) - 1

        img_writer[writer_id].write(ch_str, img_filter(img_arr))
        img_cnt[0] += 1

    def flush_mnist_image(file_list, file_dir, fetch_num):

        logging.info('Expect MNIST image: {}[{}x{}]'.format(fetch_num*10, 10,  fetch_num))

        for file in file_list:
            for digit, img in file:

                if not len(digits_cnt.keys()):
                    logging.info('Enough MNIST image')
                    return

                if digit not in digits_cnt:
                    continue

                if digits_cnt[digit] == fetch_num:
                    logging.info('digit {} has fetch enough'.format(digit))
                    del digits_cnt[digit]
                    continue

                digits_cnt[digit] += 1
                flush_image(digit, img, file_dir)

        logging.error('Not enough MNIST image: {}'.format(['%d:%d' % (
            d, fetch_num-digits_cnt[d] if d in digits_cnt else fetch_num) for d in range(10)]))

    def flush_casia_image(file_list, file_dir, img_transfer=None):

        file_num = len(file_list)
        extra_num = 1 if not img_transfer else len(img_transfer)
        logging.info('Expect CASIA image: {}[{}x{}x{}]'.format(
            3755*file_num*extra_num, 3755, file_num, extra_num))

        total_ch_num = 0
        for file in file_list:
            ch_num = 0
            for ch, img in file:
                flush_image(ch, img, file_dir)
                ch_num += 1
                if img_transfer:
                    img_out = np.zeros(img.shape)
                    ch_num += len(img_transfer)
                    for trans in img_transfer:
                        t_img = trans(img, img_out)
                        assert t_img.shape == img.shape
                        flush_image(ch, t_img, file_dir)

            total_ch_num += ch_num
            logging.info('got {} from file {}'.format(ch_num, file.name))

        logging.info('Total CASIA image: {}'.format(total_ch_num))

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

    transfer_list = ([lambda im, out, f=flag: trans_33.transform(im, f, out) for flag in factor_33] +
                     [lambda im, out, f=flag: trans_19.transform(im, f, out) for flag in factor_19] +
                     [lambda im, out, f=flag: trans_15.transform(im, f, out) for flag in factor_15])

    extra_img_num = len(transfer_list)

    if data_set.upper() == 'TRAINING':
        ensure_path('record/train')
        flush_mnist_image([MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')], 'train',
                          fetch_num=240 * (1 + extra_img_num))
        flush_casia_image([CasiaFile('train/' + name) for name in list_file('casia/train')], 'train',
                          img_transfer=transfer_list)

    if data_set.upper() == 'TEST':
        ensure_path('record/test')
        flush_mnist_image([MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')], 'test', fetch_num=60)
        flush_casia_image([CasiaFile('test/' + name) for name in list_file('casia/test')], 'test',)

    if data_set.upper() == 'ALL':
        ensure_path('record/all')
        flush_mnist_image([MnistFile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')], 'all',
                          fetch_num=240 * (1 + extra_img_num))
        flush_mnist_image([MnistFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')], 'all',
                          fetch_num=60 * (1 + extra_img_num))
        flush_casia_image([CasiaFile('train/' + name) for name in list_file('casia/train')] +
                          [CasiaFile('test/' + name) for name in list_file('casia/test')], 'all',
                          img_transfer=transfer_list)

    logging.info('Finish preparing image file')


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)

    from data.cv_filter import *
    from data import IMG_SIZE

    filter = AlignFilter(size=(IMG_SIZE, IMG_SIZE), constant_values=0)
    resize = lambda x: filter.filter(x)

    # prepare_label_dict()
    prepare_image_files('training_set_%d.tfr', data_set='TRAINING', img_filter=resize)
    # prepare_image_files('test_set_%d.tfr', data_set='TEST', img_filter=resize)
    # prepare_image_files('mixing_set_%d.tfr', data_set='ALL', img_filter=resize)

