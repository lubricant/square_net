import logging
from threading import *
from queue import *

from data.fmt_file import MnistFile, CasiaFile, TFRecordFile


class MIO(Thread):

    def __init__(self, data_queue, ctx_mgr=None, **args):
        super().__init__(**args)
        assert isinstance(data_queue, Queue)
        assert ctx_mgr is None or MIO.as_ctx_mgr(ctx_mgr)
        self.data_queue = data_queue
        self.ctx_mgr = ctx_mgr

    def execute(self):
        raise NotImplementedError

    def run(self):
        if not self.ctx_mgr:
            self.execute()
        else:
            with self.ctx_mgr:
                self.execute()

    @staticmethod
    def as_ctx_mgr(obj):
        if hasattr(obj, '__enter__') and hasattr(obj, '__exit__'):
            return obj


class Producer(MIO):

    __id = 0

    def __init__(self, data_source, data_queue):
        Producer.__id += 1
        super().__init__(data_queue, MIO.as_ctx_mgr(data_source), name='Producer-%s' % Producer. __id)
        assert data_source is not None and (
            hasattr(data_source, '__iter__'))
        self.data_src = data_source
        self.data_iter = iter(data_source)
        self.end_sign = Event()

    def produce(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            return None

    def execute(self):
        while True:
            item = self.produce()
            if item is None:
                break
            self.data_queue.put(item)
        self.end_sign.set()


class Consumer(MIO):

    __id = 0

    def __init__(self, workers, data_sink, data_queue):
        Consumer.__id += 1
        super().__init__(data_queue, MIO.as_ctx_mgr(data_sink), name='Consumer-%s' % Consumer.__id)
        assert isinstance(workers, (list, tuple)) and (
               all(isinstance(x, Producer) for x in workers))
        assert data_sink is not None and (
            hasattr(data_sink, '__call__'))
        self.data_sink = data_sink
        self.end_events = [x.end_sign for x in workers]

    def consume(self, item):
        self.data_sink(item)

    def execute(self):
        while True:
            try:
                item = self.data_queue.get(timeout=1)
            except Empty:
                item = None

            if item is not None:
                self.consume(item)
            else:
                if all(e.is_set() for e in self.end_events):
                    break


class TFWriter(object):

    def __init__(self, file_dir, img_queue, forward_dict, filename_template, img_per_file, img_filter=None):
        self.file_dir = file_dir
        self.img_queue = img_queue
        self.img_num, self.img_writer = 0, []
        self.forward_dict = forward_dict
        self.filename_template = filename_template
        self.img_per_file = img_per_file
        self.img_filter = img_filter

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
        writer_id = self.img_num // self.img_per_file
        if writer_id >= len(img_writer):
            if writer_id:
                img_writer[writer_id - 1].close()
            img_writer.append(TFRecordFile((self.file_dir + '/' + self.filename_template) % writer_id, self.forward_dict))
            assert writer_id == len(img_writer) - 1

        for ch, im in data_batch:
            img_writer[writer_id].write(ch, self.img_filter(im))
        self.img_num += len(data_batch)
        # print(self.img_queue.qsize())


class MNISTReader(object):

    def __init__(self, file_tup_list, fetch_num=None):
        assert len(file_tup_list)
        self.file_list = [MnistFile(*t) for t in file_tup_list]
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
        self.file_list = [CasiaFile(f) for f in file_list]
        self.img_transfer = img_transfer
        self.total_img_num = 0

    def __enter__(self):
        file_num = len(self.file_list)
        extra_num = 0 if not self.img_transfer else len(self.img_transfer)
        logging.info('Process {} CASIA files with extra {} images]'.
                     format(file_num, extra_num))

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
                    print(batch[-1][1].dtype, batch[-1])
                yield batch
                img_num += len(batch)

            self.total_img_num += img_num
            logging.info('got {} from file {}'.format(img_num, file.name))


class HITReader(object):

    def __init__(self, file_list, ch_dict, img_transfer=None):
        assert len(file_list)
        self.file_list = [CasiaFile(f) for f in file_list]
        self.ch_dict = ch_dict
        self.img_transfer = img_transfer
        self.total_img_num = 0

    def __enter__(self):
        file_num = len(self.file_list)
        extra_num = 0 if not self.img_transfer else len(self.img_transfer)
        logging.info('Process {} HIT files with extra {} images]'.
                     format(file_num, extra_num))

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):

        img_transfer = self.img_transfer
        for file in self.file_list:
            img_num = 0
            for ch, img in file:
                if ch not in self.ch_dict:
                    continue
                batch = [(ch, img)]
                if img_transfer:
                    batch += [(ch, trans(img)) for trans in img_transfer]
                yield batch
                img_num += len(batch)

            self.total_img_num += img_num
            logging.info('got {} from file {}'.format(img_num, file.name))


if __name__ == '__main__':

    queue = Queue(maxsize=2)

    def xrange(a, b):
        for i in range(a, b):
            yield i

    p1, p2 = Producer(xrange(1, 6), queue), Producer(xrange(6, 11), queue)
    c = Consumer([p1, p2], lambda x: print(x), queue)

    p1.start()
    p2.start()
    c.start()






