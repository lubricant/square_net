import time
from threading import *
from queue import Queue


class MIO(Thread):

    def __init__(self, data_queue, **args):
        super().__init__(**args)
        assert isinstance(data_queue, Queue)
        self.data_queue = data_queue


class Producer(MIO):

    def __init__(self, data_queue, **args):
        super().__init__(data_queue, **args)
        self.end_sign = Event()

    def produce(self):
        raise NotImplementedError

    def run(self):
        while True:
            try:
                item = self.produce()
            except StopIteration:
                break
            if item is None:
                break
            self.data_queue.put(item)
        self.end_sign.set()


class Consumer(MIO):

    def __init__(self, end_events, data_queue, **args):
        super().__init__(data_queue,  **args)
        assert isinstance(end_events, (list, tuple)) and \
               all(isinstance(e, Event) for e in end_events)
        self.end_events = end_events

    def consume(self, item):
        raise NotImplementedError

    def run(self):
        while not all(e.is_set() for e in self.end_events):
            item = self.data_queue.get(timeout=1)
            if item is not None:
                self.consume(item)


class MnistProducer(Producer):

    def __init__(self, fetch_num, file_list, data_queue, **args):
        super().__init__(data_queue, **args)

        self.digits_cnt = {}
        for i in range(10):
            self.digits_cnt[str(i)] = 0

        self.file_list = file_list

    def produce(self):
        pass
        # for file in self.file_list:
        #     for digit, img in file:
        #
        #         if not len(digits_cnt.keys()):
        #             logging.info('Enough MNIST image')
        #             return
        #
        #         if digit not in digits_cnt:
        #             continue
        #
        #         if digits_cnt[digit] == fetch_num:
        #             logging.info('fetched enough digit {}'.format(digit))
        #             del digits_cnt[digit]
        #             continue
        #
        #         digits_cnt[digit] += 1
        #         flush_image(digit, img, file_dir)
        #
        # logging.error('Not enough MNIST image: {}'.format(['%d:%d' % (
        #     d, fetch_num-digits_cnt[d] if d in digits_cnt else fetch_num) for d in range(10)]))


if __name__ == '__main__':

    queue = Queue(maxsize=2)

    class P(Producer):
        cnt = 3

        def produce(self):
            if self.cnt == 0:
                return None
            self.cnt -= 1
            return self.cnt

    class C(Consumer):
        def consume(self, item):
            print(item)

    p1, p2 = P(queue), P(queue)
    c = C([p1.end_sign, p2.end_sign], queue)

    p1.start()
    p2.start()
    c.start()






