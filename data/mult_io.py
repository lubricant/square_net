import logging
from threading import *
from queue import *


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






