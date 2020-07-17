#from multiprocessing import Process as Thread
from multiprocessing import Queue
from threading import Thread
import time
import logging
import random

#Own implementation
#Python's is, surprisingly, better.
#However, at that time, I was more focused on implementing the idea
#Than actually looking up python's own API


class ProducerThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None, daemon=None,  queue=None):
        super(ProducerThread, self).__init__()
        self.target = target
        self.name = name
        self.queue = queue

    def run(self):
        while True:
            if not self.queue.full():
                item = random.randint(1, 10)
                self.queue.put(item)
                logging.debug('Putting ' + str(item)
                              + ' : ' + str(self.queue.qsize()) + ' items in queue')
        return


class ConsumerThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None, daemon=None, queue=None):
        super(ConsumerThread, self).__init__()
        self.target = target
        self.name = name
        self.queue = queue
        self.false = False
        return

    def run(self):
        while True:
            if not self.queue.empty():
                item = self.queue.get()
                time.sleep(2)
                logging.debug('Getting ' + str(item)
                              + ' : ' + str(self.queue.qsize()) + ' items in queue')
                self.false = True
                time.sleep(random.random())

        return

"""
class Queue():
    def __init__(self, size):
        self._size = size
        self._elements = list()

    def pop(self):
        if (not len(self._elements)==0):
            return self._elements.pop(0)
        else:
            return None
    def put(self, new_element):
        if( len(self._elements) <= self._size):
            self._elements.append(new_element)
    def isFull(self):
        return True if len(self._elements) == self._size else False
    def isEmpty(self):
        return True if len(self._elements) == 0 else False
    def get_size(self):
        return len(self._elements)

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )

"""
def printIt(q, num):
    print(q.qsize())
    print(num)
    time.sleep(10)
if __name__ == '__main__':
    BUF_SIZE = 10

    q = Queue(maxsize = BUF_SIZE)
    num = 10
    p = ProducerThread(name='producer', queue=q, daemon=True)
    c = ConsumerThread(name='consumer', queue=q, daemon=True)
    d = Thread(target=printIt, args=(q,num, ), daemon=True)
    p.start()
    c.start()
    d.start()
    time.sleep(20)

    p.join()
    c.join()
    d.join()

