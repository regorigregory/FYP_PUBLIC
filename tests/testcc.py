from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import math

def foo(i):
    print("task %d started" % i)
    return i*2

def fee(i):
    for x in range(100000):
        math.cos(x)
    return i

def thread_exec(func, args):
    with ProcessPoolExecutor(max_workers = 10) as exec:
        result = exec.map(func, args)
        return result
if __name__ == '__main__':
   z = thread_exec(fee, [i for i in range(10000)])