import time
import sys
from datetime import datetime
class SimpleTimer():
    instance = None
    def __init__(self):
        self._start=0
        self._end=0
        self._time_delta = 0
    @staticmethod
    def get_instance():

        if(SimpleTimer.instance==None):
            SimpleTimer.instance = SimpleTimer()
        return SimpleTimer.instance

    @staticmethod
    def timeit():
        timer_instance = SimpleTimer.get_instance()
        if timer_instance._start==0:
            timer_instance._start=time.time()
        else:
            timer_instance._end = time.time()
            SimpleTimer.print_elapsed_time()

    @staticmethod
    def print_elapsed_time():
        timer_instance = SimpleTimer.get_instance()
        timer_instance._time_delta = timer_instance._end - timer_instance._start
        output = 'Time delta: '+str(timer_instance._time_delta)+' seconds.'
        sys.stdout.write(output)
        timer_instance._start=0
    @staticmethod
    def print_with_timestamp(message_to_output):
        output = str(datetime.now())+": "+message_to_output+"\n"
        sys.stdout.write(output)
