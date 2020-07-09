from threading import Thread
#from multiprocessing import Process
import numpy as np
import logging
import pickle
import time
class SintelThreadProvider(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None, queue = None, fileList=None, scene_step = 1):
        super().__init__()
        self.name =name
        self.fileList = fileList
        self.loaded_files = 0
        self.num_files = len(fileList)
        self.queue = queue
        self.scene_step = scene_step

    def run(self):
        while(self.loaded_files<self.num_files):
            if(not self.queue.isFull()):
                logging.debug('Getting elements'
                              + ' : ' + str(self.queue.get_size()) + ' items in queue')
                self.queue.put(self.load_next_pickled())
                logging.debug('Depickling has finished'
                              + ' : ' + str(self.queue.get_size()) + ' items in queue')

    def load_next_pickled(self):
        with open(self.fileList[self.loaded_files], "rb") as pickled:
            logging.debug('Depickling has started'
                          + ' : ' + str(self.queue.get_size()) + ' items in queue')
            depickled = pickle.load(pickled)
            depickled = depickled.squeeze()
            one_picklee_with_all_filters = np.swapaxes(depickled, 0, 3)
            one_picklee_with_all_filters = np.swapaxes(one_picklee_with_all_filters, 1, 3)
            one_picklee_with_all_filters = np.swapaxes(one_picklee_with_all_filters, 2, 3)

            ############################################################################
            ##
            ## watchout ! ##############################################################
            ## quick hack here to ensure that only every second scene is loaded ########
            ##
            ##############################################################################
            
            self.loaded_files+=self.scene_step

        #return depickled
        return one_picklee_with_all_filters

    def isReady(self):
        return True if(self.queue.isFull()) else False

class Queue():
    def __init__(self, size):
        self._size = size
        self._elements = list()
        self._isFull = False

    def pop(self):
        if (not len(self._elements)==0):
            self._isFull = False
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


if __name__=="__main__":

    import numpy as np
    import importlib as imp
    import cv2
    import os
    import matplotlib.pyplot as plt
    import gc
    import math
    import time
    import glob

    # Custom libraries/classes

    from components.classes.NumbaSimpleMatcher import Wrapper as Matcher

    from components.classes.Metrix import Wrapper as Metrix
    from components.classes.SintelReader import Wrapper as SintelReader
    from components.classes.CSVReader import Wrapper as CSVReader
    from components.classes.CSVWriter2 import Wrapper as CSVWriter

    LOG_OUTPUT_ROOT = "./optimization/final/"

    LOG_DIRECTORY = LOG_OUTPUT_ROOT + "/numba_sm/"

    SPEC_MARKER = "SM_VGG16_PREPROCESSED_GC"

    CSV_FILE_NAME = SPEC_MARKER + ".csv"

    CSV_FILE_PATH = os.path.join(LOG_DIRECTORY, CSV_FILE_NAME)

    ##################################################################################
    # Instantiating reader and csv writer ############################################
    ##################################################################################

    STEREO_IMGS_ROOT = "../../datasets/sintel/training"

    IMG_READER = SintelReader(rootPath=STEREO_IMGS_ROOT)

    CSV_WRITER = CSVWriter(file_path=CSV_FILE_PATH)


    NUM_SCENES = len(IMG_READER.get_available_scenes())


    WIDTH = 1024
    HEIGHT = 120
    IMG_DIM_FOR_MODEL = [HEIGHT, WIDTH, 3]


    PREPROCESS_IMGS_DIRECTORY = "../../~optimization/pickled_slice_predictions/vgg16/"

    # READER.convolve_slices_and_save(MODEL,  slice_height = HEIGHT, slice_width = WIDTH, output_directory=PREPROCESS_IMGS_DIRECTORY)

    # In[10]:

    ##################################################################################
    # LOADING the preprocess images: one for each scene ##############################
    ##################################################################################

    PREPROCESSED_PICKLES = IMG_READER.list_pickled_convolved(PREPROCESS_IMGS_DIRECTORY, WIDTH, HEIGHT)
    print(len(PREPROCESSED_PICKLES))
    q = Queue(5)
    pickled_file_list = [] #todo here
    dataLoaderThread = SintelThreadProvider(name="data_provider_1", queue=q, fileList = PREPROCESSED_PICKLES)
    dataLoaderThread.start()

    while(not dataLoaderThread.isReady()):
        time.sleep(1)
    print("Loader is ready")
    nextImage = q.pop()
    counter = 1
    while(nextImage is not None):
        nextImage = q.pop()
        print("I have just popped the next element ({0}). Now I am going to sleep for a bit...".format(counter))
        counter+=1
        time.sleep(1)
