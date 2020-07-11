#!/usr/bin/env python
# coding: utf-8

# <h1>SM-NUMBA-SINTEL-convolved
# </h1><br>
# 

# In[1]:


##################################################################################
# If run in google colab      #################################################### 
##################################################################################
"""
from google.colab import drive
drive.mount("/content/gdrive")
get_ipython().run_line_magic('cd', '"./gdrive/My Drive/python_projects/FYP/"')
get_ipython().system('pip install tensorflow-gpu==2.1')
"""

# In[2]:


# built in modules and libraries

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



# singleton class: we will only have one progress bar
from components.classes.SimpleProgressBar import SimpleProgressBar


# Custom libraries/utils

from components.utils import plot_utils as plu
from components.utils import utils as u


# ##Configuring and testing data generator

# In[4]:


##################################################################################
# Setting default parameters  #################################################### 
##################################################################################

LOG_OUTPUT_ROOT = "./optimization/final/"

LOG_DIRECTORY = LOG_OUTPUT_ROOT+"/numba_sm/"

SPEC_MARKER = "SM_VGG16_PREPROCESSED_GC"

CSV_FILE_NAME =  SPEC_MARKER+".csv"

CSV_FILE_PATH = os.path.join(LOG_DIRECTORY, CSV_FILE_NAME)

##################################################################################
# Instantiating reader and csv writer ############################################
##################################################################################


STEREO_IMGS_ROOT = "./datasets/sintel/training"

IMG_READER = SintelReader(rootPath=STEREO_IMGS_ROOT)

CSV_WRITER = CSVWriter(file_path = CSV_FILE_PATH)

PROGRESS_VIEWER = SimpleProgressBar.get_instance()

NUM_SCENES = len(IMG_READER.get_available_scenes())

##################################################################################
# Testing reader configuration ###################################################
##################################################################################
## CSV writer should output if there has been a permission error or file already exists (which is not a problem)

IMG_READER.set_selected_scene('cave_4')

loaded_imgs = IMG_READER.get_selected_scene_next_files()

IMG_READER.plot_images_to_compare(loaded_imgs)


# In[52]:


##################################################################################
# SPEC_DICT: columns that will be written to csv##################################
##################################################################################

SPEC_DICT = dict({})

SPEC_DICT["is_parallel"] = "True"

SPEC_DICT["alg_type"] =  SPEC_MARKER
SPEC_DICT["is_img_preprocessed"] = "yes"



SPEC_DICT["filter_strategy"] = "-"
SPEC_DICT["matching_mode"] =  "occlusions_added_to_the_groundtruth"
SPEC_DICT["matrix_init_mode"] = 2

# Have to be updated iteratively
SPEC_DICT["img_name_pretag"] = SPEC_MARKER
SPEC_DICT["img_name"] = SPEC_MARKER
SPEC_DICT["convolution_filters"] = SPEC_MARKER


# In[ ]:


##################################################################################
# Making sure that Tensorflow session is placed on the GPU, if available #########
##################################################################################
"""
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1000),
          tf.config.LogicalDeviceConfiguration(memory_limit=1000)])

    logical_devices = tf.config.list_logical_devices('GPU')
    assert len(logical_devices) == len(physical_devices) + 1

    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1000),
          tf.config.LogicalDeviceConfiguration(memory_limit=1000)])
except:
    # Invalid device or cannot modify logical devices once initialized.
    pass

"""
# In[6]:


##################################################################################
# Currently supported models: see tf_utils.py in components/utils ################
##################################################################################

#from components.utils import tf_utils as tfu

##################################################################################
# Specifying IMG Dimensions for the model#########################################
##################################################################################

WIDTH = 1024
HEIGHT = 120
IMG_DIM_FOR_MODEL = [HEIGHT,WIDTH, 3]


##################################################################################
# Getting the "first-layer" model ################################################
##################################################################################

#MODEL = tfu.get_vgg16_first_layer_model(IMG_DIM_FOR_MODEL)
#MODEL.summary()

##################################################################################
# Uncomment the line below to check model configuration ##########################
##################################################################################

#model.get_config()


# In[9]:


##################################################################################
# PRE-PROCESSING IMAGES: first image in each scene ###############################
## CAVEAT :FAST-EXECUTION BUT LOADS OF STORAGE SPACE: RUN IT EVERY TIME ... ######
## ... if new execution environment: exclude them from GIT push    ###############
##################################################################################

PREPROCESS_IMGS_DIRECTORY = "./~optimization/pickled_slice_predictions/vgg16/"

#READER.convolve_slices_and_save(MODEL,  slice_height = HEIGHT, slice_width = WIDTH, output_directory=PREPROCESS_IMGS_DIRECTORY)


# In[10]:


##################################################################################
# LOADING the preprocess images: one for each scene ##############################
##################################################################################

PREPROCESSED_PICKLES = IMG_READER.list_pickled_convolved(PREPROCESS_IMGS_DIRECTORY, WIDTH, HEIGHT)

PREPROCESSED_PICKLES = sorted(PREPROCESSED_PICKLES)

##################################################################################
# TESTING LOADED IMAGES VISUALLY #################################################
##################################################################################

one_pickle = IMG_READER.load_pickled_convolved(PREPROCESSED_PICKLES[0])
selected_filter_index = 0

print("Found images: {0}".format(len(PREPROCESSED_PICKLES)))
print("Each image's shape: {0}".format(one_pickle.shape))

fig = plt.figure()
ax = fig.gca()

ax.set_title("Example loaded convolved slice (filter {0})".format(selected_filter_index))
plt.imshow(one_pickle[0, :, :, selected_filter_index])


# ##Initialising loop parameters

# In[11]:


##################################################################################
# Uncomment to see available scnenes #############################################
##################################################################################

#SCENES = reader.get_available_scenes()
#SCENES = sorted(SCENES)
#print(SCENES)

##################################################################################
# LOADING Grountruth images: only 23, can be loaded into memory###################
##################################################################################

STARTING_ROW_FOR_SLICES = 0
ALL_DISPARITIES = IMG_READER.get_all_sliced_disparities(HEIGHT, WIDTH, STARTING_ROW_FOR_SLICES = STARTING_ROW_FOR_SLICES)


# In[12]:


##################################################################################
# GETTING N BEST PARAMETERS FROM THE SELECTED FILE ###############################
##################################################################################

SELECTED_CSV_FILE_PATH = "./optimization/final/numba_legacy/legacy_mixed_lines.csv"
CSV_READER = CSVReader(SELECTED_CSV_FILE_PATH)
CSV_FILE_DESCRIPTION = CSV_READER.get_description()

print(CSV_FILE_DESCRIPTION)


# In[17]:


top_n_config = []

n = 3


selected_metric = "ABS_ERR"


res = CSV_READER.get_best_n_params(n, selected_metric)
top_n_config.append(res)

selected_metric = "mse"

res = CSV_READER.get_best_n_params(n, selected_metric)
top_n_config.append(res)

selected_metric = "euclid_distance"

res = CSV_READER.get_best_n_params(n, selected_metric)
top_n_config.append(res)

top_n_config.append([60, -20, -1, 0])
LOADED_BEST_PARAMS = np.vstack(top_n_config)
print(LOADED_BEST_PARAMS.shape[0])
print("Please enter the estimated runtime per images in seconds below:")

runtime_multiplyer = 10
#runtime_multiplyer = input()


# In[24]:


NUMBER_OF_ITERATIONS = len(LOADED_BEST_PARAMS)  * NUM_SCENES *64
est_runtime = int(NUMBER_OF_ITERATIONS) * float(runtime_multiplyer)

print("The number of iterations based on the params: {0}".format(round(NUMBER_OF_ITERATIONS)))
print("The amount of time (s) needed to perform the parameter search: {0}".format(est_runtime))
print("The amount of time (h) needed to perform the parameter search: {0}".format(est_runtime/3600))


# In[53]:


import importlib
from components.utils import parallel_optimization as po
importlib.reload(po)


# In[ ]:


#def parallel_exec(PREPROCESSED_PICKLES, ALL_DISPARITIES, SPEC_DICT, CSV_WRITER, LOADED_BEST_PARAMS, NUMBER_OF_ITERATIONS, PROGRESS_VIEWER):
iterator = po.parallel_exec(PREPROCESSED_PICKLES, ALL_DISPARITIES, SPEC_DICT, CSV_WRITER, LOADED_BEST_PARAMS, NUMBER_OF_ITERATIONS, PROGRESS_VIEWER)


# In[47]:


print(list(iterator))


# ##Running the optimisation

# In[ ]:




# In[ ]:




