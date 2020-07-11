#!/usr/bin/env python
# coding: utf-8

# <h1>Experiment I: simple parallelisation of NW alg</h1><br>
# ...explanation should go here ...

# In[24]:


from components.classes import ParallelMatcherNW2 as m
from components.utils import utils as u
from components.classes import ExperimentLogger as l
from components.classes import Metrix as Metrix
import imp
import cv2
import os
import time


# Loading the images

# In[28]:


resolutions = list(["", "H", "F"])
selectedResolutionIndex = 0
im2 = cv2.imread("./datasets/middlebury_2003/cones"+resolutions[selectedResolutionIndex]+"/im2.ppm", cv2.IMREAD_GRAYSCALE)
im6 = cv2.imread("./datasets/middlebury_2003/cones"+resolutions[selectedResolutionIndex]+"/im6.ppm", cv2.IMREAD_GRAYSCALE)
test_folder = "./test_outputs"
image_extension = "_medium.ppm"
u.plotTwoImages(im2, im6)    
me = Metrix.Metrix


# As a start, the experiment's parameters' have to be specified in order to ensure proper logging and output file names.

# In[3]:


imp.reload(l)
spec_dict = dict({})
spec_dict["output_folder"] = "test_outputs"

spec_dict["is_parallel"] = "True"
spec_dict["img_name"] = "cones"
spec_dict["alg_type"] =  "parallel"
spec_dict["is_img_preprocessed"] = "False"
spec_dict["convolution_filters"] = "-"
spec_dict["filter_strategy"] = "-"
spec_dict["matching_mode"] = "non_occ"

spec_dict["matrix_init_mode"] = 0
match = spec_dict["match"] = 60
gap = spec_dict["gap"] = -20
egap = spec_dict["egap"] = -1


# In[4]:


log = l.ExperimentLogger()
log.set_matcher_specs(spec_dict)
matcher = m.getOne(match, gap, egap)
matcher.initialize(im2, im6)


# Additional parameters to be set:
# <h1> #todo: load groundtruth and occluded areas </h1>

# In[9]:


gt_path = "../../../optimization/metrix_test/gt/cones_groundtruth.png"
groundtruth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
occlusion_path = "../../../optimization/metrix_test/gt/cones_nonocc.png"
occlusion_map = cv2.imread(occlusion_path, cv2.IMREAD_GRAYSCALE)
u.plotTwoImages(groundtruth, occlusion_map)    


# In[6]:


#todo: load groundtruth and occluded areas
log.log_matching_started()
tic = time.time()
matcher = u.executeParallelMatching(matcher)
toc = time.time()


# In[11]:


disp = matcher.lastDisparity
u.plotTwoImages(groundtruth, disp)
if (spec_dict["matching_mode"]=="non_occ"):
    non_occ = True
    groundtruth = u.add_occlusions(groundtruth, occlusion_map)
    disp = u.add_occlusions(disp, occlusion_map)
    
u.plotTwoImages(groundtruth, disp)


# In[37]:


#here you have to load the ground truth so that you can pass it to the spec dict
import imp
imp.reload(Metrix)
me = Metrix.Metrix


# In[38]:


wanna_print = True
spec_dict["runtime"] = toc-tic
spec_dict["bad1"] = me.bad(disp, groundtruth, threshold = 1.0, non_occ= non_occ, print=wanna_print, occlusion_map = occlusion_map)
spec_dict["bad15"] = me.bad(disp, groundtruth, threshold = 1.5, non_occ= non_occ, print=wanna_print, occlusion_map = occlusion_map)
spec_dict["bad2"] = me.bad(disp, groundtruth, threshold = 2, non_occ= non_occ, print=wanna_print, occlusion_map = occlusion_map)
spec_dict["BAD8"] = me.bad(disp, groundtruth, threshold = 10.0, non_occ= non_occ, print=wanna_print, occlusion_map = occlusion_map)
spec_dict["ABS_ERR"] = me.ABS_ERR(disp, groundtruth, non_occ= non_occ, print=wanna_print, occlusion_map = occlusion_map)
filename = u.get_output_filename(spec_dict)
spec_dict["output_file_path"] = os.path.join(spec_dict["output_folder"], filename)
log.log_matching_ended(spec_dict)
cv2.imwrite(spec_dict["output_file_path"], disp)


# In[ ]:




