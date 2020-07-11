#!/usr/bin/env python
# coding: utf-8

# <p class="MsoNormal" style="line-height: normal; mso-outline-level: 1; margin: 13.5pt 0cm 6.75pt 0cm;"><span style="font-size: 25.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-font-kerning: 18.0pt; mso-fareast-language: EN-GB;">Middlebury Metrics local implementation validation</span></p>
# <p class="MsoNormal" style="line-height: normal; mso-outline-level: 2; margin: 13.5pt 0cm 6.75pt 0cm;"><span style="font-size: 20.5pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Abstract</span></p>
# <p class="MsoNormal" style="line-height: normal; mso-outline-level: 2; margin: 13.5pt 0cm 6.75pt 0cm;"><span style="font-size: 20.5pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Introduction</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Before any experiment is conducted in the world of science, the tools and means used to measure&nbsp; the outcome have to be calibrated and validated, metrics have to be unified. One noteworthy mishap&nbsp;<strong>at Nasa</strong>, for instance,&nbsp; was due to different metric systems used which eventually resulted in a loss of a spacecraft. It is not different when it comes to Computer Science and within that, Stereo Vision. This project was intended to be lead in an incremental manner from multiple aspects. One of them was testing the devised algorithm on multiple datasets each subsequent one bearing an incremented difficulity.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Therefore, different metrics and ways of quantifying the results were encountered. Therefore it was decided that in order to ensure transparency and compatibility, both datasret specific and more general metrics will be used during the experiments. However, this notebook did not aim to cover all aspects of this requirement. A rigorous testing was conducted in the form of unit tests (please see "tests/metrix_functions.py" for further details).&nbsp;</span></p>
# <p class="MsoNormal" style="line-height: normal; mso-outline-level: 3; margin: 13.5pt 0cm 6.75pt 0cm;"><span style="font-size: 17.5pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Relevant theory</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">The Middlebury Stereo dataset&nbsp; became one of the cornerstone datasets of this field starting in 2003. There have been 4 versions with each subsequent one offering improved and new features. The first dataset and its associated metrics to be used&nbsp; was the&nbsp;<strong>Middlebury 2003 dataset</strong>&nbsp;and the&nbsp;<strong>"Middlebury Stereo Evaluation&nbsp; - Version 2".&nbsp;</strong>This evaluation site is now defunct, though the submitted algorithms and their results are still public.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">The reason for using this as a starting point was two-fold:</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">&nbsp;</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">1. The first "Video-mics" algorithm was benchmarked on this dataset. Therefore, in order to be able to evaluate newly implemented algorithms part of this project comparatively, it was thought to be necessary.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">&nbsp;</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">2. This dataset was thought to be small enough to be used as an initial tool for debugging incidental errors in the implementation.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">&nbsp;</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">It consists of 4 rectified image paris ("scenes"), "Tsukaba", "Venus", "Teddy" and "Cones". Occlusions are represented with 0 pixel intensities.&nbsp; The metrics used to evaluate the algorithms was the so called&nbsp;<strong>"Bad N".&nbsp;</strong>This measure calculates the average absolute differences between a disparity map generated by an algorithm and the grountruth pertaining to that particular scene. The&nbsp;<strong>"N"</strong>&nbsp;stands for a threshold value which is used to ignore absolute differences below that value.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">This ranges from 0.5 to 2 with varying steps of 0.25 and 0.5.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Additionally, occlusion masks were provided entailing another dimension for evaluation. If the evaluation is "nonoccluded", only pixels outside of the mask (occlusions) and unknown regions (disparity is unknown) are counted. Otherwise, occluded regions are included as well.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">&nbsp;</span></p>
# <p class="MsoNormal" style="line-height: normal; mso-outline-level: 2; margin: 13.5pt 0cm 6.75pt 0cm;"><span style="font-size: 20.5pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Method</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">In order to establish equivalence between the local implementation of Middlebury 2003 metrics and its online counterpart, the following was performed.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">As a first step, 5 top benchmarking results pertaining to <span style="mso-spacerun: yes;">&nbsp;</span>4 top-performing algorithms were collected from the Evaluation Site's chart into a table. There were&nbsp;<strong>IGSM, MULTIRBF, SEGAGGR and GCC+LocalExp.&nbsp;</strong>Additionally, the performance metrics for each algorithm was downloaded for each threshold value.</span></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><strong><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Then the evaluation results for these algorithms were collected into a table ("\tests\metrix_test\metrics_test_check_result.csv").</span></strong></p>
# <p class="MsoNormal" style="margin-bottom: .0001pt; line-height: normal;"><strong><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">&nbsp;</span></strong><span style="font-size: 10.0pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">The outputs of these algorithms were also downloaded.&nbsp;</span></p>
# <p class="MsoNormal" style="line-height: normal; mso-outline-level: 2; margin: 13.5pt 0cm 6.75pt 0cm;"><span style="font-size: 20.5pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Results and discussion</span></p>
# <p class="MsoNormal" style="line-height: normal; mso-outline-level: 2; margin: 13.5pt 0cm 6.75pt 0cm;"><span style="font-size: 20.5pt; font-family: 'Helvetica',sans-serif; mso-fareast-font-family: 'Times New Roman'; color: black; mso-fareast-language: EN-GB;">Conclusion</span></p>
# <p class="MsoNormal">&nbsp;</p>

# In[1]:


###################################################################
# Importing built-in modules ######################################
###################################################################

import os
import math

ROOT=os.path.join("..", "..")

import sys
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import imp
import cv2
import matplotlib.pyplot as plt
import gc
import glob
import re

###################################################################
# Importing custom modules ######################################
###################################################################


from components.utils import utils as u
from components.utils.Metrix import Wrapper as me
from components.utils import plot_utils as plu
from components.utils.CSVWriter2 import Wrapper as csv


# In[6]:


###################################################################
# Initialising path variables #####################################
###################################################################


ROOT_PATH = os.path.join("..", "..")
EXPERIMENT_TITLE = "Metrics_001_validation"

LOG_FOLDER = os.path.join(ROOT_PATH, "experiments", "logs")
CSV_FILEPATH = os.path.join(LOG_FOLDER, EXPERIMENT_TITLE+".csv")


IMAGE_DIRECTORY = os.path.join(ROOT_PATH, "tests", "metrix_test") 
SELECTED_METRIC = "bad*"
IMG_FORMAT="*.png"


images_to_test = ["venus", "tsukaba", "teddy", "cones"]
image_dict = dict({"venus": [], "tsukaba":[], "teddy":[], "cones":[]})

header_func = lambda  : "ALG_NAME,THRESHOLD,IMG,NONOCC,ALL".split(",")


csv_logger = csv(CSV_FILEPATH, default_header=False)
csv_logger.set_header_function(header_func)
csv_logger.write_csv_header()
csv_logger.set_line_function(csv.format_stereo_matching_results_v2)

###################################################################
# Scanning directory for disparities###############################
###################################################################

folders = glob.glob(os.path.join(IMAGE_DIRECTORY, SELECTED_METRIC, IMG_FORMAT))

for key, value in image_dict.items():
    r = re.compile(".*"+key+".*")
    matches = list(filter(r.match, folders))
    image_dict[key] = matches

###################################################################
# Scanning for ground truth disparities ###########################
###################################################################
    



gt_files = glob.glob(os.path.join(IMAGE_DIRECTORY,  "gt", "*ground*"))
non_occ_files = glob.glob(os.path.join(IMAGE_DIRECTORY,  "gt", "*nonocc*"))
unknown_files = glob.glob(os.path.join(IMAGE_DIRECTORY,  "gt", "*unknown*"))

###################################################################
# Loading benchmarked results into DataFrame#######################
###################################################################

benchmark_results_path = os.path.join("..", "..", "tests", "metrix_test", 
                                      "metrics_test_check_result.csv")
if not os.path.exists(benchmark_results_path):
    raise Exception("Benchmarked results data file does not exist.")
    
###################################################################
# Note that the local dataframe's keys are initialised to 4x of ###
# the original metrics.############################################
###################################################################

local_dataframe_data = dict(ALG_NAME=[], IMG=[], NONOCC=[])
local_dataframe_data["BAD_2.0"] = []
local_dataframe_data["BAD_4.0"] = []
local_dataframe_data["BAD_8.0"] = []
    

online_dataframe = pd.read_csv(benchmark_results_path).dropna()
online_dataframe["ALG_NAME"] = online_dataframe["ALG_NAME"].str.strip()
online_dataframe.info()


###################################################################
# Evaluation of the saved disparity maps ##########################
###################################################################


for gt, non_occ, unknown in zip(gt_files, non_occ_files, unknown_files):
    
    key = gt.split(os.path.sep)[-1].split(".")[0].split("_")[0]
  

    
    occ_key = non_occ.split(os.path.sep)[-1].split(".")[0].split("_")[0]
    
    assert (key == occ_key), "The key and occlusion key does not match. Please use a dictionary instead. \n%s: %s"%(key, occ_key)
    
    gt_raw = cv2.imread(gt, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    
    nonocc_loaded = cv2.imread(non_occ, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    unknown_loaded = cv2.imread(unknown, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    for disp_map in image_dict[key]:
        
        # Submitted result
        
        disp_raw = cv2.imread(disp_map, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        
        label = disp_map.split(os.path.sep)[-1].split(".")[0]
        
        threshold = float(re.search("[0-9]+\.[0-9]+", disp_map)[0])


            # this is probably wrong...
        filename = os.path.split(disp_map)[-1]
        alg_name, scene, threshold = filename[0:filename.rfind("_")], disp_map[int(disp_map.rfind("_")+1):disp_map.rfind("-")], disp_map[disp_map.rfind("-"):]

        # do both occluded and nonoccluded thresholds...
        
        MULTIPLIER = 1
        OFFSET = 0
        BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = me.evaluate_over_all(disp_raw*MULTIPLIER, gt_raw, nonocc_loaded, occlusions_counted_in_errors = False)
        formatted_str = "raw ({4}): bad1:{0:.4f} bad2:{1:.4f}, bad4:{2:.4f}, bad8:{3:.4f}".format(BAD1, BAD2, BAD4, 
                                                                                               BAD8,  label)
        local_dataframe_data["ALG_NAME"].append(alg_name)
        local_dataframe_data["IMG"].append(scene)
        local_dataframe_data["NONOCC"].append(1)
        local_dataframe_data["BAD_2.0"].append(BAD2)
        local_dataframe_data["BAD_4.0"].append(BAD4)
        local_dataframe_data["BAD_8.0"].append(BAD8)

        print(formatted_str)
        main_label =  "raw ({4}):\n bad1:{0:.4f} bad2:{1:.4f}, bad4:{2:.4f}, bad8:{3:.4f}".format(BAD1, BAD2, BAD4, 
                                                                                                  BAD8,  label)
        
        BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = me.evaluate_over_all(disp_raw*MULTIPLIER, gt_raw, unknown_loaded, occlusions_counted_in_errors = False)
        formatted_str = "raw ({4}): bad1:{0:.4f} bad2:{1:.4f}, bad4:{2:.4f}, bad8:{3:.4f}".format(BAD1, BAD2, BAD4, 
                                                                                                  BAD8,  label)
        local_dataframe_data["ALG_NAME"].append(alg_name)
        local_dataframe_data["IMG"].append(scene)
        local_dataframe_data["NONOCC"].append(0)
        local_dataframe_data["BAD_2.0"].append(BAD2)
        local_dataframe_data["BAD_4.0"].append(BAD4)
        local_dataframe_data["BAD_8.0"].append(BAD8)

        #plu.plot_images(list([disp_raw, gt_raw, nonocc_loaded]), list([main_label, label+"_gt", label+"_occl"]))


###################################################################
# Equivalence test between local and published results ############
###################################################################


local_dataframe= pd.DataFrame(local_dataframe_data)
local_dataframe.info()

filtered_local_dataframe = local_dataframe[(local_dataframe["IMG"]=="teddy") | (local_dataframe["IMG"]=="cones")]

"""row = filtered_local_dataframe.values[0]
#print("row values: {0}, {1}, {2}".format(row[0], row[1].upper(), bool(row[2])))
#print(row)
temp_online = online_dataframe[(online_dataframe["ALG_NAME"] == row[0]) &
                        (online_dataframe["IMG"] == row[1].upper()) &
                        (online_dataframe["NONOCCLUDED"] == bool(row[2]))]
print("---------------------------------------------")
obad05 = temp_online[["BAD_0.5"]].values[0]
obad10 = temp_online[["BAD_1.0"]].values[0]
obad20 = temp_online[["BAD_2.0"]].values[0]
lbad2 = round(row[3]*100, 2)
lbad4 = round(row[4]*100, 2)
lbad8= round(row[5]*100, 2)
m1 = lbad2 == obad05[0]
m2 = lbad4 == obad10[0]
m3 = lbad8 == obad10[0]
message = "Alg: {0}," \
         " scene: {1}," \
         " nonocc: {2}, \n"\
      "bad 0.5 vs local bad 2.0 match: {3}\n"\
      "bad 1.0  vs local bad 4.0 match: {4}\n"\
      "bad 2.0 vs local bad 8.0 match: {5}\n"\
      .format(
              temp_online["ALG_NAME"].iloc[0],
              temp_online["IMG"].iloc[0],
              bool(temp_online["NONOCCLUDED"].iloc[0] ), m1, m2, m3 )
print(message)""""""

"""
errors = []
for row in filtered_local_dataframe.values:
    #print("row values: {0}, {1}, {2}".format(row[0], row[1].upper(), bool(row[2])))
    #print(row)
    temp_online = online_dataframe[(online_dataframe["ALG_NAME"] == row[0]) &
                        (online_dataframe["IMG"] == row[1].upper()) &
                        (online_dataframe["NONOCCLUDED"] == bool(row[2]))]
    if(temp_online.size==0):
        raise Exception("Missing entry in the downloaded data. ALG_NAME: {0}, IMG: {1}, NONOCC: {2}".format(row[0], row[1], row[2]))
    print("----------------------------------------------------------")
    obad05 = temp_online[["BAD_0.5"]].values[0]
    obad10 = temp_online[["BAD_1.0"]].values[0]
    obad20 = temp_online[["BAD_2.0"]].values[0]
    lbad2 = round(row[3]*100, 1)
    lbad4 = round(row[4]*100, 1)
    lbad8= round(row[5]*100, 1)



    m1 = lbad2 == obad05[0]
    m2 = lbad4 == obad10[0]
    m3 = lbad8 == obad20[0]

    if not (m1 and m2 and m3):
        errors.append([temp_online["ALG_NAME"].iloc[0],
                  temp_online["IMG"].iloc[0],
                  bool(temp_online["NONOCCLUDED"].iloc[0] ),
                  m1,
                  m2,
                  m3,
                  obad05,
                  lbad2,
                  obad10,
                  lbad4,
                  obad20,
                  lbad8])
    message = "Alg: {0}," \
             " scene: {1}," \
             " nonocc: {2}, \n"\
          "bad 0.5 ({6}) vs local bad 2.0 ({7}) match: {3}\n"\
          "bad 1.0 ({8}) vs local bad 4.0 ({9}) match: {4}\n"\
          "bad 2.0 ({10}) vs local bad 8.0 ({11}) match: {5}\n"\
          .format(
                  temp_online["ALG_NAME"].iloc[0],
                  temp_online["IMG"].iloc[0],
                  bool(temp_online["NONOCCLUDED"].iloc[0] ),
                  m1,
                  m2,
                  m3,
                  obad05,
                  lbad2,
                  obad10,
                  lbad4,
                  obad20,
                  lbad8
    )
    print(message)

for err in errors:
    message = "------------------------------Discrepancy found ---------------------------------------\n" \
              "Alg: {0}," \
              " scene: {1}," \
              " nonocc: {2}, \n" \
              "bad 0.5 ({6}) vs local bad 2.0 ({7}) match: {3}\n" \
              "bad 1.0 ({8}) vs local bad 4.0 ({9}) match: {4}\n" \
              "bad 2.0 ({10}) vs local bad 8.0 ({11}) match: {5}\n" \
        .format(
        err[0],
        err[1],
        err[2],
        err[3],
        err[4],
        err[5],
        err[6],
        err[7],
        err[8],
        err[9],
        err[10],
        err[11]
    )
    print(message)


