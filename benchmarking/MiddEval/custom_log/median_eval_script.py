import cv2
import pandas as pd
import numpy as np
import project_helpers
import os
from components.utils.Metrix import Wrapper as me
from components.utils import utils as u

file_path = "D:\gdrive\python_projects\FYP_FINAL\\benchmarking\MiddEval\custom_log\\best_results.csv"

pd = pd.read_csv(file_path)

gt_mask = "disp0GT.pfm"
occ_mask = "mask0nocc.png"

average = dict(bm_30_9x3=0, blg_40_5x7gc_8_gs_90_alph_0=0, plusblg_40_5x7gc_3_gs_1_alph_0=0)
for img_path, scene, exp_id in pd[["image_filename", "scene", "experiment_id"]].values:

    root = project_helpers.get_project_dir()
    img_full_path = os.path.join(root, img_path)
    img_raw = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
    img_temp = cv2.bilateralFilter(img_raw, 7, 7,7)
    dirname = os.path.dirname(img_full_path)
    
    gt_path = os.path.join(dirname, gt_mask)
    occ_path = os.path.join(dirname, occ_mask)
    
    
    gt = u.load_pfm(gt_path)[0]
    occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE)
    
    evals = me.evaluate_over_all(img_temp, gt*4, occ, occlusions_counted_in_errors = False)
    average[exp_id]+=evals[2]
    print("{2}:Bad4 for scene {0}: {1:.4f}".format(scene, evals[2], exp_id))

for k, v in average.items():
    print("{0}: {1}".format(k,v/15))