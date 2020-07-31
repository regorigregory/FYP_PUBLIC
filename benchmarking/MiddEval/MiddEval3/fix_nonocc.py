# Script to fix different calculation methods
import project_helpers
import glob
import os
from components.utils.Metrix import Wrapper as me
from components.utils.CSVWriter2 import Wrapper as csv
from components.utils import utils as u
import cv2
import numpy as np
from components.utils.SimpleProgressBar import SimpleProgressBar

def custom_benchmarking(EXP_PARAMS, disp, gt, occ, max_disp, ARE_OCCLUSIONS_ERRORS = False):
    gt_scaled = np.where(gt==np.inf, 0, gt)
    """dmin = gt.min()
    dmax = gt_temp.max()
    print("dmin: {0}, dmax: {1}".format(dmin, dmax))

    scale = 1.0 / (dmax-dmin)
    gt_scaled = scale*(gt-dmin)
    gt_scaled = np.where(gt_scaled==np.inf, 0, gt_scaled)
    gt_scaled  /= 1.15 + 0.1
    print("gt_scaled_min: {0}, gt_scaled_max: {1}".format(gt_scaled.min(), gt_scaled.max()))
    """
    EXP_PARAMS["are_occlusions_errors"] = ARE_OCCLUSIONS_ERRORS
    occ = gt_scaled if ARE_OCCLUSIONS_ERRORS else occ

    EXP_PARAMS["bad1"], EXP_PARAMS["bad2"], EXP_PARAMS["bad4"],\
    EXP_PARAMS["BAD8"], EXP_PARAMS["abs_error"], EXP_PARAMS["mse"],\
    EXP_PARAMS["avg"], EXP_PARAMS["eucledian"]  = \
        me.evaluate_over_all(disp, gt_scaled, occ, occlusions_counted_in_errors=ARE_OCCLUSIONS_ERRORS)

if __name__ == "__main__":
    from skimage.filters import median
    cur_dir = os.path.dirname(__file__)
    sub_dir = "trainingQ"
    middle_dir = "*"
    file_mask = "disp0blg_40_7x3gc_8_gs_90_alph_0*.png"
    gt_mask = "disp0GT.pfm"
    nonocc_mask = "mask0nocc.png"


    img_files = glob.glob(os.path.join(cur_dir, sub_dir, middle_dir, file_mask))
    for i, img_file_path in enumerate(img_files):

        filename = os.path.split(img_file_path)[1][:-4] #no extension
        new_filename = "median_"+filename+".png"

        filename_split = filename.split("_")

        MATCH = int(filename_split[1])
        GAP = -20
        EGAP= -1

        dir_name = os.path.dirname(img_file_path)

        gt_path = os.path.join(dir_name, gt_mask)
        nonocc_path = os.path.join(dir_name, nonocc_mask)

        disp = median(cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE))
        gt = u.load_pfm(gt_path)[0]*4
        occ = cv2.imread(nonocc_path, cv2.IMREAD_GRAYSCALE)
        new_file_path = os.path.join(dir_name, new_filename)
        cv2.imwrite(new_file_path, disp)


        DATASET = "Middlebury_2014"
        EXPERIMENT_TITLE = filename[5:]

        EXP_PARAMS = {"experiment_id": "median_"+EXPERIMENT_TITLE, "match": MATCH, "gap": GAP, "egap": EGAP,
                      "algo":"components.matchers.NumbaPatchMatcher", "init_method": "default", "dataset": DATASET,
                      "preprocessing_method": "None", "kernel_size": 1, "kernel_spec": "None",
                      "init_method": "maclean_et_al"}

        EXP_PARAMS["runtime"] = 0


        EXP_PARAMS["preprocessing_method"] = "None"

        EXP_PARAMS["scene"] = dir_name.split("\\")[-2]

        path_start =  dir_name.index("benchmarking")
        EXP_PARAMS["image_filename"] = os.path.join(dir_name[path_start:], new_filename)

        EXP_PARAMS["img_res"] = "{0}x{1}".format(disp.shape[1], disp.shape[0])
        EXP_PARAMS["kernel_size"] = "{0}".format(filename_split[2])


        custom_benchmarking(EXP_PARAMS, disp, gt, occ, "None", ARE_OCCLUSIONS_ERRORS = False)

        CSV_FILEPATH = os.path.join("..", "custom_log", "all_benchmarking_fix.csv")

        csv_logger = csv(CSV_FILEPATH, default_header=False)
        csv_logger.set_header_function(csv_logger.get_header_v3)
        csv_logger.write_csv_header()
        csv_logger.set_line_function(csv.format_stereo_matching_results_v2)
        csv_logger.append_new_sm_results(EXP_PARAMS, selected_keys=csv.get_header_v3())

        SimpleProgressBar.get_instance().progress_bar(int(i+1), len(img_files), header = "Fixing logs:")