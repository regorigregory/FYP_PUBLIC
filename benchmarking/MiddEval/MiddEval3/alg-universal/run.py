import time
import numpy as np
import sys
import cv2
import os
from skimage.filters import median
from skimage.morphology import disk
from components.matchers.NumbaPatchMatcher import Wrapper as m


def StrToBytes(text):
    if sys.version_info[0] == 2:
        return text
    else:
        return bytes(text, 'UTF-8')

def benchmark_me(img1, img2, MATCH, GAP, EGAP, kernel, ndisp):

    matcher = m(MATCH, GAP, EGAP, verbose=True)
    matcher.set_images(img1, img2)
    matcher.set_filter(kernel)
    matcher.configure_instance(passed_dmax = ndisp)


    tic = time.time()
    ms, disp = matcher.test_pipeline()
    toc=time.time()
    return turn_me_into_pfm(disp), toc-tic

def turn_me_into_pfm(disp):
    # no modification is necessary?
    # offline sdk handles 0 disps, but how about online?
    # highly inconsistent.

    return disp

def custom_benchmarking(EXP_PARAMS, disp, gt, occ, max_disp, ARE_OCCLUSIONS_ERRORS = False):
    gt_temp = np.where(gt==np.inf, 0, gt)
    gt_temp = gt_temp/gt_temp.max()*(max_disp)

    EXP_PARAMS["are_occlusions_errors"] = ARE_OCCLUSIONS_ERRORS
    occ = gt_temp if ARE_OCCLUSIONS_ERRORS else occ

    EXP_PARAMS["bad1"], EXP_PARAMS["bad2"], EXP_PARAMS["bad4"],\
    EXP_PARAMS["BAD8"], EXP_PARAMS["abs_error"], EXP_PARAMS["mse"],\
    EXP_PARAMS["avg"], EXP_PARAMS["eucledian"]  = \
        me.evaluate_over_all(disp, gt_temp, occ, occlusions_counted_in_errors=False)


if __name__ == "__main__":

    from components.utils.CSVWriter2 import Wrapper as csv
    from components.utils.Metrix import Wrapper as me

    ###################################################################
    # Getting passed arguments ########################################
    ###################################################################

    """path_to_alg_runfile, method_name, left, right, gt,
                         nonocc, outpath, kernel_width, kernel_height, match, gap, egap"""


    method_name = EXPERIMENT_TITLE = sys.argv[1]
    im1_path = sys.argv[2]
    im2_path = sys.argv[3]

    output_dir_path = sys.argv[4]
    kernel_width = int(sys.argv[5])
    kernel_height = int(sys.argv[6])
    MATCH = int(sys.argv[7])
    GAP = int(sys.argv[8])
    EGAP = int(sys.argv[9])
    part_of_test_set = True if sys.argv[10] == "True" else False
    #print(part_of_test_set)

    dir = os.path.dirname(im1_path)
    calib = os.path.join(dir, "calib.txt")

    with open(calib, "r") as f:
        readin = f.readlines()
        n_disp_line = readin[6]
        ndisp =  int(n_disp_line.split("=")[1])
        vmax_line = readin[9]
        max_disp = int(vmax_line.split("=")[1])
        print("\nmax disp: {0}".format(max_disp))
    ###################################################################
    # Loading images  #################################################
    ###################################################################


    img1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    img2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    if(img1 is None or img2 is None):
        raise Exception("The image files could not be loaded.")


    kernel = np.ones([kernel_height, kernel_width])

    ###################################################################
    # "Administration" ################################################
    ###################################################################

    runtime_file = "time"+method_name + ".txt"

    output_runtime_file = os.path.join(output_dir_path, runtime_file)

    if(not os.path.isdir(output_dir_path)):
        os.makedirs(output_dir_path)

    pfm_output_path = os.path.join(output_dir_path, 'disp0' + EXPERIMENT_TITLE + '.png')
    DATASET = "Middlebury_2014"

    EXP_PARAMS = {"experiment_id": EXPERIMENT_TITLE, "match": MATCH, "gap": GAP, "egap": EGAP, \
                  "algo": str(m.__module__), "init_method": "default", "dataset": DATASET, \
                  "preprocessing_method": "None", "kernel_size": 1, "kernel_spec": "None", \
                  "init_method": "maclean_et_al"}



    ###################################################################
    # Benchmark #######################################################
    ###################################################################


    print("Benchmarking is in progress.")

    result, runtime = benchmark_me(img1, img2, MATCH, GAP, EGAP, kernel, ndisp)

    EXP_PARAMS["runtime"] = runtime

    print("Benchmarking has finished.")

    disp = result

    ###################################################################
    # Saving results ##################################################
    ###################################################################

    cv2.imwrite(pfm_output_path, result*4)

    with open(output_runtime_file, "wb") as rf:
        rf.write(StrToBytes(str(runtime)))

    ###################################################################
    # Logging results ##################################################
    ###################################################################

    if (not part_of_test_set):
        gt_path = sys.argv[11]
        occl_path = sys.argv[12]
        img_paths = [im1_path, im2_path, gt_path, occl_path]

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        occ = cv2.imread(occl_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        CSV_FILEPATH = os.path.join("..", "custom_log", "all_benchmarking.csv")

        EXP_PARAMS["preprocessing_method"] = "None"
        EXP_PARAMS["scene"] = im1_path.split("\\")[-2]
        EXP_PARAMS["image_filename"] = pfm_output_path

        EXP_PARAMS["img_res"] = "{0}x{1}".format(img1.shape[1], img1.shape[0])
        EXP_PARAMS["kernel_size"] = "{0}x{1}".format(kernel_height, kernel_width)

        csv_logger = csv(CSV_FILEPATH, default_header=False)
        csv_logger.set_header_function(csv_logger.get_header_v3)
        csv_logger.write_csv_header()
        csv_logger.set_line_function(csv.format_stereo_matching_results_v2)
        custom_benchmarking(EXP_PARAMS, disp, gt, occ, max_disp,ARE_OCCLUSIONS_ERRORS=False)
        csv_logger.append_new_sm_results(EXP_PARAMS, selected_keys=csv.get_header_v3())

        for i in img_paths:
            print(i)

        print("bad4: {0}".format(EXP_PARAMS["bad4"]))

        custom_benchmarking(EXP_PARAMS, disp, gt, occ, max_disp, ARE_OCCLUSIONS_ERRORS=True)
        csv_logger.append_new_sm_results(EXP_PARAMS, selected_keys=csv.get_header_v3())

