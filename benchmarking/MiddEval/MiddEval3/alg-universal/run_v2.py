import time
import numpy as np
import sys
import cv2
import os
from skimage.filters import median
from skimage.morphology import disk
from components.matchers.NumbaPatchMatcherBilateral import Wrapper as m
from components.utils import SimpleConvolution as SC
from components.utils import utils as u


def StrToBytes(text):
    if sys.version_info[0] == 2:
        return text
    else:
        return bytes(text, 'UTF-8')

def benchmark_me(img1, img2, MATCH, GAP, EGAP, kernel, ndisp, gamma_c=10, gamma_s =90, alpha=0):

    matcher = m(MATCH, GAP, EGAP, verbose=True)
    matcher.set_images(img1, img2)
    matcher.set_filter(kernel)
    matcher.configure_instance(passed_dmax = ndisp, gamma_c=gamma_c, gamma_s =gamma_s, alpha=alpha, product_flag=False)

    tic = time.time()
    ms, disp = matcher.test_pipeline()
    toc=time.time()

    return turn_me_into_pfm(disp), disp, toc-tic

def turn_me_into_pfm(disp):
    # no modification is necessary?
    # offline sdk handles 0 disps, but how about online?
    # highly inconsistent.
    disp_mod = np.where(disp==0, np.inf, disp)
    #return disp
    return disp_mod

def log_results(EXP_PARAMS, disp, gt, occ, max_disp, ARE_OCCLUSIONS_ERRORS = False):
    gt_noninf = np.where(gt==np.inf, 0, gt)

    EXP_PARAMS["are_occlusions_errors"] = ARE_OCCLUSIONS_ERRORS
    occ = gt_noninf if ARE_OCCLUSIONS_ERRORS else occ

    EXP_PARAMS["bad1"], EXP_PARAMS["bad2"], EXP_PARAMS["bad4"],\
    EXP_PARAMS["BAD8"], EXP_PARAMS["abs_error"], EXP_PARAMS["mse"],\
    EXP_PARAMS["avg"], EXP_PARAMS["eucledian"]  = \
        me.evaluate_over_all(disp, gt_noninf, occ, occlusions_counted_in_errors=False)

def get_preprocessing_options():
    options = dict(naive_median=naive_median, naive_vertical=naive_vertical,
                   naive_horizontal=naive_horizontal, naive_typo=naive_typo,
                   naive_all=naive_all)
    return options

def naive_median(im2, im6, convolver):
    convolver.filter = np.ones([3,3])
    im2_mod= convolver.convolve(im2)
    im6_mod = convolver.convolve(im6)
    return im2_mod, im6_mod

def naive_horizontal(im2, im6, convolver):
    u.getHorizontalFeatureFilter(convolver)
    im2_mod = convolver.convolve(im2)
    im6_mod = convolver.convolve(im6)
    return im2_mod, im6_mod

def naive_vertical(im2, im6, convolver):
    u.getVerticalFeatureFilter(convolver)
    im2_mod = convolver.convolve(im2)
    im6_mod = convolver.convolve(im6)
    return im2_mod, im6_mod


def naive_typo(im2, im6, convolver):
    u.getFilterByTypo(convolver)
    im2_mod = convolver.convolve(im2)
    im6_mod = convolver.convolve(im6)
    return im2_mod, im6_mod


def naive_all(im2, im6, convolver):
    options = get_preprocessing_options()

    im2_mod = np.zeros(im2.shape)
    im6_mod = np.zeros(im2.shape)

    for k,v in options.items():
        if(k=="naive_all"):
            continue
        temp = v(im2, im6, convolver)
        im2_mod+=temp[0]
        im6_mod+=temp[1]
    return im2_mod/4, im6_mod/4



def please_preprocess_me(im2, im6, preprocessing_request):
    print("I am just about to be pre-processed!!!")
    options = get_preprocessing_options()

    convolver = SC.getOne()

    im2_mod, im6_mod = options[preprocessing_request](im2, im6, convolver)
    print("I have been pre-processed!!!")

    return im2_mod, im6_mod

if __name__ == "__main__":

    from components.utils.CSVWriter2 import Wrapper as csv
    from components.utils.Metrix import Wrapper as me
    import project_helpers
    ###################################################################
    # Getting passed arguments ########################################
    ###################################################################

    #for a in sys.argv:
    #    print(a)
    benchmarking_root =os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    EXPERIMENT_TITLE = sys.argv[1]
    im1_path = os.path.join(benchmarking_root, sys.argv[2])
    im2_path = os.path.join(benchmarking_root, sys.argv[3])

    output_dir_path = os.path.join(benchmarking_root, sys.argv[4])
    kernel_width = int(sys.argv[5])
    kernel_height = int(sys.argv[6])

    MATCH = int(sys.argv[7])
    GAP = int(sys.argv[8])
    EGAP = int(sys.argv[9])
    part_of_test_set = True if sys.argv[10] == "True" else False

    preprocessing_request = False

    gamma_c = 0 if len(sys.argv) < 15 else int(sys.argv[14])
    gamma_s= 0 if len(sys.argv) < 16 else int(sys.argv[15])
    alpha = 0 if len(sys.argv) < 17 else int(sys.argv[16])

    ###################################################################
    # Loading + pre-processing images  #################################
    ###################################################################

    kernel = np.ones([kernel_height, kernel_width])

    img1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    img2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    if (img1 is None or img2 is None):
        raise Exception("The image files could not be loaded.")
    EXPERIMENT_TITLE+="mge_{3}_{4}_{5}_gc_{0}_gs_{1}_a_{2}".format(gamma_c, gamma_s, alpha, MATCH, GAP, EGAP)
    if (len(sys.argv) > 13):
        temp = sys.argv[13]
        if(temp in get_preprocessing_options().keys()):
            EXPERIMENT_TITLE += "_pre_"+temp
            preprocessing_request=temp
            img1_mod, img2_mod = please_preprocess_me(img1, img2, preprocessing_request)
            img1 += img1_mod
            img2 += img2_mod
            img1 /= 2
            img2 /= 2
        else:
            print("I won't be pre-processed.")

    dir = os.path.dirname(im1_path)
    calib = os.path.join(dir, "calib.txt")

    ###################################################################
    # Loading calibration file ########################################
    ###################################################################

    with open(calib, "r") as f:
        readin = f.readlines()
        n_disp_line = readin[6]
        ndisp =  int(n_disp_line.split("=")[1])
        vmax_line = readin[9]
        max_disp = int(vmax_line.split("=")[1])
        print("\nndisp: {0}".format(ndisp))


    ###################################################################
    # "Administration" ################################################
    ###################################################################

    runtime_file = "time"+EXPERIMENT_TITLE + ".txt"

    output_runtime_file = os.path.join(output_dir_path, runtime_file)

    if(not os.path.isdir(output_dir_path)):
        os.makedirs(output_dir_path)

    pfm_output_path = os.path.join(output_dir_path, 'disp0' + EXPERIMENT_TITLE + '.pfm')
    png_output_path = os.path.join(output_dir_path, 'disp0' + EXPERIMENT_TITLE + '.png')

    DATASET = "Middlebury_2014"

    EXP_PARAMS = {"experiment_id": EXPERIMENT_TITLE, "match": MATCH, "gap": GAP, "egap": EGAP,
                  "algo" : str(m.__module__), "init_method" : "default", "dataset": DATASET,
                  "preprocessing_method": "None", "kernel_size": 1, "kernel_spec": "None",
                  "init_method": "maclean_et_al"}



    ###################################################################
    # Benchmarking ####################################################
    ###################################################################


    print("Benchmarking is in progress.")

    result_inf, result, runtime = benchmark_me(img1, img2, MATCH, GAP, EGAP, kernel, ndisp,
                                   gamma_c = gamma_c, gamma_s=gamma_s, alpha=alpha)

    EXP_PARAMS["runtime"] = runtime

    print("Benchmarking has finished.")

    disp = result

    ###################################################################
    # Saving results *4 ###############################################
    ###################################################################

    cv2.imwrite(png_output_path, result*4)
    cv2.imwrite(pfm_output_path, result_inf)
    with open(output_runtime_file, "wb") as rf:
        rf.write(StrToBytes(str(runtime)))

    ###################################################################
    # Logging & benchmarking results ##################################
    ###################################################################

    if (not part_of_test_set):
        gt_path = sys.argv[11]
        occl_path = sys.argv[12]

        img_paths = [im1_path, im2_path, gt_path, occl_path]
        print(img_paths[0])

        gt = u.load_pfm(gt_path)[0].astype(np.float64)
        occ = cv2.imread(occl_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        CSV_FILEPATH = os.path.join("..", "custom_log", "trunc_plusblg.csv")

        EXP_PARAMS["preprocessing_method"] = "None"
        EXP_PARAMS["scene"] = im1_path.split("\\")[-2]



        logged_path = os.path.relpath(png_output_path, project_helpers.get_project_dir()).replace("..\\", "").replace("../", "")

        EXP_PARAMS["image_filename"] = logged_path

        EXP_PARAMS["img_res"] = "{0}x{1}".format(img1.shape[1], img1.shape[0])
        EXP_PARAMS["kernel_size"] = "{0}x{1}".format(kernel_height, kernel_width)
        EXP_PARAMS["kernel_spec"] = "blg_gs_{0}_gc_{1}_alpha_{2}"\
            .format(gamma_s, gamma_c, alpha)

        csv_logger = csv(CSV_FILEPATH, default_header=False)
        csv_logger.set_header_function(csv_logger.get_header_v3)
        csv_logger.write_csv_header()
        csv_logger.set_line_function(csv.format_stereo_matching_results_v2)
        log_results(EXP_PARAMS, disp * 4, gt * 4, occ, max_disp, ARE_OCCLUSIONS_ERRORS=False)
        csv_logger.append_new_sm_results(EXP_PARAMS, selected_keys=csv.get_header_v3())

        print("bad4: {0}".format(EXP_PARAMS["bad4"]))

