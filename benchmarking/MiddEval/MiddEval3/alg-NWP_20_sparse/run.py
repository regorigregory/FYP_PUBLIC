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

def benchmark_me(img_path1, img_path2):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    if(img1 is None or img2 is None):
        raise Exception("The image files could not be loaded.")
    MATCH = 10
    GAP = -20
    EGAP = -1

    matcher = m(MATCH, GAP, EGAP, verbose=True)
    matcher.set_images(img1, img2)
    matcher.set_filter(np.ones((5,3)))
    matcher\
        .configure_instance()


    tic = time.time()
    ms, disp = matcher.test_pipeline()
    toc=time.time()
    return turn_me_into_pfm(disp), toc-tic

def turn_me_into_pfm(disp):

    d = disk(10)
    disp_mod = median(disp, d)
    #disp_mod = np.where(disp_mod==0.0, np.inf, disp_mod)

    #disp_with_inf = np.where(disp_mod==0.0, np.inf, disp_mod)
    return disp_mod

if __name__ == "__main__":
    method_name = sys.argv[1].split("-")[-1]
    im1_path = sys.argv[2]
    im2_path = sys.argv[3]
    output_dir_path = sys.argv[4]

    if(not os.path.isdir(output_dir_path)):
        os.makedirs(output_dir_path)
    print("Benchmarking is in progress.")
    result, runtime = benchmark_me(im1_path, im2_path)
    print("Benchmarking has finished.")

    runtime_file = method_name+".txt"
    output_runtime_file = os.path.join(output_dir_path, runtime_file)

    pfm_output_path = os.path.join(output_dir_path, 'disp0' + method_name + '_s.pfm')

    cv2.imwrite(pfm_output_path, result)

    with open(output_runtime_file, "wb") as rf:
        rf.write(StrToBytes(str(runtime)))