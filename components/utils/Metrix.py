import numpy as np
from components.utils import utils as u

#metrix need serious revision
class Wrapper:
#Where it is occluded, it should be removed from the calculation
    @staticmethod
    def add_occlusions(disp, gt, occ, occlusions_counted_in_errors):
        disp = disp.copy()
        gt = gt.copy()

        gt = np.where((occ == 0) | (occ == 128), 0, gt)
        if(not occlusions_counted_in_errors):
            disp = np.where((occ == 0) | (occ == 128), 0, disp)
        return disp, gt
    #this is the percentage of bad pixels whose difference from gt is greater than the threshold
    #it does not matter if it is 10 or 1000, the pixel will be counted as 1.
    @staticmethod
    def bad(disp, gt, occ, threshold = 1, occlusions_counted_in_errors = True):
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)

        num_pixels = disp.size
        num_occluded_pixels = np.sum(occ==0)

        abs_diff = np.abs(np.subtract(disp, gt))
        above_threshold_count = np.sum(abs_diff>threshold)
        divisor = num_pixels if occlusions_counted_in_errors else (num_pixels-num_occluded_pixels)
        result = above_threshold_count/divisor
        return result

    @staticmethod
    def mse(disp, gt, occ, occlusions_counted_in_errors = False):
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)
        n = disp.size if occlusions_counted_in_errors else (disp.size-np.sum(occ==0))
        return np.nan_to_num(np.sum(np.power(disp - gt, 2)) / n)

    @staticmethod
    def eucledian_distance(disp, gt, occ, occlusions_counted_in_errors = False):
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)
        return np.nan_to_num(np.sqrt(np.sum(np.power(disp-gt, 2))))

    @staticmethod
    def avg_err(disp, gt, occ, occlusions_counted_in_errors = False):
        # avg abs error
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)
        n = disp.size if occlusions_counted_in_errors else (disp.size-np.sum(occ==0))
        absolute_error = np.sum(np.abs(disp-gt))
        return np.nan_to_num(absolute_error/n)

    @staticmethod
    def abs_error(disp, gt, occ, occlusions_counted_in_errors = False):
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)
        absolute_error = np.sum(np.abs(disp - gt))
        return np.nan_to_num(absolute_error)
    @staticmethod
    def evaluate_over_all(disp, gt, occ, occlusions_counted_in_errors = False):
        BAD1 = Wrapper.bad(disp, gt, occ, threshold=1, occlusions_counted_in_errors = occlusions_counted_in_errors)
        BAD2 = Wrapper.bad(disp, gt, occ, threshold=2, occlusions_counted_in_errors = occlusions_counted_in_errors)
        BAD4 = Wrapper.bad(disp, gt, occ, threshold=4, occlusions_counted_in_errors = occlusions_counted_in_errors)
        BAD8 = Wrapper.bad(disp, gt, occ, threshold=8, occlusions_counted_in_errors = occlusions_counted_in_errors)
        ABS_ERROR = Wrapper.abs_error(disp, gt, occ, occlusions_counted_in_errors = occlusions_counted_in_errors)
        MSE = Wrapper.mse(disp, gt, occ, occlusions_counted_in_errors = occlusions_counted_in_errors)
        AVG = Wrapper.avg_err(disp, gt, occ, occlusions_counted_in_errors = occlusions_counted_in_errors)
        EUCLEDIAN = Wrapper.eucledian_distance(disp, gt, occ, occlusions_counted_in_errors = occlusions_counted_in_errors)

        return BAD1, BAD2, BAD4, BAD8, ABS_ERROR, MSE, AVG, EUCLEDIAN
    @staticmethod
    def precision(disp, gt):
        disp_matches= disp[disp==gt]
        HIT = np.sum(disp_matches==0)
        #TP/(TP+FP)
        return HIT/np.sum(disp==0)
    @staticmethod
    def recall(disp, gt):
        all_matches = disp == gt
        disp_matches = disp[all_matches]
        HIT = np.sum(disp_matches == 0)
        # TP/(TP+FN)
        return HIT / np.sum(gt==0)

    @staticmethod
    def f1(disp, gt, occl=True):
        PR = Wrapper.precision(disp, gt)
        RC = Wrapper.recall(disp, gt)
        return 2*(PR*RC)/(PR+RC)

    # it should return an array of errors...this sentence is also a pun at this stage, as it does not work...

    @staticmethod
    def rowwise_bad(disp, gt, occ, threshold=1, occlusions_counted_in_errors=True):
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)

        num_pixels_in_a_row = disp.shape[1]
        num_occluded_pixels = np.sum(occ == 0, axis=1)

        abs_diff = np.abs(np.subtract(disp, gt))
        above_threshold_count = np.sum(abs_diff > threshold, axis=1)
        # so far so good

        divisor = num_pixels_in_a_row if occlusions_counted_in_errors else (num_pixels_in_a_row - num_occluded_pixels)
        result = above_threshold_count / divisor
        return result

    @staticmethod
    def rowwise_abs(disp, gt, occ, threshold=1, occlusions_counted_in_errors=True):
        #print("The threshold value is not used here. Stop changing it! It has no effect. Hee-hee.")
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)

        num_pixels_in_a_row = disp.shape[0]
        num_occluded_pixels = np.sum(occ == 0, axis=1)

        abs_diff = np.abs(np.subtract(disp, gt))
        above_threshold_count = np.sum(abs_diff, axis=1)
        divisor = num_pixels_in_a_row if occlusions_counted_in_errors else (num_pixels_in_a_row - num_occluded_pixels)
        return above_threshold_count/divisor

if __name__ == "__main__":
    from components.utils import plotly_helpers as plh
    from components.utils import middlebury_utils as mbu
    import pandas as pd
    import numpy as np
    import os
    import cv2
    import project_helpers


    ROOT_PATH = project_helpers.get_project_dir()

    csv_path = os.path.join(ROOT_PATH, "experiments", 'logs', 'ALG_004_EXP_002-Baseline-MacLean_et_al-Numba_param_search.csv')

    DATASET = "middlebury"

    DATASET_FOLDER = os.path.join(ROOT_PATH, "datasets", DATASET)
    SCENES = ["teddy", "cones"]
    SIZE = ""
    YEAR = 2003

    loaded_imgs_and_paths = list(mbu.get_images(DATASET_FOLDER, YEAR, scene) for scene in SCENES)
    df = plh.load_n_clean(csv_path, gts=False, kernel_sizes=False)
    teddy = df[df["scene"] == "teddy"]
    teddy_filtered = teddy[teddy["are_occlusions_errors"] == True].sort_values(by="match")
    teddy_gt, teddy_occl_map = loaded_imgs_and_paths[0][0][2], loaded_imgs_and_paths[0][0][3]

    teddy_row_wise_errors = np.zeros([60, 375])
    teddy_images = [cv2.imread(os.path.join(ROOT_PATH, disp_path), cv2.IMREAD_GRAYSCALE)
                                     for disp_path in
                                     teddy_filtered["image_filename"].values]

    for index, img in enumerate(teddy_images):
        match = index
        disp = img
        result = Wrapper.rowwise_bad(disp/4, teddy_gt/4, teddy_occl_map, threshold=1, occlusions_counted_in_errors=True)
        test = Wrapper.bad(disp/4, teddy_gt/4, teddy_occl_map, threshold=1, occlusions_counted_in_errors=True)
        print(np.sum(result)/375-test)

        teddy_row_wise_errors[match] = np.array(result)

    min_vals = np.min(teddy_row_wise_errors, axis=0)
    min_indices = np.argmin(teddy_row_wise_errors, axis=0)
    print(np.sum(min_vals)/375)
    recompiled_image = np.zeros([375, 450])
    for row, match in enumerate(min_indices):
        recompiled_image[row] = teddy_images[match][row]

    import matplotlib.pyplot as plt

    plt.imshow(recompiled_image)




