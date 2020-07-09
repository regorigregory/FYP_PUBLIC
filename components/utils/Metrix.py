import numpy as np
from components.utils import utils as u

#metrix need serious revision
class Wrapper:
#Where it is occluded, it should be removed from the calculation
    @staticmethod
    def add_occlusions(disp, gt, occ, occlusions_counted_in_errors):
        disp = disp.copy()
        gt = gt.copy()

        gt = np.where(occ == 0, 0, gt)
        if(not occlusions_counted_in_errors):
            disp = np.where(occ==0, 0, disp)
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
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)
        n = disp.size if occlusions_counted_in_errors else (disp.size-np.sum(occ==0))
        absolute_error = np.sum(np.abs(disp-gt))
        return np.nan_to_num(absolute_error/n)

    @staticmethod
    def abs_error(disp, gt, occ, occlusions_counted_in_errors = False):
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)
        n = disp.size if occlusions_counted_in_errors else (disp.size - np.sum(occ == 0))
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
    @staticmethod
    def rowwise_bad(disp, gt, occ, threshold=1, occlusions_counted_in_errors=True):
        disp, gt = Wrapper.add_occlusions(disp, gt, occ, occlusions_counted_in_errors)

        num_pixels_in_a_row = disp.shape[0]
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
        return above_threshold_count

if __name__ == "__main__":
    pass


