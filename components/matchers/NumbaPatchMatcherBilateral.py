from components.interfaces.NumbaWrapper import Interface
from components.numba_functions import NPM_BAW_Functions as patch_functions
from components.numba_functions import common_functions as cf
from components.utils.SimpleTimer import SimpleTimer
import numpy as np



class Wrapper(Interface):
    def set_filter(self, filter):
        self._filter = filter

    def configure_instance(self, passed_dmax=64, gamma_c=10, gamma_s=90, alpha=256, product_flag=True):
        super().configure_instance(
            match_images=patch_functions.match_images,
            match_scanlines=patch_functions.match_scanlines_maclean,
            initialize_matrix_template=cf.initialize_matrix_template_maclean,
            dmax=passed_dmax
        )
        self.gamma_c = gamma_c
        self.gamma_s = gamma_s
        self.alpha = alpha
        self.product_flag = product_flag

    def configure_instance_for_optimisation(self):
        self.configure_instance()

    def run_pipeline(self):
        self.test_pipeline()

    def test_pipeline(self):
        if (self._verbose):
            SimpleTimer.print_with_timestamp("Compilation and matching has started...")
        if (self._filter is None):
            self._filter = np.ones((3, 3), dtype=np.int32)
        scores_raw, moves_raw = self._initialize_matrix_template(self._gap, self._egap, self._im1,
                                                                 rows_init_func=self._fill_up_first_rows_func)
        scores_n_moves = np.zeros((2, self._im1.shape[0], self._im1.shape[1] + 1, self._im1.shape[1] + 1),
                                  dtype=np.float64)
        disparity = np.zeros(self._im1.shape, dtype=np.float64)
        x, z = self._match_images(self._match,
                                  self._gap,
                                  self._egap,
                                  self._im1,
                                  self._im2,
                                  scores_raw, moves_raw,
                                  scores_n_moves,
                                  disparity,
                                  self._scanline_match_function,
                                  filter=self._filter,
                                  gamma_c=self.gamma_c,
                                  gamma_s=self.gamma_s,
                                  alpha=self.alpha,
                                  product_flag = self.product_flag)
        return x, z


if __name__ == "__main__":
    import cv2
    import os
    from components.utils.Metrix import Wrapper as m

    scene = "cones"
    im1_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im2.png")
    im2_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im6.png")
    gt_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "disp2.png")
    occ_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "nonocc.png")

    im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    match = 30
    gap = -15
    egap = -1

    NumbaMatcherInstance = Wrapper(match, gap, egap, verbose=True)
    NumbaMatcherInstance.set_images(im1, im2)
    NumbaMatcherInstance.configure_instance(product_flag = False, alpha = 30)
    NumbaMatcherInstance.set_filter(np.ones((7, 3), dtype=np.float64))

    SimpleTimer.timeit()
    x, z = NumbaMatcherInstance.test_pipeline()
    SimpleTimer.timeit()

    BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = \
        m.evaluate_over_all(z * 4, gt, occ, occlusions_counted_in_errors=False)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(z, cmap="gray")
    plt.title("Bad4:{0}".format(BAD4))
