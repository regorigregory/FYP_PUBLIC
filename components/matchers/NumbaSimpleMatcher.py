from components.utils.SimpleTimer import SimpleTimer
from components.interfaces.NumbaWrapper import Interface as NumbaWrapperSkeleton
from components.numba_functions import common_functions as cf
import numpy as np

class Wrapper(NumbaWrapperSkeleton):
    def run_pipeline(self):
        self.test_pipeline()
    def test_pipeline(self):
        if (self._verbose):
            SimpleTimer.print_with_timestamp("Compilation and matching has started...")
        scores_raw, moves_raw = self._initialize_matrix_template(self._gap, self._egap, self._im1, rows_init_func= self._fill_up_first_rows_func)
        scores_n_moves = np.zeros((2, self._im1.shape[0], self._im1.shape[1] + 1, self._im1.shape[1] + 1), dtype=np.float64)
        disparity = np.zeros(self._im1.shape, dtype=np.float64)
        x, z = self._match_images(self._match, self._gap, self._egap, self._im1, self._im2, scores_raw, moves_raw, scores_n_moves, disparity,
                                  self._scanline_match_function, dmax = self._dmax)
        return x, z

if __name__ == "__main__":
    import numpy as np
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

    match = 35
    gap = -20
    egap = -1


    NumbaMatcherInstance = Wrapper(match, gap, egap, verbose=True)

    NumbaMatcherInstance.set_images(im1, im2)

    #NumbaMatcherInstance.set_images(loaded_imgs[1], loaded_imgs[0])
    NumbaMatcherInstance\
        .configure_instance(initialize_matrix_template= NumbaMatcherInstance.matrix_template_intit[1], match_scanlines = NumbaMatcherInstance.match_functions[1], dmax=128)
    #NumbaMatcherInstance.configure_instance()

    SimpleTimer.timeit()
    x,z = NumbaMatcherInstance.test_pipeline()
    #x,y,z = match_images(80, -30, -2, im2, im1)
    #x,y,z = FakeNumbaClass["match_images"] (100, -15, -5, im2, im1)
    SimpleTimer.timeit()

    BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = \
        m.evaluate_over_all(z * 4, gt, occ, occlusions_counted_in_errors=False)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(z, cmap="gray")
    plt.title("Bad4:{0}".format(BAD4))



