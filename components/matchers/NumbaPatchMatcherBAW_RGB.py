from components.utils.SimpleTimer import SimpleTimer
from components.interfaces.NumbaWrapper import Interface
from components.numba_functions import NPM_BAW_RGB_Functions as patch_functions
import numpy as np

class Wrapper(Interface):
    def set_filter(self, filter):
        self._filter = filter

    def configure_instance(self, passed_dmax=64, gamma_c=10, gamma_s =90, alpha=0):
        super().configure_instance(
            match_images=patch_functions.match_images,
            match_scanlines=patch_functions.match_scanlines_maclean,
            initialize_matrix_template=patch_functions.initialize_matrix_template_maclean,
            dmax=passed_dmax
        )
        self.gamma_c = gamma_c
        self.gamma_s = gamma_s
        self.alpha = alpha

    def configure_instance_for_optimisation(self):
        self.configure_instance()
        # super().configure_instance(match_scanlines = patch_functions.match_scanlines_param_search, match_images = patch_functions.match_images_param_search)

    def run_pipeline(self):
        self.test_pipeline()
    def set_images(self, im1, im2):
        #if (len(im1.shape) + len(im2.shape))!=6 or (im1.shape[2]+im2.shape[2])!=6:
        #    raise Exception("Please ensure that the images passed are in RGB and the channels are the last dimensions.")
        super().set_images(im1, im2)
    def test_pipeline(self):
        if (self._verbose):
            SimpleTimer.print_with_timestamp("Compilation and matching has started...")
        if (self._filter is None):
            self._filter = np.ones((3, 3, 3), dtype=np.int32)
        #if(len(self._filter.shape)!=3) or self._filter.shape[2]!=3:
        #    raise Exception("Please ensure that the filter passed is in RGB and the channels are the last dimension.")
        scores_raw, moves_raw = self._initialize_matrix_template(self._gap,
                                                                 self._egap,
                                                                 self._im1,
                                                                 rows_init_func=self._fill_up_first_rows_func)
        scores_n_moves = np.zeros((2, self._im1.shape[0],
                                   self._im1.shape[1] + 1,
                                   self._im1.shape[1] + 1),
                                   dtype=np.float64)

        disparity = np.zeros((self._im1.shape[0], self._im1.shape[1]), dtype=np.float64)
        x, z = self._match_images(self._match,
                                  self._gap,
                                  self._egap,
                                  self._im1,
                                  self._im2,
                                  scores_raw,
                                  moves_raw,
                                  scores_n_moves,
                                  disparity,
                                  self._scanline_match_function,
                                  filter=self._filter,
                                  gamma_c = self.gamma_c,
                                  gamma_s = self.gamma_s,
                                  alpha = self.alpha)
        return x, z


if __name__ == "__main__":
    import cv2
    import numpy as np
    import os
    import project_helpers

    im1_path = os.path.join(project_helpers.get_project_dir(), "datasets", "middlebury", "middlebury_2003", "cones", "im2.png")
    im2_path = os.path.join(project_helpers.get_project_dir(), "datasets", "middlebury", "middlebury_2003", "cones", "im6.png")
    left = cv2.imread(im1_path).astype(np.float64)
    right = cv2.imread(im2_path).astype(np.float64)

    match = 35
    gap = -20
    egap = -1

    NumbaMatcherInstance = Wrapper(match, gap, egap, verbose=True)
    NumbaMatcherInstance.set_images(left, right)
    NumbaMatcherInstance.configure_instance()
    NumbaMatcherInstance.set_filter(np.ones((7, 3, 3), dtype=np.int32))


    SimpleTimer.timeit()
    x, z = NumbaMatcherInstance.test_pipeline()


    SimpleTimer.timeit()
    import matplotlib.pyplot as plt

    plt.imshow(z, cmap="gray")