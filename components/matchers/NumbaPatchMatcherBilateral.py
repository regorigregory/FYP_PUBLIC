from components.utils.SimpleTimer import SimpleTimer
from components.interfaces.NumbaWrapper import Interface
from components.numba_functions import NPM_BAW_Functions as patch_functions

import numpy as np

class Wrapper(Interface):
    def set_filter(self, filter):
        self._filter = filter

    def configure_instance(self, passed_dmax=64, gamma_c=10, gamma_s =90, alpha=0.1):
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
    left = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    right = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    from components.utils.SintelReader import Wrapper as SintelReader

    path = os.path.join(project_helpers.get_project_dir(), "datasets", "sintel", "training")
    reader = SintelReader(rootPath=path)
    reader.print_available_scenes()
    reader.set_selected_scene('cave_2')

    # left, right, disp, occ, outoff = reader.get_selected_scene_next_files()
    match = 60
    gap = -20
    egap = -1

    NumbaMatcherInstance = Wrapper(match, gap, egap, verbose=True)
    NumbaMatcherInstance.set_images(left, right)
    NumbaMatcherInstance.configure_instance()
    NumbaMatcherInstance.set_filter(np.ones((7, 1), dtype=np.int32))
    # NumbaMatcherInstance.configure_instance_for_optimisation()
    # print(NumbaMatcherInstance._scanline_match_function)

    SimpleTimer.timeit()
    x, z = NumbaMatcherInstance.test_pipeline()
    # x,y,z = match_images(80, -30, -2, im2, im1)
    # x,y,z = FakeNumbaClass["match_images"] (100, -15, -5, im2, im1)

    SimpleTimer.timeit()
    import matplotlib.pyplot as plt

    plt.imshow(z, cmap="gray")