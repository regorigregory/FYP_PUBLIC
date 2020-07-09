from components.utils.SimpleTimer import SimpleTimer
from components.interfaces.NumbaWrapper import Interface as NumbaWrapperSkeleton
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
    from components.utils.SintelReader import Wrapper as SintelReader
    import matplotlib.pyplot as plt
    from components.utils import middlebury_utils as mbu
    from components.utils import plot_utils as plu
    import os

    path = os.path.join("..", "..", "datasets", "sintel", "training")
    reader = SintelReader(rootPath=path)
    reader.print_available_scenes()
    reader.set_selected_scene('alley_1')
    loaded_imgs = reader.get_selected_scene_next_files()
    match = 60
    gap = -20
    egap = -1

    ###################################################################
    # Middlebury Images################################################
    ###################################################################
    import sys

    #sys.path.append("../../")

    ROOT_PATH = os.path.join("..", "..")
    EXPERIMENT_TITLE = "EXP_000-Baseline_Maclean_Numba_Parallel_Q"

    INIT_METHOD = "original"
    DATASET = "middlebury"

    DATASET_FOLDER = os.path.join(ROOT_PATH, "datasets", DATASET)
    LOG_FOLDER = os.path.join(ROOT_PATH, "experiments", "logs")
    CSV_FILEPATH = os.path.join(LOG_FOLDER, EXPERIMENT_TITLE + ".csv")
    IMG_RES = "450X375"
    PREPROCESSING_METHOD = "None"
    KERNEL_SIZE = 1
    KERNEL_SPEC = "None"

    SCENES = ["teddy", "cones"]
    SIZE = ""
    YEAR = 2003
    EXP_PARAMS = dict()

    midd_loaded_imgs = list(mbu.get_images(DATASET_FOLDER, YEAR, scene) for scene in SCENES)

    for im, path in midd_loaded_imgs:
        filenames = list(os.path.split(p)[-1] for p in path)
        #plu.plot_images(im, filenames)


    NumbaMatcherInstance = Wrapper(match, gap, egap, verbose=True)

    NumbaMatcherInstance.set_images(loaded_imgs[0].astype(np.float64), loaded_imgs[1].astype(np.float64))

    #NumbaMatcherInstance.set_images(loaded_imgs[1], loaded_imgs[0])
    NumbaMatcherInstance\
        .configure_instance(initialize_matrix_template= NumbaMatcherInstance.matrix_template_intit[1], match_scanlines = NumbaMatcherInstance.match_functions[1], dmax=128)
    #NumbaMatcherInstance.configure_instance()

    SimpleTimer.timeit()
    x,z = NumbaMatcherInstance.test_pipeline()
    #x,y,z = match_images(80, -30, -2, im2, im1)
    #x,y,z = FakeNumbaClass["match_images"] (100, -15, -5, im2, im1)
    SimpleTimer.timeit()
    #z_mod = np.max(z)-z
    #import matplotlib.pyplot as plt
    #from matplotlib import cm
    plt.subplots(1,5)
    plt.subplot(151)
    plt.imshow(z)
    plt.subplot(152)
    plt.imshow(loaded_imgs[1])
    plt.subplot(153)

    plt.imshow(loaded_imgs[2])
    plt.subplot(154)

    plt.imshow(loaded_imgs[3])
    plt.subplot(155)

    plt.imshow(loaded_imgs[4])



