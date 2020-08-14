import abc
from components.numba_functions import NSM_Functions as default_functions
from components.numba_functions import common_functions as cf
import numpy as np
from components.utils.SimpleTimer import SimpleTimer


class Interface(abc.ABC):

    def __init__(self, match, gap, egap, verbose=False):
        self._match = match
        self._gap = gap
        self._egap = egap
        self._verbose = verbose
        self._fitness = 0
        self._filter = None
        if(verbose):
            SimpleTimer.print_with_timestamp("Numba Matcher isntance has been initialised with: {0}, {1}, {2} (m,g,egap)".format(self._match, self._gap, self._egap))

        self.matrix_init_modes = []
        self.matrix_init_modes.append(cf.fill_up_first_rows_default)
        self.matrix_init_modes.append(cf.fill_up_first_rows_v2)
        self.matrix_init_modes.append(cf.fill_up_first_rows_v3)

        self.matrix_template_intit = []
        self.matrix_template_intit.append(cf.initialize_matrix_template)
        self.matrix_template_intit.append(cf.initialize_matrix_template_maclean)

        self.match_functions = []
        self.match_functions.append(default_functions.match_scanlines)
        self.match_functions.append(default_functions.match_scanlines_maclean)

    def set_images(self, im1, im2):
        self._im2 = im2
        self._im1 = im1
        if(self._verbose):
            SimpleTimer.print_with_timestamp("Images have been set.")

    def configure_init_function(self, matrix_init_function):
        self._initialize_matrix_template = matrix_init_function

    def configure_instance(self,
                           match_images = default_functions.match_images,
                           match_scanlines = default_functions.match_scanlines_maclean,
                           initialize_matrix_template=cf.initialize_matrix_template_maclean,
                           fill_up_first_rows_func = cf.fill_up_first_rows_default,
                           dmax=256):

        self._dmax = dmax
        self._match_images =match_images
        self._scanline_match_function = match_scanlines
        self._initialize_matrix_template = initialize_matrix_template
        self._fill_up_first_rows_func = fill_up_first_rows_func
        if (self._verbose):
            SimpleTimer.print_with_timestamp("Instance has been configured.\n")

    @abc.abstractmethod
    def test_pipeline(self):
        pass

    def run_pipeline(self):
        self.test_pipeline()


