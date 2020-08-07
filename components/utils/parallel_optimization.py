# built in modules and libraries

import numpy as np
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from components.utils.SintelReader import Wrapper as SintelReader
from components.utils.Metrix import Wrapper as Metrix
from components.matchers.NumbaSimpleMatcher import Wrapper as Matcher

# Custom libraries/classes

# Custom libraries/utils
# map(func, params)
from components.utils import plot_utils as plu
from components.utils import utils as u

# selected_imgs_paths: input images, 23
# all_disparities_loaded: 23 disparities having been loaded
# spec_dict: dictionary used to store csv file column values
# log: CSV Writer instance
# best_params: loaded best match, gap and egap values
# steps: the number of iterations, used by the progress bar
# progress_bar: progress bar instance that outputs the progress...

def parallel_exec(selected_imgs_paths, all_disparities_loaded, spec_dict, log, best_params, steps, progress_bar):

    progress_counter=0

    def img_pair_loop(convolved_img_pair_path, disp):
        nonlocal spec_dict
        nonlocal best_params
        nonlocal steps
        nonlocal progress_counter

        spec_dict_l1 = spec_dict.copy()
        spec_dict_l1["img_name"] = convolved_img_pair_path

        one_picklee = SintelReader.load_pickled_convolved(convolved_img_pair_path)

        gt = SintelReader.add_masks_to_raw_disparity(disp[0], disp[1], disp[2])

        one_picklee_with_all_filters = np.swapaxes(one_picklee, 0, 3)
        one_picklee_with_all_filters = np.swapaxes(one_picklee_with_all_filters, 1, 3)
        filter_indices = [i for i in range(one_picklee_with_all_filters.shape[1])]

        def filter_loop(one_picklee, filter_index):
            nonlocal spec_dict_l1
            nonlocal progress_bar
            nonlocal steps
            nonlocal progress_counter
            nonlocal log
            nonlocal best_params
            nonlocal gt
            scope_spec_dict_l2 = spec_dict_l1.copy()
            scope_spec_dict_l2["convolution_filters"] = scope_spec_dict_l2["img_name_pretag"] + str(
                filter_index)

            leftee = one_picklee[0, :, :]
            rightee = one_picklee[1, :, :]

            def best_params_loop(current_best_params):
                nonlocal scope_spec_dict_l2
                nonlocal progress_bar
                nonlocal steps
                nonlocal progress_counter
                nonlocal log
                nonlocal gt
                nonlocal leftee
                nonlocal rightee

                scope_spec_dict_l3 = scope_spec_dict_l2.copy()

                match = scope_spec_dict_l3["match"] = current_best_params[0]
                gap = scope_spec_dict_l3["gap"] = current_best_params[1]
                egap = scope_spec_dict_l3["egap"] = current_best_params[2]

                matcher = Matcher(match, gap, egap)

                matcher.set_images(rightee, leftee)
                matcher.configure_instance()
                tic = time.time()

                x, raw_disp_map = matcher.run_pipeline()

                toc = time.time()
                scope_spec_dict_l3["runtime"] = toc - tic
                scope_spec_dict_l3["filter_strategy"] = "VGG16_one_by_one"
                scope_spec_dict_l3["mse"] = Metrix.mse(raw_disp_map, gt)
                scope_spec_dict_l3["euclid_distance"] = Metrix.eucledian_distance(raw_disp_map, gt)
                scope_spec_dict_l3["avg_err"] = Metrix.avgerr(raw_disp_map, gt, print=False)

                progress_bar.progress_bar(progress_counter, steps, header="Parameter search progress: ",
                                          progress_bar_steps=100)
                progress_counter += 1
                log.append_new_sm_results(scope_spec_dict_l3)

            with ThreadPoolExecutor(max_workers=1) as exec:
                results = exec.map(best_params_loop, best_params)
                return results

        with ThreadPoolExecutor(max_workers=1) as exec:
            results = exec.map(filter_loop, one_picklee_with_all_filters, filter_indices)
            return results

#todo AttributeError: Can't pickle local object 'parallel_exec.<locals>.img_pair_loop'
    with ProcessPoolExecutor(max_workers = 1) as exec:
        results = exec.map(img_pair_loop, selected_imgs_paths, all_disparities_loaded)
        return results
