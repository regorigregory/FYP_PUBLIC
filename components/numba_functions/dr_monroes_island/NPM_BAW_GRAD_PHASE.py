import numpy as np
from numba import jit, prange

from components.numba_functions import common_functions as cf
from components.utils.SimpleTimer import SimpleTimer

disable_debug = True


@jit(nopython=disable_debug)
def get_patch_absolute_difference(currentIndex,
                                  im1_pixel_index, im2_pixel_index,
                                  im1_padded, im2_padded,
                                  filter=np.ones((3, 3), dtype=np.float64),
                                  gamma_c=10, gamma_s=10,
                                  cache_left=(), cache_right=(),
                                  gcl=(),
                                  gcr=(),
                                  phase_left=(),
                                  phase_right=(),
                                  alpha=0.0,
                                  product_flag=True,
                                  T_c=256):
    # here, you have to calculate the patches indices without the default value
    filter_y = filter.shape[0]
    filter_x = filter.shape[1]
    startRow = currentIndex
    endRow = int(currentIndex + (filter_y))  # slicing is not inclusive

    # what if you are at the edges? you need to handle it
    img1_start_column = int(im1_pixel_index)
    img1_end_column = int(im1_pixel_index + filter_x)
    img2_start_column = int(im2_pixel_index)
    img2_end_column = int(im2_pixel_index + filter_x)

    # bilateral weighting

    key_left = im1_pixel_index
    key_right = im2_pixel_index

    if cache_left[key_left, 0, 0] == np.inf:
        # grad1 = np.asarray(gcl[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64) * filter
        patch1 = np.asarray(im1_padded[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64) #* filter

        p1_weights = cf.get_bilateral_suport_weights(patch1, gamma_c,
                                                     gamma_s) if product_flag else cf.get_bilateral_suport_weights_sum(
            patch1, gamma_c, gamma_s)
        cache_left[key_left] = p1_weights  # * grad2

    if cache_right[key_right, 0, 0] == np.inf:
        # grad2 = np.asarray(gcr[startRow:endRow, img2_start_column:img2_end_column], dtype=np.float64) * filter
        patch2 = np.asarray(im2_padded[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64) #* filter

        p2_weights = cf.get_bilateral_suport_weights(patch2, gamma_c,
                                                     gamma_s) if product_flag else cf.get_bilateral_suport_weights_sum(
            patch2, gamma_c, gamma_s)
        cache_right[key_right] = p2_weights  # * grad2

    grad1 = np.asarray(gcl[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64) #* filter
    grad2 = np.asarray(gcr[startRow:endRow, img2_start_column:img2_end_column], dtype=np.float64) #* filter
    phase1 = np.asarray(phase_left[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64) #* filter
    phase2 = np.asarray(phase_right[startRow:endRow, img2_start_column:img2_end_column], dtype=np.float64) #* filter

    grad_abs_difference = np.abs(grad1 - grad2)
    phase_abs_difference = np.abs(phase1 - phase2)



    balanced_phase_and_grad_component = alpha * grad_abs_difference + (1-alpha) *phase_abs_difference
    # absolute_difference = np.abs(left_window-right_window)

    left_window = cache_left[key_left]
    right_window = cache_right[key_right]

    common_term = left_window * right_window
    weighted_phase_and_grad_component = np.sum(balanced_phase_and_grad_component * common_term)/np.sum(common_term)

    return min(weighted_phase_and_grad_component, T_c)
    """
    return 0  # """


@jit(nopython=disable_debug)
def match_scanlines_maclean(match,
                            gap,
                            egap,
                            current_index,
                            im2_padded,
                            im1_padded,
                            scores,
                            moves,
                            filter=np.ones((3, 3), dtype=np.float64),
                            dmax=256,
                            gamma_c=10,
                            gamma_s=10,
                            gcl=(),
                            gcr=(),
                            phase_left=(),
                            phase_right=(),
                            alpha=0.0,
                            product_flag=True,
                            T_c=256):
    start_x = int((filter.shape[1] - 1) / 2)
    start_y = int((filter.shape[0] - 1) / 2)

    im1_scanline, im2_scanline = im1_padded[int(current_index + start_y)], im2_padded[int(current_index + start_y)]

    cache_left = np.full((scores.shape[0], filter.shape[0], filter.shape[1]), np.inf, dtype=np.float64)
    cache_right = np.full((scores.shape[0], filter.shape[0], filter.shape[1]), np.inf, dtype=np.float64)

    for i in range(1, im1_scanline.shape[0] - start_x * 2 + 1):

        starting_index = 1 if i <= dmax + 1 else i - dmax

        # Is this correct? doesn't seem to be right, Mr. Maclean
        # It is! You were not right Mr. Gergo! Maclean:Gergo : 1:0

        for j in range(starting_index, i + 1):
            im1_index, im2_index = j - 1, i - 1

            # match_raw = match - abs(im1_scanline[im1_index] - im2_scanline[im2_index])
            # match_raw =  match - abs(im1_scanline[im1_index] - im2_scanline[im2_index]) #get_patch_absolute_difference(current_index, im1_index, im2_index,im1, im2)
            match_raw = match - get_patch_absolute_difference(current_index,
                                                              im1_index,
                                                              im2_index,
                                                              im1_padded,
                                                              im2_padded,
                                                              filter,
                                                              gamma_c=gamma_c,
                                                              gamma_s=gamma_s,
                                                              cache_left=cache_left,
                                                              cache_right=cache_right,
                                                              gcl=gcl,
                                                              gcr=gcr,
                                                              phase_left=phase_left,
                                                              phase_right=phase_right,
                                                              alpha=alpha,
                                                              product_flag=product_flag,
                                                              T_c=T_c)
            east, north, ne = (i, j - 1), (i - 1, j), (i - 1, j - 1)
            east_raw, north_raw, ne_raw = scores[east], scores[north], scores[ne]
            east_dir, north_dir, ne_dir = moves[east], moves[north], moves[ne]

            east_raw += egap if east_dir == 2 else gap
            north_raw += egap if north_dir == 0 else gap
            match_raw += ne_raw

            all_scores = np.array([north_raw, match_raw, east_raw])

            winner_index = np.argmax(all_scores)

            scores[i, j] = all_scores[winner_index]
            moves[i, j] = winner_index

    return scores, moves


@jit(nopython=disable_debug, parallel=False)
def match_images(match,
                 gap,
                 egap,
                 im1,
                 im2,
                 scores_raw,
                 moves_raw,
                 scores_n_moves,
                 disparity,
                 scanline_match_function,
                 filter=np.ones((5, 5), dtype=np.float64),
                 gamma_c=10,
                 gamma_s=10,
                 alpha=0.0,
                 product_flag=True,
                 T_c=256):
    im1_padded = cf.pad_image_advanced(im1, filter.shape)
    im2_padded = cf.pad_image_advanced(im2, filter.shape)

    gcl, phase_left = cf.calculate_gradients_de_maetzu(im1_padded)
    gcr, phase_right = cf.calculate_gradients_de_maetzu(im2_padded)
    squared_255 = 255**2
    max_possible_gradient = np.sqrt(2*squared_255)

    gcl = (gcl / max_possible_gradient) * 256

    # normalised by left image's max

    gcr = (gcr / max_possible_gradient) * 256

    # max arctan value ....
    # tbf, a bit greater

    max_tangent = 1.5707963267949

    # ensuring that they have only positive values
    # perhaps not the best idea....

    phase_left = phase_left + max_tangent
    phase_right = phase_right + max_tangent

    # normalised by max_tangent

    phase_left = (phase_left / (2 * max_tangent)) * 256
    phase_right = (phase_right / (2 * max_tangent)) * 256
    for i in prange(im1.shape[0]):
        # print(i)
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = \
            scanline_match_function(match,
                                    gap,
                                    egap,
                                    i,
                                    im1_padded,
                                    im2_padded,
                                    np.copy(scores_raw),
                                    np.copy(moves_raw),
                                    filter=filter,
                                    gamma_c=gamma_c,
                                    gamma_s=gamma_s,
                                    gcl=gcl,
                                    gcr=gcr,
                                    phase_left=phase_left,
                                    phase_right=phase_right,
                                    alpha=alpha,
                                    product_flag=product_flag,
                                    T_c=T_c)

        temp = cf.generate_disparity_line(im1.shape[1], cf.get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
        # print(i)
    return scores_n_moves, disparity


@jit(nopython=disable_debug)
def test_pipeline(match, gap, egap, im1, im2,
                  scanline_match_function=match_scanlines_maclean,
                  filter=np.ones((3, 3),
                                 dtype=np.float64),
                  gamma_c=10,
                  gamma_s=10,
                  alpha=1,
                  beta=1,
                  product_flag=True,
                  T_c=256):
    scores_raw, moves_raw = cf.initialize_matrix_template_maclean(gap, egap, im1)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    disparity = np.zeros(im1.shape, dtype=np.float64)

    x, z = match_images(match,
                        gap,
                        egap,
                        im1,
                        im2,
                        scores_raw,
                        moves_raw,
                        scores_n_moves,
                        disparity,
                        scanline_match_function,
                        filter=filter,
                        gamma_c=gamma_c,
                        gamma_s=gamma_s,
                        alpha=alpha,
                        product_flag=product_flag,
                        T_c=T_c)
    return x, z


if __name__ == "__main__":
    import cv2
    from components.utils.Metrix import Wrapper as m
    import os
    import cv2
    import os

    scene = "teddy"
    im1_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im2.png")
    im2_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im6.png")
    gt_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "disp2.png")
    occ_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "nonocc.png")

    im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    # im1 = np.random.randint(0, 255, [2,4])
    # im2 = np.random.randint(0, 255, [2, 4])
    match = 100
    gap = -5
    egap = -1
    scores_raw, moves_raw = cf.initialize_matrix_template(gap, egap, im1)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    disparity = np.zeros(im1.shape, dtype=np.float64)
    # test_1, test_2 = match_scanlines(match, gap, egap, 0, im2, im1, scores_raw, moves_raw)

    SimpleTimer.timeit()
    x, z = test_pipeline(match, gap, egap, im1, im2,
                         filter=np.ones((3, 3),
                         dtype=np.float64),
                         gamma_c=5,
                         gamma_s=1,
                         alpha=0.2,
                         product_flag=False,
                         T_c = 1000)
    # x,y,z = match_images(80, -30, -2, im2, im1)
    # x,y,z = FakeNumbaClass["match_images"] (100, -15, -5, im2, im1)
    # Wrapper.match_images( im2, im1)

    SimpleTimer.timeit()
    z_mod = z

    BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = \
        m.evaluate_over_all(z * 4, gt, occ, occlusions_counted_in_errors=False)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(z_mod, cmap="gray")
    plt.title("Bad4:{0}".format(BAD4))
