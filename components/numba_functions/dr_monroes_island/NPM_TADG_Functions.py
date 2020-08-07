import numpy as np
from numba import jit, prange

from components.numba_functions import common_functions as cf
from components.utils.SimpleTimer import SimpleTimer

disable_debug = True


@jit(nopython=disable_debug)
def get_patch_absolute_difference(currentIndex,
                                  im1_pixel_index,
                                  im2_pixel_index,
                                  im1_padded,
                                  im2_padded,
                                  filter=np.ones((3, 3), dtype=np.float64),
                                  gamma_c=10,
                                  gamma_s=10,
                                  cache_left=(),
                                  cache_right=(),
                                  gcl=(),
                                  gcr=(),
                                  alpha=0.9,
                                  T_c=0.028,
                                  T_s=0.008):
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
    patch1 = np.asarray(im1_padded[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64)
    patch2 = np.asarray(im2_padded[startRow:endRow, img2_start_column:img2_end_column], dtype=np.float64)

    if cache_left[key_left, 0, 0] == np.inf:
        p1_weights = cf.get_bilateral_suport_weights(patch1, gamma_c, gamma_s)
        cache_left[key_left] = p1_weights

    # if cache_right[key_right, 0, 0] == np.inf:
    #    p2_weights = get_bilateral_suport_weights(patch2, gamma_c, gamma_s)
    #    cache_right[key_right] = p2_weights

    left_window = cache_left[key_left]
    # right_window = cache_right[key_right]

    intensity_min_array = np.full_like(filter, T_c)

    absolute_difference = np.minimum(intensity_min_array, np.abs(patch1 - patch2))
    sum_of_absolute_difference = absolute_difference * (1 - alpha)

    # if(alpha>0):
    grad_left = gcl[startRow:endRow, img1_start_column:img1_end_column]
    grad_right = gcr[startRow:endRow, img2_start_column:img2_end_column]

    # hmmm
    grad_min_array = np.full_like(filter, T_s)
    sum_of_absolute_grad_difference = np.minimum(grad_min_array, np.abs(grad_left - grad_right))

    sum_of_absolute_grad_difference = sum_of_absolute_grad_difference * alpha
    lf_multiplier = left_window
    return np.sum(((sum_of_absolute_difference + sum_of_absolute_grad_difference) * lf_multiplier))


@jit(nopython=disable_debug)
def match_scanlines_maclean(match, gap, egap,
                            current_index,
                            im2_padded, im1_padded,
                            scores, moves,
                            filter=np.ones((3, 3), dtype=np.float64),
                            dmax=128,
                            gamma_c=10,
                            gamma_s=10,
                            gcl=(),
                            gcr=(),
                            alpha=0.0,
                            T_c=0.028,
                            T_s=0.008):
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

            match_raw = match - get_patch_absolute_difference(current_index, im1_index, im2_index,
                                                              im1_padded, im2_padded, filter,
                                                              gamma_c=gamma_c,
                                                              gamma_s=gamma_s,
                                                              cache_left=cache_left,
                                                              cache_right=cache_right,
                                                              gcl=gcl,
                                                              gcr=gcr,
                                                              alpha=alpha,
                                                              T_c=T_c,
                                                              T_s=T_s)
            # print(match_raw)

            east, north, ne = (i, j - 1), (i - 1, j), (i - 1, j - 1)
            east_raw, north_raw, ne_raw = scores[east], scores[north], scores[ne]
            east_dir, north_dir, ne_dir = moves[east], moves[north], moves[ne]

            east_raw += egap if east_dir == 2 else gap
            north_raw += egap if north_dir == 0 else gap
            match_raw += ne_raw

            all_scores = np.array([north_raw, match_raw, east_raw]).astype(np.float64)

            winner_index = np.argmax(all_scores)

            scores[i, j] = all_scores[winner_index]
            moves[i, j] = winner_index
            # """
    return scores, moves


@jit(nopython=disable_debug, parallel=False)
def match_images(match, gap, egap,
                 im1,
                 im2,
                 scores_raw,
                 moves_raw,
                 scores_n_moves,
                 disparity,
                 scanline_match_function,
                 filter=np.ones((3, 3), dtype=np.float64),
                 gamma_c=10,
                 gamma_s=10,
                 alpha=0.1,
                 T_c=0.028,
                 T_s=0.008):
    im1_padded = cf.pad_image_advanced(im1, filter.shape)
    im2_padded = cf.pad_image_advanced(im2, filter.shape)
    gcl, _ = cf.calculate_gradients_de_maetzu(im1_padded)
    gcr, _ = cf.calculate_gradients_de_maetzu(im2_padded)
    for i in prange(im1.shape[0]):
        # print(i)
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = \
            scanline_match_function(match, gap, egap,
                                    i, im1_padded, im2_padded,
                                    np.copy(scores_raw), np.copy(moves_raw), filter=filter,
                                    gamma_c=gamma_c,
                                    gamma_s=gamma_s,
                                    gcl=gcl,
                                    gcr=gcr,
                                    alpha=alpha,
                                    T_c=T_c,
                                    T_s=T_s)

        temp = cf.generate_disparity_line(im1.shape[1], cf.get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
        # print(i)
    return scores_n_moves, disparity


@jit(nopython=disable_debug)
def test_pipeline(match, gap, egap, im1, im2,
                  scanline_match_function=match_scanlines_maclean,
                  filter=np.ones((3, 3),
                                 dtype=np.float64),
                  gamma_c=130,
                  gamma_s=130,
                  alpha=0.9,
                  T_c=0.028,
                  T_s=0.008
                  ):
    scores_raw, moves_raw = cf.initialize_matrix_template_maclean(gap, egap, im1)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    disparity = np.zeros(im1.shape, dtype=np.float64)

    x, z = match_images(match, gap, egap, im1, im2, scores_raw,
                        moves_raw, scores_n_moves, disparity,
                        scanline_match_function, filter=filter,
                        gamma_c=gamma_c, gamma_s=gamma_s,
                        alpha=alpha,
                        T_c=T_c,
                        T_s=T_s)
    return x, z


if __name__ == "__main__":
    if __name__ == "__main__":
        import cv2
        import os
        from components.utils.Metrix import Wrapper as m
        from skimage.filters import median

        scaler = 256
        scene = "teddy"
        im1_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im2.png")
        im2_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im6.png")
        gt_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "disp2.png")
        occ_path = os.path.join("../..", "..", "datasets", "middlebury", "middlebury_2003", scene, "nonocc.png")

        im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / scaler
        im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / scaler
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        #match = 2.37439627e+01 / scaler  # 20.08/(256)
        #gap = -1.91974998e+01 / scaler  # -4.9/(256)
        #egap = -5.16750735e+00 / scaler

        match = 80 / scaler  # 20.08/(256)
        gap = -20 / scaler  # -4.9/(256)
        egap = -15 / scaler

        filter = np.ones((5, 7), dtype=np.float64)
        SimpleTimer.timeit()
        x, z = test_pipeline(match, gap, egap, im1, im2, filter=filter,
                             gamma_c=1.6e+00,
                             gamma_s=2.6e+00,
                             alpha=0.60,
                             T_c=0.6e-01,
                             T_s=3.5e-03)

        SimpleTimer.timeit()
        z_mod = z * 4
        # calc metrix!!!
        BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = \
            m.evaluate_over_all(z * 4, gt, occ, occlusions_counted_in_errors=False)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(z_mod, cmap="gray")
        plt.title("Bad4:{0}".format(BAD4))
