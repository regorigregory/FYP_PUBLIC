import numpy as np
from numba import jit, prange

from components.numba_functions import common_functions as cf
from components.utils.SimpleTimer import SimpleTimer

disable_debug = True


@jit(nopython=disable_debug)
def get_patch_absolute_difference(currentIndex,
                                  im1_pixel_index, im2_pixel_index,
                                  im1_padded, im2_padded,
                                  filter=np.ones((3, 3, 3), dtype=np.float64),
                                  gamma_c=10, gamma_s=10,
                                  cache_left=(),
                                  cache_right=(),
                                  gcl=(),
                                  gcr=(),
                                  alpha=0.0):
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
    patch1 = np.asarray(im1_padded[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64) * filter
    patch2 = np.asarray(im2_padded[startRow:endRow, img2_start_column:img2_end_column], dtype=np.float64) * filter

    absolute_difference = np.abs((patch1 - patch2))
    ##element-wise maximum difference along the three channels.
    max_absolute_difference = np.maximum(absolute_difference[:, :, 0],
                                         np.maximum(absolute_difference[:, :, 1], absolute_difference[:, :, 2]))
    spatial_weights = cf.get_spacial_distance_weights(filter)
    single_weights = cf.get_bilateral_suport_weights(max_absolute_difference, gamma_c, gamma_s)

    # weighted_difference = max_absolute_difference*spatial_weights[:, :, 0]
    weighted_difference = max_absolute_difference * single_weights

    sum_of_weighted_absolute_difference = np.sum(weighted_difference)

    return 3 * sum_of_weighted_absolute_difference / np.sum(filter)


@jit(nopython=disable_debug)
def match_scanlines_maclean(match, gap, egap,
                            current_index,
                            im2_padded, im1_padded,
                            scores, moves,
                            filter=np.ones((3, 3, 3), dtype=np.float64),
                            dmax=256,
                            gamma_c=10,
                            gamma_s=10,
                            gcl=(),
                            gcr=(),
                            alpha=0.0):
    start_x = int((filter.shape[1] - 1) / 2)
    start_y = int((filter.shape[0] - 1) / 2)

    im1_scanline, im2_scanline = im1_padded[int(current_index + start_y)], im2_padded[int(current_index + start_y)]

    cache_left = np.full((scores.shape[0], filter.shape[0], filter.shape[1], 3), np.inf, dtype=np.float64)
    cache_right = np.full((scores.shape[0], filter.shape[0], filter.shape[1], 3), np.inf, dtype=np.float64)

    for i in range(1, im1_scanline.shape[0] - start_x * 2 + 1):

        starting_index = 1 if i <= dmax + 1 else i - dmax

        # Is this correct? doesn't seem to be right, Mr. Maclean
        # It is! You were not right Mr. Gergo! Maclean:Gergo : 1:0

        for j in range(starting_index, i + 1):
            im1_index, im2_index = j - 1, i - 1
            match_raw = match - get_patch_absolute_difference(current_index, im1_index, im2_index,
                                                              im1_padded, im2_padded, filter,
                                                              gamma_c=gamma_c,
                                                              gamma_s=gamma_s,
                                                              cache_left=cache_left,
                                                              cache_right=cache_right,
                                                              gcl=gcl,
                                                              gcr=gcr,
                                                              alpha=alpha)
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


@jit(nopython=disable_debug, parallel=True)
def match_images(match, gap, egap,
                 im1, im2, scores_raw, moves_raw, scores_n_moves,
                 disparity, scanline_match_function,
                 filter=np.ones((3, 3, 3), dtype=np.float64),
                 gamma_c=10, gamma_s=10,
                 alpha=0.0):
    im1_padded = cf.pad_image_advanced_rgb(im1, filter.shape)
    im2_padded = cf.pad_image_advanced_rgb(im2, filter.shape)

    for i in prange(im1.shape[0]):
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = \
            scanline_match_function(match, gap, egap,
                                    i, im1_padded, im2_padded,
                                    np.copy(scores_raw),
                                    np.copy(moves_raw),
                                    filter=filter,
                                    gamma_c=gamma_c,
                                    gamma_s=gamma_s,
                                    # gcl = gcl,
                                    # gcr=gcr,
                                    alpha=alpha)

        temp = cf.generate_disparity_line(im1.shape[1], cf.get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
    return scores_n_moves, disparity


@jit(nopython=disable_debug)
def test_pipeline(match, gap, egap, im1, im2,
                  scanline_match_function=match_scanlines_maclean,
                  filter=np.ones((3, 3, 3),
                                 dtype=np.float64),
                  gamma_c=10,
                  gamma_s=10,
                  alpha=0.0):
    stop_here = 1
    scores_raw, moves_raw = cf.initialize_matrix_template_maclean(gap, egap, im1)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    disparity = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.float64)

    x, z = match_images(match, gap, egap, im1, im2, scores_raw,
                        moves_raw, scores_n_moves, disparity,
                        scanline_match_function, filter=filter,
                        gamma_c=gamma_c, gamma_s=gamma_s,
                        alpha=alpha)
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

    im1 = cv2.imread(im1_path).astype(np.float64)
    im2 = cv2.imread(im2_path).astype(np.float64)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    match = 10
    gap = -10
    egap = -1

    # scores_raw, moves_raw = initialize_matrix_template(gap, egap, im1)
    # scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    # disparity = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.float64)
    # test_1, test_2 = match_scanlines(match, gap, egap, 0, im2, im1, scores_raw, moves_raw)

    filter = np.ones((7, 3, 3), dtype=np.int32)
    SimpleTimer.timeit()
    x, z = test_pipeline(match, gap, egap, im1, im2, filter=filter, gamma_c=10, gamma_s=50, alpha=0.0)
    # x,y,z = match_images(80, -30, -2, im2, im1)
    # x,y,z = FakeNumbaClass["match_images"] (100, -15, -5, im2, im1)
    # Wrapper.match_images( im2, im1)

    SimpleTimer.timeit()
    z_mod = z
    # calc metrix!!!
    BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = \
        m.evaluate_over_all(z * 4, gt, occ, occlusions_counted_in_errors=False)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(z_mod, cmap="gray")
    plt.title("Bad4:{0}".format(BAD4))
