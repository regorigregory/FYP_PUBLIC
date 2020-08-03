from numba import jit
from numba import prange
from components.numba_functions import common_functions as cf
from components.utils.SimpleTimer import SimpleTimer
import numpy as np

disable_debug = True

@jit(nopython=disable_debug)
def match_images(match, gap, egap, im1, im2, scores_raw, moves_raw, scores_n_moves, disparity, scanline_match_function,
                 dmax=256):
    for i in prange(im1.shape[0]):
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = scanline_match_function \
            (match, gap, egap, i, im1, im2, np.copy(scores_raw), np.copy(moves_raw), dmax=dmax)
        temp = cf.generate_disparity_line(im1.shape[1], cf.get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
    return scores_n_moves, disparity


@jit(nopython=disable_debug)
def match_scanlines(match, gap, egap, current_index, im2, im1, scores, moves, dmax=None):
    im1_scanline, im2_scanline = im1[current_index], im2[current_index]
    for i in range(1, im1_scanline.shape[0] + 1):
        for j in range(1, im1_scanline.shape[0] + 1):
            im1_index, im2_index = j - 1, i - 1
            match_raw = match - abs(im1_scanline[im1_index] - im2_scanline[im2_index])
            print(match_raw)
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


@jit(nopython=disable_debug)
def match_scanlines_maclean(match, gap, egap, current_index, im2, im1, scores, moves, dmax=256):
    im1_scanline, im2_scanline = im1[current_index], im2[current_index]
    for i in range(1, im1_scanline.shape[0] + 1):
        starting_index = 1 if i <= dmax + 1 else i - dmax

        for j in range(starting_index, i + 1):
            im1_index, im2_index = j - 1, i - 1
            match_raw = match - abs(im1_scanline[im1_index] - im2_scanline[im2_index])
            print(match_raw)
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


def test_pipeline(match, gap, egap, im1, im2, scanline_match_function=match_scanlines,
                 row_init_function=cf.fill_up_first_rows_default):
    scores_raw, moves_raw = cf.initialize_matrix_template_maclean(gap, egap, im1, rows_init_func=row_init_function)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    traceback = np.zeros_like([im1.shape[0], ], dtype=np.int32)
    disparity = np.zeros(im1.shape, dtype=np.float64)
    threadsperblock = 32
    # blockspergrid = (an_array.size + (threadsperblock - 1)) // threadperblock

    x, z = match_images(match, gap, egap, im1, im2, scores_raw, moves_raw, scores_n_moves, disparity,
                        match_scanlines_maclean, dmax=64)
    return x, z


if __name__ == "__main__":
    import cv2
    import os
    import numpy as np
    from components.utils.Metrix import Wrapper as m

    scene = "teddy"
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

    SimpleTimer.timeit()
    x, z = test_pipeline(match, gap, egap, im1, im2)

    SimpleTimer.timeit()

    BAD1, BAD2, BAD4, BAD8, ABS_ERR, MSE, AVG, EUCLEDIAN = \
        m.evaluate_over_all(z * 4, gt, occ, occlusions_counted_in_errors=False)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(z, cmap="gray")
    plt.title("Bad4:{0}".format(BAD4))
