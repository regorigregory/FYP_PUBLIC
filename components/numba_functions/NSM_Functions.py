import numpy as np
from numba import njit, cuda
from numba import prange
from components.utils.SimpleTimer import SimpleTimer
@njit(parallel=True)
def match_images(match, gap, egap, im1, im2, scores_raw, moves_raw, scores_n_moves, disparity, scanline_match_function, dmax=256):
    for i in prange(im1.shape[0]):
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = scanline_match_function\
            (match, gap, egap, i, im1, im2, np.copy(scores_raw), np.copy(moves_raw), dmax=dmax)
        temp = generate_disparity_line(im1.shape[1], get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
    return scores_n_moves, disparity

@njit(parallel=False)
def match_scanlines(match, gap, egap, current_index, im2, im1, scores, moves, dmax=None):
    im1_scanline, im2_scanline = im1[current_index], im2[current_index]
    for i in range(1, im1_scanline.shape[0]+1):
        for j in range(1, im1_scanline.shape[0]+1):
            im1_index, im2_index= j-1, i-1
            match_raw = match - abs(im1_scanline[im1_index] - im2_scanline[im2_index])

            east, north, ne = (i,j-1), (i-1,j), (i-1,j-1)
            east_raw, north_raw, ne_raw = scores[east], scores[north], scores[ne]
            east_dir, north_dir, ne_dir = moves[east], moves[north], moves[ne]

            east_raw += egap if east_dir == 2 else gap
            north_raw += egap if north_dir == 0 else gap
            match_raw += ne_raw

            all_scores = np.array([north_raw, match_raw, east_raw])

            winner_index = np.argmax(all_scores)

            scores[i, j] =all_scores[winner_index]
            moves[i, j] = winner_index

    return scores, moves

@njit(parallel=False)
def match_scanlines_maclean(match, gap, egap, current_index, im2, im1, scores, moves, dmax=256):
    im1_scanline, im2_scanline = im1[current_index], im2[current_index]
    for i in range(1, im1_scanline.shape[0]+1):
        starting_index = 1 if i <= dmax + 1 else i - dmax

        for j in range(starting_index, i+1):
            im1_index, im2_index= j-1, i-1
            match_raw = match - abs(im1_scanline[im1_index] - im2_scanline[im2_index])

            east, north, ne = (i,j-1), (i-1,j), (i-1,j-1)
            east_raw, north_raw, ne_raw = scores[east], scores[north], scores[ne]
            east_dir, north_dir, ne_dir = moves[east], moves[north], moves[ne]

            east_raw += egap if east_dir == 2 else gap
            north_raw += egap if north_dir == 0 else gap
            match_raw += ne_raw

            all_scores = np.array([north_raw, match_raw, east_raw])

            winner_index = np.argmax(all_scores)

            scores[i, j] =all_scores[winner_index]
            moves[i, j] = winner_index

    return scores, moves

@njit
def fill_up_first_rows_default(matrix, gap, egap):
    matrix[0, 1:] =  gap * (matrix.shape[0]-1)

@njit
def fill_up_first_rows_v2(matrix, gap, egap):
    for i in range(len(matrix[0])):
        matrix[0, i] = gap * (len(matrix[0])-i)

@njit
def fill_up_first_rows_v3(matrix, gap, egap):

    for i in range(len(matrix[0])):
        if (i == 1):
           matrix[0, i] = gap
        if (i > 1):
           matrix[0, i] = gap + egap * (i - 1)

@njit(parallel = True)
def initialize_matrix_template(gap, egap, img, rows_init_func = fill_up_first_rows_default):

    scores= np.zeros((img.shape[1] + 1, img.shape[1]+1), dtype=np.float64)
    moves = np.zeros((img.shape[1] + 1, img.shape[1]+1), dtype = np.uint32)

    moves[0, 1:] = 2
    rows_init_func(scores, gap, egap)
    rows_init_func(scores.T, gap, egap)
    return scores, moves

@njit(parallel = True)
def initialize_matrix_template_maclean\
                (gap, egap, img,
                 rows_init_func = fill_up_first_rows_default):

    scores= np.zeros((img.shape[1] + 1, img.shape[1]+1), dtype=np.float64)
    moves = np.zeros((img.shape[1] + 1, img.shape[1]+1), dtype = np.uint32)

    moves = maclean_moves(moves)
    scores = maclean_scores(scores, gap, egap)
    return scores, moves

@njit(parallel=False)
def maclean_scores(matrix, gap, egap):
    for i in range(0, matrix.shape[0]):
        matrix[i:, i] = np.array([(i) * gap for i in range(i, matrix.shape[0])]).T
        matrix[i, i:] = np.array([(i) * gap for i in range(i, matrix.shape[1])])
    return matrix

@njit(parallel=False)
def maclean_moves(matrix):
    for i in range(0, matrix.shape[0]):
        matrix[i:, i] = 0
        matrix[i, i:] = 2
    return matrix
@njit(parallel=False)
def get_traceback_path(currentIndex, scores_n_moves):
    curY, curX = get_straceback_start(currentIndex, scores_n_moves)

    next_move_mapping = np.array([(-1, 0), (-1, -1), (0, -1)], dtype = np.int32)
    moves = [0]
    while (curY > 0 and curX > 0):
        #curY-=1
        #curX -= 1
        previousMove= np.uint32(scores_n_moves[1, currentIndex, curY, curX])

        nexCoordinates = next_move_mapping[previousMove]

        curY += nexCoordinates[0]
        curX += nexCoordinates[1]

        moves.append(previousMove)
    temp =np.array(moves[1:]).astype(np.int32)
    return temp[::-1]

@njit(parallel=False)
def get_straceback_start(currentIndex, scores_n_moves):
    lastColumn = scores_n_moves[0, currentIndex, :, -1]
    lastRow = scores_n_moves[0, currentIndex, -1, :]
    last_column_max = np.max(lastColumn)
    last_row_max = np.max(lastRow)

    last_column_max_index = np.argmax(lastColumn)
    last_row_max_index = np.argmax(lastRow)



    return np.array([last_column_max_index, scores_n_moves.shape[3]-1]).astype(np.int32) if (last_column_max > last_row_max) else np.array([scores_n_moves.shape[3]-1, last_row_max_index]).astype(np.int32)

@njit(parallel=False)
def generate_disparity_line(cols, scanline_traceback):
    scanline = np.zeros((cols), dtype=np.float64)
    tops = 0
    lefts = 0
    currentPixel=0
    #print(chr(48))
    for direction in scanline_traceback:
        if (direction == 2):
            lefts += 1
        elif (direction == 0):
            tops += 1

            scanline[currentPixel] = 0
            currentPixel += 1
        elif (direction == 1):
            scanline[currentPixel] = np.abs(tops - lefts)
            currentPixel += 1
        else:
            print("Something is not right here!")
    #print(scanline.shape)
    return scanline

def test_pipline(match, gap, egap, im1, im2, scanline_match_function=match_scanlines, row_init_function = fill_up_first_rows_default):
    scores_raw, moves_raw = initialize_matrix_template_maclean(gap, egap, im1, rows_init_func=row_init_function)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    traceback = np.zeros_like([im1.shape[0], ], dtype=np.int32)
    disparity = np.zeros(im1.shape, dtype=np.float64)
    threadsperblock = 32
    #blockspergrid = (an_array.size + (threadsperblock - 1)) // threadperblock
    
    x,z = match_images(match, gap, egap, im1, im2, scores_raw, moves_raw, scores_n_moves, disparity, match_scanlines_maclean, dmax=64)
    return x,z


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    from components.utils.SintelReader import Wrapper as SintelReader
    ###################################################################
    # Middlebury Images################################################
    ###################################################################
    import sys
    import os
    from components.utils import middlebury_utils as mbu
    from components.utils import plot_utils as plu
    # sys.path.append("../../")

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
        plu.plot_images(im, filenames)
    plt.show()
    path = os.path.join("..", "..", "datasets", "sintel", "training")
    reader = SintelReader(rootPath=path)
    reader.print_available_scenes()
    reader.set_selected_scene('ambush_5')
    left, right, disp,occ, outoff = reader.get_selected_scene_next_files()
    #im1 = np.random.randint(0, 255, [2,4])
    #im2 = np.random.randint(0, 255, [2, 4])
    match = 60
    gap = -30
    egap = -1
    SimpleTimer.timeit()
    x,z = test_pipline(match, gap, egap, midd_loaded_imgs[0][0][0], midd_loaded_imgs[0][0][1])
    #x,y,z = match_images(80, -30, -2, im2, im1)

    SimpleTimer.timeit()
    z_mod = np.max(z)-z

    plt.figure()
    plt.imshow(z, cmap = cm.gray)
    plt.show()
