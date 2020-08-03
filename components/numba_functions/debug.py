import numpy as np
from numba import jit
from numba import vectorize, float64

from components.numba_functions import common_functions as cf
from components.utils.SimpleTimer import SimpleTimer

disable_debug = True


disable_debug = True

#@jit(nopython=disable_debug, parallel=True)
def calculate_gradients_de_maetzu(img_padded):
    grad_filter_x = np.array([1, 0, -1]).astype(np.float64)
    grad_filter_y = grad_filter_x.T

    grads = np.zeros(img_padded.shape, dtype=np.float64)
    phase = np.zeros(img_padded.shape, dtype=np.float64)
    for i in range(1, int(img_padded.shape[0] - 1)):
        for j in range(1, int(img_padded.shape[1] - 1)):
            row_start = i - 1
            row_end = i + 2
            cols_start = j - 1
            cols_end = j + 2

            G_x = np.sum((img_padded[i, cols_start:cols_end] * grad_filter_x))
            G_y = np.sum((img_padded[row_start:row_end, j] * grad_filter_y))

            # print(img_padded[row_start: row_end, cnp.sumols_start:cols_end].shape)

            grads[i, j] = np.sqrt(G_x ** 2 + G_y ** 2)
            angle_tan = G_y / G_x if G_x !=0 else 0
            angle_arctan = np.arctan(angle_tan)
            phase[i, j] = angle_arctan
    return grads, phase
#@jit(nopython=disable_debug)
def get_straceback_start(currentIndex, scores_n_moves):
    lastColumn = scores_n_moves[0, currentIndex, :, -1]
    lastRow = scores_n_moves[0, currentIndex, -1, :]
    last_column_max = np.max(lastColumn)
    last_row_max = np.max(lastRow)

    last_column_max_index = np.argmax(lastColumn)
    last_row_max_index = np.argmax(lastRow)
    return np.array([last_column_max_index, scores_n_moves.shape[3] - 1]).astype(np.int32) \
        if (last_column_max > last_row_max) \
        else np.array([scores_n_moves.shape[3] - 1, last_row_max_index]).astype(np.int32)


#@jit(nopython=disable_debug)
def get_traceback_path(currentIndex, scores_n_moves):
    curY, curX = get_straceback_start(currentIndex, scores_n_moves)

    next_move_mapping = np.array([(-1, 0), (-1, -1), (0, -1)], dtype=np.int32)
    moves = [0]
    while (curY > 0 and curX > 0):
        # curY-=1
        # curX -= 1
        previousMove = np.uint32(scores_n_moves[1, currentIndex, curY, curX])

        nexCoordinates = next_move_mapping[previousMove]

        curY += nexCoordinates[0]
        curX += nexCoordinates[1]

        moves.append(previousMove)
    temp = np.array(moves[1:]).astype(np.int32)
    return temp[::-1]


#@jit(nopython=disable_debug)
def generate_disparity_line(cols, scanline_traceback):
    scanline = np.zeros((cols), dtype=np.float64)
    tops = 0
    lefts = 0
    currentPixel = 0
    # print(chr(48))
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
    # print(scanline.shape)
    return scanline


# support weight functions
#@jit(nopython=disable_debug)
def get_color_distance(p, q):
    return np.sqrt(np.square(np.subtract(p, q)))


#@jit(nopython=disable_debug)
def get_color_rule_component(window, gamma_c):
    p_x = int(window.shape[0] / 2)
    p_y = int(window.shape[1] / 2)
    reference_pixel = window[p_y, p_x]
    delta_c = get_color_distance(reference_pixel, window)
    return np.exp(-(delta_c / gamma_c))
# Eucledian distance of pixel coordinates
#@jit(nopython=disable_debug)
def get_spacial_distance_weights(window):
    spatial_distance = np.zeros(window.shape)
    p_x = int(window.shape[0] / 2)
    p_y = int(window.shape[1] / 2)
    center_coords = (p_y, p_x)
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            d_y = np.square(center_coords[0] - i)
            d_x = np.square(center_coords[1] - j)
            spatial_distance[i, j] = np.sqrt(d_y + d_x)
    # as the spatial difference for one window size will always
    # be the same throughout the image
    return spatial_distance




#@jit(nopython=disable_debug)
def get_spatial_rule_component(gamma_s, spacial_distance_weight):
    delta_s = spacial_distance_weight
    return np.exp(-(delta_s / gamma_s))


#@jit(nopython=disable_debug)
def get_bilateral_suport_weights(window, gamma_c, gamma_s):
    w_c = get_color_rule_component(window, gamma_c)
    spatial_weights = get_spacial_distance_weights(window)
    w_s = get_spatial_rule_component(gamma_s, spatial_weights)
    return w_c * w_s


#@jit(nopython=disable_debug)
def get_bilateral_suport_weights_sum(window, gamma_c, gamma_s):
    w_c = get_color_rule_component(window, gamma_c)
    spatial_weights = get_spacial_distance_weights(window)
    w_s = get_spatial_rule_component(gamma_s, spatial_weights)
    return w_c + w_s



#@jit(nopython=disable_debug, parallel=True)
def calculate_gradients(img_padded):
    grad_filter = np.array([1, 0, -1])
    grad_indices = np.array([-1, 0, 1])

    grads = np.zeros(img_padded.shape, dtype=np.float64)

    for i in range(1, img_padded.shape[0]-1):
        current_row = img_padded[i]
        for j in range(1, img_padded.shape[1]-1):
            current_indices = j + grad_indices
            # print(current_indices)
            grads[int(i), int(j)] = np.sum(np.abs(current_row[current_indices] * grad_filter)) / 2
    return grads


#@jit(nopython=disable_debug, parallel=True)
def calculate_gradients_sobel(img_padded):
    grad_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float64)

    grads = np.zeros(img_padded.shape, dtype=np.float64)

    for i in range(1, int(img_padded.shape[0] - 1)):
        for j in range(1, int(img_padded.shape[1] - 1)):
            row_start = i - 1
            row_end = i + 2
            cols_start = j - 1
            cols_end = j + 2
            # print(img_padded[row_start: row_end, cols_start:cols_end].shape)

            grads[i + 1, j + 1] = np.sum(img_padded[row_start: row_end, cols_start:cols_end] * grad_filter)
    return grads


#@jit(nopython=disable_debug, parallel=True)
def calculate_gradients_de_maetzu_rgb(img_padded):
    grad_filter_x = np.array([1, 0, -1]).astype(np.float64) *np.ones([3,1])
    grad_filter_y = grad_filter_x.T

    grads = np.zeros(img_padded.shape, dtype=np.float64)
    phase = np.zeros(img_padded.shape, dtype=np.float64)
    for i in range(1, int(img_padded.shape[0] - 1)):
        for j in range(1, int(img_padded.shape[1] - 1)):
            row_start = i - 1
            row_end = i + 2
            cols_start = j - 1
            cols_end = j + 2
            img_x_slice = img_padded[i, cols_start:cols_end]
            img_y_slice = img_padded[row_start:row_end, j]
            G_x = np.sum((img_x_slice * grad_filter_x), axis=1)
            G_y = np.sum((img_y_slice * grad_filter_y), axis=0)

            modulus = np.sqrt(G_x ** 2 + G_y ** 2)
            grads[i, j] = modulus
            angle_tan = safe_division(G_y, G_x)
            angle_arctan = np.arctan(angle_tan)
            phase[i, j] = angle_arctan
    return grads, phase

#@jit(nopython=disable_debug)
def fill_up_first_rows_default(matrix, gap, egap):
    matrix[0, 1:] = gap * (matrix.shape[0] - 1)


#@jit(nopython=disable_debug)
def fill_up_first_rows_v2(matrix, gap, egap):
    for i in range(len(matrix[0])):
        matrix[0, i] = gap * (len(matrix[0]) - i)


#@jit(nopython=disable_debug)
def fill_up_first_rows_v3(matrix, gap, egap):
    for i in range(len(matrix[0])):
        if (i == 1):
            matrix[0, i] = gap
        if (i > 1):
            matrix[0, i] = gap + egap * (i - 1)


#@jit(nopython=disable_debug)
def initialize_matrix_template(gap, egap, img, rows_init_func=fill_up_first_rows_default):
    scores = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.float64)
    moves = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.uint32)

    moves[0, 1:] = 2
    rows_init_func(scores, gap, egap)
    rows_init_func(scores.T, gap, egap)
    return scores, moves


#@njit(parallel=False)
def maclean_scores(matrix, gap, egap):
    for i in range(0, matrix.shape[0]):
        matrix[i:, i] = np.array([(i) * gap for i in range(i, matrix.shape[0])]).T
        matrix[i, i:] = np.array([(i) * gap for i in range(i, matrix.shape[1])])
    return matrix


#@njit(parallel=False)
def maclean_moves(matrix):
    for i in range(0, matrix.shape[0]):
        matrix[i:, i] = 0
        matrix[i, i:] = 2
    return matrix


#@njit(parallel=False)
def initialize_matrix_template_maclean(gap, egap, img, rows_init_func=fill_up_first_rows_default):
    scores = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.float64)
    moves = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.uint32)

    moves = maclean_moves(moves)
    scores = maclean_scores(scores, gap, egap)
    return scores, moves


#@jit(nopython=disable_debug)
def calculate_same_padding_offset(filter_dims):
    # based on: output_dim = (n+2*p-f+1)**2 if they are symmetrical
    vpadding = np.int32((filter_dims[0] - 1) / 2) if filter_dims[0] != 1 else 0
    hpadding = np.int32((filter_dims[1] - 1) / 2) if filter_dims[1] != 1 else 0
    return np.array((vpadding, hpadding), dtype=np.int32)


#@jit(nopython=disable_debug)
def pad_image_advanced(img, filter_dims):
    img_dims = img.shape
    padding = calculate_same_padding_offset(filter_dims)

    height = int(img_dims[0] + padding[0] * 2)
    width = int(img_dims[1] + padding[1] * 2)

    # fix assymetrix paddings and 1x1 filters here!
    new_img = np.zeros((height, width), dtype=np.float64)
    end_row = -padding[0] if padding[0] > 0 else img_dims[0]
    end_column = -padding[1] if padding[1] > 0 else img_dims[1]

    new_img[padding[0]:end_row, padding[1]:end_column] = img

    return new_img


#@jit(nopython=disable_debug)
def pad_image_advanced_rgb(img, filter_dims):
    img_dims = img.shape
    padding = calculate_same_padding_offset(filter_dims)

    height = int(img_dims[0] + padding[0] * 2)
    width = int(img_dims[1] + padding[1] * 2)

    # fix assymetrix paddings and 1x1 filters here!
    new_img = np.zeros((height, width, 3), dtype=np.float64)
    end_row = -padding[0] if padding[0] > 0 else img_dims[0]
    end_column = -padding[1] if padding[1] > 0 else img_dims[1]

    new_img[padding[0]:end_row, padding[1]:end_column] = img

    return new_img


#@vectorize([float64(float64)])
#@jit(nopython=disable_debug)
def phase_function(phase):
    pie = np.pi
    if (phase <= pie):
        return phase
    return 2 * pie - phase

@vectorize([float64(float64, float64)])
#@jit(nopython=disable_debug)
def safe_division(a,b):
    if(b!=0):
        return a/b
    return b

#@jit(nopython=disable_debug)
def get_patch_absolute_difference_de_maetzu(currentIndex,
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
                                            T_s=0.008,
                                            pl=(),
                                            pr=()):
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
        p1_weights = get_bilateral_suport_weights_sum(patch1, gamma_c, gamma_s)
        cache_left[key_left] = p1_weights

    if cache_right[key_right, 0, 0] == np.inf:
        p2_weights = get_bilateral_suport_weights_sum(patch2, gamma_c, gamma_s)
        cache_right[key_right] = p2_weights

    left_window = cache_left[key_left]
    right_window = cache_right[key_right]

    grad_left = gcl[startRow:endRow, img1_start_column:img1_end_column]
    grad_right = gcr[startRow:endRow, img2_start_column:img2_end_column]

    phase_left = pl[startRow:endRow, img1_start_column:img1_end_column]
    phase_right = pr[startRow:endRow, img2_start_column:img2_end_column]

    grad_component = alpha * np.abs(left_window*grad_left - grad_right*right_window)
    phase_component = np.abs(left_window*phase_left - right_window*phase_right)

    sum_of_components = grad_component + phase_function(phase_component)
    # sum_of_components = grad_component + phase_component

    intensity_min_array = np.full_like(filter, T_c)

    cost = np.minimum(sum_of_components, intensity_min_array)
    weighted_cost = cost

    return np.sum(weighted_cost)
    """
    return 0 #"""


#@jit(nopython=disable_debug)
def get_patch_absolute_difference_de_maetzu_bak(currentIndex,
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
                                            T_s=0.008,
                                            pl=(),
                                            pr=()):
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
        p1_weights = get_bilateral_suport_weights_sum(patch1, gamma_c, gamma_s)
        cache_left[key_left] = p1_weights

    if cache_right[key_right, 0, 0] == np.inf:
        p2_weights = get_bilateral_suport_weights_sum(patch2, gamma_c, gamma_s)
        cache_right[key_right] = p2_weights

    left_window = cache_left[key_left]
    right_window = cache_right[key_right]

    grad_left = gcl[startRow:endRow, img1_start_column:img1_end_column]
    grad_right = gcr[startRow:endRow, img2_start_column:img2_end_column]

    phase_left = pl[startRow:endRow, img1_start_column:img1_end_column]
    phase_right = pr[startRow:endRow, img2_start_column:img2_end_column]

    grad_component = alpha * np.abs(grad_left - grad_right)
    phase_component = np.abs(phase_left - phase_right)

    sum_of_components = grad_component + phase_function(phase_component)
    # sum_of_components = grad_component + phase_component

    intensity_min_array = np.full_like(filter, T_c)

    cost = np.minimum(sum_of_components, intensity_min_array)
    common_term = left_window * right_window
    weighted_cost = (cost * common_term) / common_term

    return np.sum(weighted_cost)
    """
    return 0 #"""

#@jit(nopython=disable_debug)
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
                            pl=(),
                            pr=(),
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

            match_raw = match - get_patch_absolute_difference_de_maetzu(current_index, im1_index, im2_index,
                                                                        im1_padded, im2_padded, filter,
                                                                        gamma_c=gamma_c,
                                                                        gamma_s=gamma_s,
                                                                        cache_left=cache_left,
                                                                        cache_right=cache_right,
                                                                        gcl=gcl,
                                                                        gcr=gcr,
                                                                        pl=pl,
                                                                        pr=pr,
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


#@jit(nopython=disable_debug, parallel=False)
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
    im1_padded = pad_image_advanced_rgb(im1, filter.shape)
    im2_padded = pad_image_advanced_rgb(im2, filter.shape)
    gcl, pl = calculate_gradients_de_maetzu_rgb(im1_padded)
    gcr, pr = calculate_gradients_de_maetzu_rgb(im2_padded)
    for i in range(im1.shape[0]):
        # print(i)
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = \
            scanline_match_function(match, gap, egap,
                                    i, im1_padded, im2_padded,
                                    np.copy(scores_raw), np.copy(moves_raw), filter=filter,
                                    gamma_c=gamma_c,
                                    gamma_s=gamma_s,
                                    gcl=gcl,
                                    gcr=gcr,
                                    pl=pl,
                                    pr=pr,
                                    alpha=alpha,
                                    T_c=T_c,
                                    T_s=T_s)

        temp = generate_disparity_line(im1.shape[1], get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
        # print(i)
    return scores_n_moves, disparity


#@jit(nopython=disable_debug)
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
    scores_raw, moves_raw = initialize_matrix_template_maclean(gap, egap, im1)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    disparity = np.zeros(im1.shape, dtype=np.float64)

    x, z = match_images(match, gap, egap, im1, im2, scores_raw,
                        moves_raw, scores_n_moves, disparity,
                        scanline_match_function,
                        filter=filter,
                        gamma_c=gamma_c,
                        gamma_s=gamma_s,
                        alpha=alpha,
                        T_c=T_c,
                        T_s=T_s)
    return x, z


if __name__ == "__main__":
    import cv2
    import os
    from components.utils.Metrix import Wrapper as m
    import matplotlib.pyplot as plt

    scaler = 1
    scene = "teddy"
    im1_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im2.png")
    im2_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im6.png")
    gt_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "disp2.png")
    occ_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "nonocc.png")
    im1_gray = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / scaler

    im1 = cv2.imread(im1_path).astype(np.float64) / scaler
    im2 = cv2.imread(im2_path).astype(np.float64) / scaler
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    img_padded = pad_image_advanced_rgb(im1, np.array([3,3]))
    grads = calculate_gradients_de_maetzu_rgb(im1)
    #plt.imshow(grads[0][:,:,2]/grads[0].max())
