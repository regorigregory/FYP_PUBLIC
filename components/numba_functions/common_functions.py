import numpy as np
from numba import njit, jit, prange


disable_debug = True

@jit(nopython=disable_debug)
def get_straceback_start(currentIndex, scores_n_moves):
    lastColumn = scores_n_moves[0, currentIndex, :, -1]
    lastRow = scores_n_moves[0, currentIndex, -1, :]
    last_column_max = np.max(lastColumn)
    last_row_max = np.max(lastRow)

    last_column_max_index = np.argmax(lastColumn)
    last_row_max_index = np.argmax(lastRow)
    return np.array([last_column_max_index, scores_n_moves.shape[3]-1]).astype(np.int32) \
        if (last_column_max > last_row_max) \
        else np.array([scores_n_moves.shape[3]-1, last_row_max_index]).astype(np.int32)

@jit(nopython=disable_debug)
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

@jit(nopython=disable_debug)
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

# support weight functions
@jit(nopython=disable_debug)
def get_color_distance(p,q):
    return np.sqrt(np.square(np.subtract(p,q)))

#Eucledian distance of pixel coordinates
@jit(nopython=disable_debug)
def get_spacial_distance_weights(window):
    spatial_distance = np.zeros(window.shape)
    p_x = int(window.shape[0] / 2)
    p_y = int(window.shape[1] / 2)
    center_coords = (p_y, p_x)
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            d_y = np.square(center_coords[0]-i)
            d_x = np.square(center_coords[1]-j)
            spatial_distance[i, j] = np.sqrt(d_y+d_x)
    # as the spatial difference for one window size will always
    #be the same throughout the image
    return spatial_distance

@jit(nopython=disable_debug)
def get_color_rule_component(window, gamma_c):
    p_x = int(window.shape[0]/2)
    p_y = int(window.shape[1]/2)
    reference_pixel = window[p_y, p_x]
    delta_c = get_color_distance(reference_pixel, window)
    return np.exp(-(delta_c/gamma_c))

@jit(nopython=disable_debug)
def get_spatial_rule_component(gamma_s, spacial_distance_weight):
    delta_s = spacial_distance_weight
    return np.exp(-(delta_s/gamma_s))

@jit(nopython=disable_debug)
def get_bilateral_suport_weights(window, gamma_c, gamma_s):
    w_c = get_color_rule_component(window, gamma_c)
    spatial_weights = get_spacial_distance_weights(window)
    w_s = get_spatial_rule_component(gamma_s, spatial_weights)
    return w_c*w_s


@jit(nopython=disable_debug, parallel = True)
def calculate_gradients(img_padded):

    grad_filter = np.array([1,0,-1])
    grad_indices = np.array([-1,0,1])

    grads = np.zeros(img_padded.shape, dtype=np.float64)

    for i in prange(1, img_padded.shape[0]):
        current_row = img_padded[i]
        for j in prange(1, img_padded.shape[1]):
            current_indices = j+grad_indices
            #print(current_indices)
            grads[int(i),int(j)] = np.sum(np.abs(current_row[current_indices]*grad_filter))/2
    return grads

@jit(nopython=disable_debug, parallel = True)
def calculate_gradients_sobel(img_padded):

    grad_filter = np.array([[1,0,-1], [2,0,-2], [1,0,-1]]).astype(np.float64)

    grads = np.zeros(img_padded.shape, dtype=np.float64)

    for i in prange(1, int(img_padded.shape[0]-1)):
        for j in prange(1, int(img_padded.shape[1]-1)):
            row_start = i-1
            row_end =  i+2
            cols_start  = j-1
            cols_end = j+2
            #print(img_padded[row_start: row_end, cols_start:cols_end].shape)

            grads[i+1,j+1] = np.sum(img_padded[row_start: row_end, cols_start:cols_end]*grad_filter)
    return grads

@jit(nopython=disable_debug)
def fill_up_first_rows_default(matrix, gap, egap):
    matrix[0, 1:] =  gap * (matrix.shape[0]-1)

@jit(nopython=disable_debug)
def fill_up_first_rows_v2(matrix, gap, egap):
    for i in range(len(matrix[0])):
        matrix[0, i] = gap * (len(matrix[0])-i)

@jit(nopython=disable_debug)
def fill_up_first_rows_v3(matrix, gap, egap):

    for i in range(len(matrix[0])):
        if (i == 1):
           matrix[0, i] = gap
        if (i > 1):
           matrix[0, i] = gap + egap * (i - 1)

@jit(nopython=disable_debug)
def initialize_matrix_template(gap, egap, img, rows_init_func=fill_up_first_rows_default):
    scores = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.float64)
    moves = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.uint32)

    moves[0, 1:] = 2
    rows_init_func(scores, gap, egap)
    rows_init_func(scores.T, gap, egap)
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
def initialize_matrix_template_maclean(gap, egap, img, rows_init_func=fill_up_first_rows_default):
    scores = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.float64)
    moves = np.zeros((img.shape[1] + 1, img.shape[1] + 1), dtype=np.uint32)

    moves = maclean_moves(moves)
    scores = maclean_scores(scores, gap, egap)
    return scores, moves

@jit(nopython=disable_debug)
def calculate_same_padding_offset(filter_dims):
    #based on: output_dim = (n+2*p-f+1)**2 if they are symmetrical
    vpadding = np.int32((filter_dims[0]-1)/2) if filter_dims[0] !=1 else 0
    hpadding = np.int32((filter_dims[1]-1)/2) if filter_dims[1] !=1 else 0
    return np.array((vpadding, hpadding), dtype=np.int32)

@jit(nopython=disable_debug)
def pad_image_advanced(img, filter_dims):
    img_dims = img.shape
    padding = calculate_same_padding_offset(filter_dims)

    height = int(img_dims[0]+padding[0]*2)
    width = int(img_dims[1]+padding[1]*2)

    #fix assymetrix paddings and 1x1 filters here!
    new_img = np.zeros((height, width), dtype=np.float64)
    end_row = -padding[0] if padding[0]>0 else img_dims[0]
    end_column = -padding[1] if padding[1]>0 else img_dims[1]

    new_img[padding[0]:end_row, padding[1]:end_column] = img

    return new_img



