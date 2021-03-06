import numpy as np
from numba import njit, jit, prange
from components.utils.SimpleTimer import SimpleTimer
from components.numba_functions import common_functions as cf

disable_debug = True
@jit(nopython=disable_debug)
def match_scanlines(match, gap, egap, current_index, im2_padded, im1_padded, scores, moves, filter=np.ones((3,3), dtype=np.int32)):

    start_x =int((filter.shape[1]-1)/2)
    start_y = int((filter.shape[0]-1)/2)

    im1_scanline, im2_scanline = im1_padded[int(current_index+start_y)], im2_padded[int(current_index+start_y)]

    for i in range(1, im1_scanline.shape[0]-start_x*2+1):
        #print(i)
        for j in range(1, im1_scanline.shape[0]-start_x*2+1):
            im1_index, im2_index= j-1, i-1
            #match_raw = match - abs(im1_scanline[im1_index] - im2_scanline[im2_index])
            #match_raw =  match - abs(im1_scanline[im1_index] - im2_scanline[im2_index]) #get_patch_absolute_difference(current_index, im1_index, im2_index,im1, im2)
            match_raw = match -  get_patch_absolute_difference(current_index, im1_index, im2_index,im1_padded, im2_padded, filter)
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
def initialize_matrix_template(gap, egap, img, rows_init_func = fill_up_first_rows_default):

    scores= np.zeros((img.shape[1] + 1, img.shape[1]+1), dtype=np.float64)
    moves = np.zeros((img.shape[1] + 1, img.shape[1]+1), dtype = np.uint32)

    moves[0, 1:] = 2
    rows_init_func(scores, gap, egap)
    rows_init_func(scores.T, gap, egap)
    return scores, moves



@jit(nopython=disable_debug)
def get_patch_absolute_difference(currentIndex, im1_pixel_index, im2_pixel_index,im1_padded, im2_padded, filter=np.ones((3,3), dtype=np.int32)):
    #here, you have to calculate the patches indices without the default value
    filter_y = filter.shape[0]
    filter_x = filter.shape[1]
    startRow = currentIndex
    endRow = int(currentIndex + (filter_y))# slicing is not inclusive

    # what if you are at the edges? you need to handle it
    img1_start_column = int(im1_pixel_index)
    img1_end_column = int(im1_pixel_index+filter_x)
    img2_start_column = int(im2_pixel_index)
    img2_end_column = int(im2_pixel_index + filter_x)


    patch1 = np.asarray(im1_padded[startRow:endRow, img1_start_column:img1_end_column], dtype=np.float64)*filter


    patch2 = np.asarray(im2_padded[startRow:endRow, img2_start_column:img2_end_column], dtype=np.float64)*filter
    absolute_difference = np.abs(patch1-patch2)
    sum_of_absolute_difference = np.sum(absolute_difference)

    return sum_of_absolute_difference/filter_y*filter_x


@jit(nopython=disable_debug, parallel = False)
def match_images(match, gap, egap, im1, im2, scores_raw, moves_raw, scores_n_moves, disparity, scanline_match_function, filter = np.ones((3,3), dtype=np.int32)):
    im1_padded = pad_image_advanced(im1, filter.shape)
    im2_padded = pad_image_advanced(im2, filter.shape)
    for i in prange(im1.shape[0]):
        #print(i)
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = \
            scanline_match_function(match, gap, egap, i, im1_padded, im2_padded, np.copy(scores_raw), np.copy(moves_raw), filter=filter)
        temp = generate_disparity_line(im1.shape[1], get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
        #print(i)
    return scores_n_moves, disparity

@jit(nopython=disable_debug)
def calculate_same_padding(filter_dims):
    #based on: output_dim = (n+2*p-f+1)**2 if they are symmetrical
    vpadding = np.int32((filter_dims[0]-1)/2)
    hpadding = np.int32((filter_dims[1]-1)/2)
    return np.array((vpadding, hpadding), dtype=np.int32)

@jit(nopython=disable_debug)
def get_output_size(img_dim, padding, filter_dims):
    v = img_dim[0]+2*padding[0]-filter_dims[0]+1
    h = img_dim[1]+2*padding[1]-filter_dims[1]+1
    return np.array((v, h), dtype=np.int32)

@jit(nopython=disable_debug)
def pad_image_advanced(img, filter_dims):
    img_dims = img.shape
    padding = calculate_same_padding(filter_dims)
    width = int(img_dims[1]+padding[1]*2)
    height = int(img_dims[0]+padding[0]*2)
    new_img = np.zeros((height, width), dtype=np.float64)
    new_img[padding[0]:-padding[0], padding[1]:-padding[1]] = img
    return new_img



@jit(nopython=disable_debug)
def pad_image(img):
    y = img.shape[0]
    x = img.shape[1]
    padded_y = y+2
    padded_x = x+2

    padding_in_progress = np.zeros((padded_y, padded_x), dtype=np.float64)
    padding_in_progress[1:-1, 1:-1] =img

    return padding_in_progress

@jit(nopython=disable_debug, parallel=False)
def match_images_param_search(match, gap, egap, im1, im2, scores_raw, moves_raw, scores_n_moves, disparity, scanline_match_function):
    im1_padded = pad_image(im1)
    im2_padded = pad_image(im2)
    third = int(im1.shape[0]/3)
    for i in prange(0,third):
        #print(i)
        scores_n_moves[0, i, :], scores_n_moves[1, i, :] = \
            match_scanlines_param_search(match, gap, egap, i, im1_padded, im2_padded, np.copy(scores_raw), np.copy(moves_raw))
        temp = generate_disparity_line(im1.shape[1], get_traceback_path(i, scores_n_moves)).astype(np.float64)
        disparity[i] = temp
        #print(i)
    return scores_n_moves[0:third], disparity[0:third]

@jit(nopython=disable_debug)
def match_scanlines_param_search(match, gap, egap, current_index, im2_padded, im1_padded, scores, moves):
    calc_index_value = np.int32(current_index*3+1)
    im1_scanline, im2_scanline = im1_padded[calc_index_value], im2_padded[calc_index_value]
    #print(calc_index_value)
    for i in range(1, im1_scanline.shape[0]-1):
        #print(i)
        for j in range(1, im1_scanline.shape[0]-1):
            im1_index, im2_index= j-1, i-1
            #match_raw = match - abs(im1_scanline[im1_index] - im2_scanline[im2_index])
            #match_raw =  match - abs(im1_scanline[im1_index] - im2_scanline[im2_index]) #get_patch_absolute_difference(current_index, im1_index, im2_index,im1, im2)
            match_raw = match -  get_patch_absolute_difference(calc_index_value-1, im1_index, im2_index,im1_padded, im2_padded)
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
def get_straceback_start(currentIndex, scores_n_moves):
    lastColumn = scores_n_moves[0, currentIndex, :, -1]
    lastRow = scores_n_moves[0, currentIndex, -1, :]
    last_column_max = np.max(lastColumn)
    last_row_max = np.max(lastRow)

    last_column_max_index = np.argmax(lastColumn)
    last_row_max_index = np.argmax(lastRow)



    return np.array([last_column_max_index, scores_n_moves.shape[3]-1]).astype(np.int32) if (last_column_max > last_row_max) else np.array([scores_n_moves.shape[3]-1, last_row_max_index]).astype(np.int32)
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

@jit(nopython=disable_debug)
def pipline(match, gap, egap, im1, im2, scanline_match_function=match_scanlines, filter = np.ones((3,3), dtype=np.int32)):
    scores_raw, moves_raw = initialize_matrix_template(gap, egap, im1)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    disparity = np.zeros(im1.shape, dtype=np.float64)
    x,z = match_images(match, gap, egap, im1, im2, scores_raw, moves_raw, scores_n_moves, disparity, scanline_match_function, filter = filter)
    return x,z

if __name__ == "__main__":
    import cv2

    im1_path = "../../datasets/middlebury/middlebury_2003/cones/im2.png"
    im2_path = "../../datasets/middlebury/middlebury_2003/cones/im6.png"
    im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    #im1 = np.random.randint(0, 255, [2,4])
    #im2 = np.random.randint(0, 255, [2, 4])
    match = 60
    gap = -20
    egap = -1
    scores_raw, moves_raw = initialize_matrix_template(gap, egap, im1)
    scores_n_moves = np.zeros((2, im1.shape[0], im1.shape[1] + 1, im1.shape[1] + 1), dtype=np.float64)
    disparity = np.zeros(im1.shape, dtype=np.float64)
    #test_1, test_2 = match_scanlines(match, gap, egap, 0, im2, im1, scores_raw, moves_raw)


    SimpleTimer.timeit()
    x,z = pipline(60, -20, -1, im2, im1, filter = np.ones((5,5), dtype = np.int32))
    #x,y,z = match_images(80, -30, -2, im2, im1)
    #x,y,z = FakeNumbaClass["match_images"] (100, -15, -5, im2, im1)
    #Wrapper.match_images( im2, im1)

    SimpleTimer.timeit()
    z_mod = z

    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.figure()
    plt.imshow(z_mod, cmap = cm.binary)
