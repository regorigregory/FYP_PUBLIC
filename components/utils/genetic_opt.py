def get_me_random_slices(left, right, gt, nonocc, amount_of_slices = 10, start_y=0, end_y=200):

    sliced = np.zeros([4, amount_of_slices*3, int(end_y-start_y)])
    num_rows = np.arange(left.shape[0]-3)
    np.random.seed(5
                   )
    selected_rows = np.random.choice(num_rows, amount_of_slices)

    all = np.array([left, right, gt, nonocc])

    for i, start_row in enumerate(selected_rows):
        end_row = start_row+3
        temp = all[:, start_row:end_row, start_y:end_y]
        sliced[:, int(i*3):int(i*3+3), :] = temp
    return sliced

def get_me_random_slices_ordered(left, right, gt, nonocc, row_height = 30, start_y=0, end_y=450):

    random_rowstart = np.random.randint(0, int(left.shape[0]-row_height))
    all = np.array([left, right, gt, nonocc])
    return all[:, random_rowstart:int(random_rowstart+row_height), start_y:end_y]

if __name__ == "__main__":
    import numpy as np
    from geneticalgorithm import geneticalgorithm as ga
    import numpy as np
    import matplotlib.pyplot as plt
    from components.numba_functions import temp as TAD_G
    import pickle
    import cv2
    import os
    from components.utils.Metrix import Wrapper as m

    scaler = 256
    scene = "teddy"
    im1_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im2.png")
    im2_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "im6.png")
    gt_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "disp2.png")
    occ_path = os.path.join("..", "..", "datasets", "middlebury", "middlebury_2003", scene, "nonocc.png")

    im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)[30:-15, :]/ scaler
    im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)[30:-15, :]/ scaler
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)[30:-15, :]
    occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)[30:-15, :]

    sliced_images = [im1, im2, gt, occ]

    filter = np.ones((3, 3), dtype=np.float64)

    BEST_BAD = 10000000
    best_x=None
    COUNTER = 1
    pop_size=75
    max_num_iteration=30
    #sliced_images = get_me_random_slices(im1, im2, gt, occ)
    #sliced_images = get_me_random_slices_ordered(im1, im2, gt, occ)

    algorithm_params = {
        'max_num_iteration': max_num_iteration, \
        'population_size': pop_size, \
        'mutation_probability': 0.17, \
        'elit_ratio': 0.2, \
        'crossover_probability': 0.5, \
        'parents_portion': 0.4, \
        'crossover_type': 'uniform', \
        'max_iteration_without_improv': 4}

    varbound = np.array([[5, 40], [-20, -1], [-20, -1], [0.05, 20], [0.1, 20], [0.3, 0.9], [0.01, 0.5], [0.001, 0.1]])


    def f(x):
        global COUNTER
        global BEST_BAD
        global best_x
        global sliced_images
        global pop_size
        x = x.astype(np.float64)
        match = x[0]/ scaler
        gap = x[1] / scaler
        egap = x[2] / scaler

        x, z = TAD_G.test_pipeline(match, gap, egap,
                                   sliced_images[0],
                                   sliced_images[1],
                                   filter=filter,
                                   gamma_c=x[3],
                                   gamma_s=x[4],
                                   alpha=x[5],
                                   T_c = x[6],
                                   T_s = x[7])

        #EUC = m.eucledian_distance(z * 4, sliced_images[2], sliced_images[3])
        BAD1 = m.bad(z * 4, sliced_images[2], sliced_images[3], threshold=1)
        if BAD1<BEST_BAD:
            BEST_BAD = BAD1
            best_x = x
        COUNTER+=1
        if(COUNTER%pop_size==0):
            print("\nBest BAD1 so far: {0:.6f}".format(BEST_BAD))
            #sliced_images = get_me_random_slices_ordered(im1, im2, gt, occ)
        return BAD1



    model=ga(function=f,
             dimension=8,
             variable_type='real',
             variable_boundaries=varbound,
             algorithm_parameters = algorithm_params,
             function_timeout= 100000
             )

    results = model.run()
    """
    with open("best_results.pickle", "wb") as f:
        pickle.dump(results, f)
    print(results)
    #"""
