import numpy as np
import unittest
import cv2
import random
import logging
import os
logger = logging.getLogger('padding_addition')
logger.setLevel(logging.DEBUG)

def calculate_same_padding(filter_dims):
    #based on: output_dim = (n+2*p-f+1)**2 if they are symmetrical
    vpadding = (filter_dims[0]-1)/2
    hpadding = (filter_dims[1]-1)/2
    return np.array([vpadding, hpadding], dtype=np.int64)

def get_output_size(img_dim, padding, filter_dims):
    v = img_dim[0]+2*padding[0]-filter_dims[0]+1
    h = img_dim[1]+2*padding[1]-filter_dims[1]+1
    return np.array([v, h], dtype=np.int64)

def add_padding(img, filter_dims):

    img_dims = img.shape
    padding = calculate_same_padding(filter_dims)
    new_img = np.zeros([img_dims[0]+padding[0]*2, img_dims[1]+padding[1]*2])
    new_img[padding[0]:-padding[0], padding[1]:-padding[1]] = img
    return new_img

class SamePaddingTests(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_same_padding_calc(self):
        for i in range(10000):
            #odd numbers only
            filter_dims = np.array([random.randrange(3, 100, 2), random.randrange(3, 100, 2)])
            img_dims = np.random.randint(300, 1000, 2)
            padding = calculate_same_padding(filter_dims)
            check_img_dims = get_output_size(img_dims, padding, filter_dims)
            message = "Mismatch in input and output dimensions. Filter size: {0}".format(str(filter_dims))
            self.assertEqual(img_dims[0], check_img_dims[0], message)
            self.assertEqual(img_dims[1], check_img_dims[1], message)

    #@unittest.skip("demonstrating skipping")
    def test_img_dims(self):
        PATH = os.path.join("..", "datasets", "middlebury", "middlebury_2003", "teddy", "im2.png")
        test_img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
        test_img_dims = test_img.shape
        for i in range(10000):
            # odd numbers only
            filter_dims = np.array([random.randrange(3, 100, 2), random.randrange(3, 100, 2)])

            padding = calculate_same_padding(filter_dims)

            expected_img_dims = test_img_dims + padding * 2
            new_img = add_padding(test_img, filter_dims)
            new_img_dims = new_img.shape

            # checks below
            message = "Mismatch in input and output dimensions. Filter size: {0}".format(str(filter_dims))


            self.assertEqual(expected_img_dims[0], new_img_dims[0], message)
            self.assertEqual(expected_img_dims[1], new_img_dims[1], message)

if __name__ == "__main__":
    unittest.main()
    PATH = os.path.join("..", "datasets", "middlebury", "middlebury_2003", "teddy", "im2.png")
    test_img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    filter_dims = np.array([random.randrange(3, 100, 2), random.randrange(3, 100, 2)])
    new_img = add_padding(test_img, filter_dims)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(new_img, cmap="gray")

