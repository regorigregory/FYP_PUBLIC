import os
import cv2
import numpy as np
import re
from components.utils.CSVWriter2 import Wrapper as csv

def get_last_saved_image_id(path):
    paths = os.scandir(path)
    paths_stringified = sorted([p.name for p in paths])
    if(len(paths_stringified))==0:
        return 1

    temp = paths_stringified[-1].split(".")[0][-4:]
    return int(temp)+1

def save_disparity(path, disp):
    experiment_title = os.path.split(path)[-1]
    if(not os.path.isdir(path)):
        os.makedirs(path)
    counter = get_last_saved_image_id(path)
    filename = "{exp_title}_{counter:04d}.png".format(exp_title=experiment_title, counter=counter)
    fqn = os.path.join(path, filename)
    cv2.imwrite(fqn, disp)
    return fqn

def getNextFileName(test_folder ="./test_outputs", image_extension=".png", pre="test_disparity"):
    counter = 1
    full_filename = pre + str(counter) + image_extension

    while os.path.isfile(os.path.join(test_folder, full_filename)):
        counter += 1
        full_filename = pre + str(counter) + image_extension
    return os.path.join(test_folder, full_filename)

def executeParallelMatching(initializedMatcher):
        result = initializedMatcher.alignImagesParallel()
        initializedMatcher.recompileObject(result)
        initializedMatcher.generateDisparity()
        return initializedMatcher

def get_output_filename(specs):
    separator = "&"
    kv_separator = "="
    fname =  "{img_name}_"\
              "is_img_p{kv}{is_img_preprocessed}{sep}" \
              "alg_t{kv}{alg_type}{sep}" \
              "is_p{kv}{is_parallel}{sep}" \
              "m{kv}{match}{sep}" \
              "g{kv}{gap}{sep}" \
              "eg{kv}{egap}{sep}" \
              "mim{kv}{matrix_init_mode}{sep}" \
              "cfil{kv}{convolution_filters}{sep}" \
              "fs{kv}{filter_strategy}{sep}" \
              "mm{kv}{matching_mode}{sep}"\
              "rt{kv}{runtime}{sep}" \
              "b1{kv}{bad1:.2f}{sep}b1.5{kv}{bad15:.2f}{sep}b2{kv}{bad2:.2f}{sep}b10{kv}{bad10:.2f}{sep}avg_e{kv}{avg_err:.2f}"\
        .format(
        is_parallel=specs["is_parallel"],
        img_name=specs["img_name"],
        alg_type=specs["alg_type"],
        is_img_preprocessed=specs["is_img_preprocessed"],
        convolution_filters=specs["convolution_filters"],
        filter_strategy=specs["filter_strategy"],
        matching_mode=specs["matching_mode"],
        match=specs["match"],
        gap=specs["gap"],
        egap=specs["egap"],
        matrix_init_mode=specs["matrix_init_mode"],
        runtime=specs["runtime"],
        bad1=specs["bad1"],
        bad15=specs["bad15"],
        bad2=specs["bad2"],
        bad10=specs["bad10"],
        avg_err=specs["avg_err"],
        sep=separator,
        kv = kv_separator
    )
    fname = re.sub("\.", "_", fname)+"."
    fname += specs["ext"]

    return fname

def saveTwoImages(img1, img2, test_folder ="./test_outputs", image_extension=".png", pre="test_disparity"):
    fname = getNextFileName(test_folder , image_extension, pre)
    success1 = cv2.imwrite(fname, img1)
    if(success1):
        print("File has been successfully written to path: %s"%(fname))
    fname = getNextFileName(test_folder , image_extension, pre)
    success2 = cv2.imwrite(fname, img2)
    if (success2):
        print("File has been successfully written to path: %s"%(fname))
    return success1 and success2

# Adapted from: https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

def gaussian_kernel(dimension_x, dimension_y, sigma_x, sigma_y):
    x = cv2.getGaussianKernel(dimension_x, sigma_x)
    y = cv2.getGaussianKernel(dimension_y, sigma_y)
    kernel = x.dot(y.T)
    return kernel

def getHorizontalFeatureFilter(convolver):
    filter = np.zeros([3,3])
    filter[0, :] =1
    filter[2, :] = -1
    convolver.setFilter(filter)

def getVerticalFeatureFilter(convolver):
    filter = np.zeros([3, 3])
    filter[:, 0] = 1
    filter[:, 2] = -1
    convolver.setFilter(filter)

def getFilterByTypo(convolver):
    filter = np.zeros([3, 3])
    filter[:, 0] = 1
    filter[2, :] = -1
    convolver.setFilter(filter)
def add_occlusions(img, occlusions):
    masked = np.where(occlusions==0, 0, img)
    return masked


def apply_demo_filters(loaded_imgs):
    from components.utils import SimpleConvolution as SC

    convolver = SC.getOne()
    im2 = loaded_imgs[0]
    im6 = loaded_imgs[1]

    im2_blurred = convolver.convolve(im2)
    im6_blurred = convolver.convolve(im6)

    getHorizontalFeatureFilter(convolver)


    im2_h = convolver.convolve(im2)
    im6_h = convolver.convolve(im6)

    getVerticalFeatureFilter(convolver)

    im2_v = convolver.convolve(im2)
    im6_v = convolver.convolve(im6)

    getFilterByTypo(convolver)

    im2_t = convolver.convolve(im2)
    im6_t = convolver.convolve(im6)

    im2_features_added = im2 + im2 + im2_h + im2_t
    im6_features_added = im6 + im6 + im6_h + im6_t

    im2s = [im2, im2_blurred, im2_h, im2_v, im2_t, im2_features_added]
    im6s = [im6, im6_blurred, im6_h, im6_v, im6_t, im6_features_added]

    return im2s, im6s
