import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def show_histogram(img, title="test"):
    _ = plt.hist(img, range=(0,255))
    plt.title(title)
    plt.show()

def compare_histograms(img, gt, titles=["disparity", "groundtruth"]):
    plt.subplots(1,2)
    ax = plt.subplot(121)
    _ = plt.hist(img, range=(0,255), label=titles[0], color=["red" for i in range(img.shape[1])])
    plt.title(titles[0])
    ax = plt.subplot(122)
    _ = plt.hist(gt, range=(0,255), label=titles[1], color=["green" for i in range(img.shape[1])])
    plt.title(titles[1])
    pass

def show_hit_and_miss(disp,gt, titles=["hit", "miss"], threshold=1):
    c = _get_correct_pixels(gt, disp, threshold=threshold)
    ic = _get_incorrect_pixels(gt, disp, threshold=threshold)
    plt.subplots(1,2)
    ax = plt.subplot(121)
    ax.set_title(titles[0])
    plt.imshow(c, cmap="gray")
    ax = plt.subplot(122)
    ax.set_title(titles[1])
    plt.imshow(ic, cmap="gray")

def show_difference(disp, gt,  cmm = "Accent"):
    diff = _get_difference(gt, disp)
    x = np.arange(0, disp.shape[0])[:, np.newaxis]
    y = np.arange(0, disp.shape[1])[:, np.newaxis]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x, y.T, diff, cmap=cmm)

def _get_difference(gt, disp):
    differences = disp-gt
    return differences


def _get_correct_pixels(gt, disp, threshold=1):
    mask = np.abs(gt - disp)<=threshold
    return mask


def _get_incorrect_pixels(gt, disp, threshold=1):
    mask = np.abs(gt - disp)>threshold
    return mask


def get_discrepancies(gt, disp, img=None, correct_ones=False):
    m = _get_correct_pixels(gt, disp) if correct_ones else _get_incorrect_pixels(gt, disp)
    visualisation = np.where(m, img, 0) if img is not None else m
    return visualisation


def plod3d_with_img_surface(disp, finess=5, surface=None, rotation=False, surface_color_converter = cv2.COLOR_BGR2RGB):
    x = np.arange(0, disp.shape[0])[:, np.newaxis]
    y = np.arange(0, disp.shape[1])[:, np.newaxis]
    fig = plt.figure()
    if (surface is not None):
        facecolors_param = cv2.cvtColor(surface, surface_color_converter) / 255
    else:
        facecolors_param = "blue"

    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(100, 0)
    ax.plot_surface(x, y.T, disp, rstride=finess, cstride=finess, facecolors=facecolors_param)

    if rotation:
        for angle in range(0, 360):
            ax.view_init(130, angle)
            plt.draw()
            plt.pause(.0000001)


if __name__ == "__main__":
    from pathlib import Path
    import os
    import project_helpers as ph
    test_img = cv2.imread(os.path.join(ph.get_project_dir(), Path("/datasets/middlebury/middlebury_2003/teddy/im2.png")), cv2.IMREAD_GRAYSCALE)
    # show_incorrect_pixels(test_img, test_img)
