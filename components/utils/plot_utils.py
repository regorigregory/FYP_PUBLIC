import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px

import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.collections import PolyCollection

from mpl_toolkits.mplot3d import Axes3D
import math

def plot_disp_line(disp, scanline_index, color, title):
    coords = get_disparity_plot_coords(disp, scanline_index = scanline_index)
    plt.plot(coords)

def get_disparity_plot_coords(disp, scanline_index=0):
    current = next = disp[0,0]
    current_plot_coords = [0,0]
    for j in range ((disp.shape[1])):
        next = disp[scanline_index, j]
        coordinate_diff = get_disparity_scanline_move(current, next)
        current_plot_coords.append(
            (current_plot_coords[-1][0] + coordinate_diff[0],
             current_plot_coords[-1][1] + coordinate_diff[1])
        )
        current = next
    return current_plot_coords

def get_disparity_scanline_move(current, next):
    if next==0:
        return (1,0)
    if(current==next):
        return (1,1)
    #if it is brighter?
    if(next>current):
        return (np.abs(next-current)+1, 1)
    #if it is darker?
    return (1,np.abs(next - current)+1)


def scatter_3d_results(x_label, y_label, metrix, FILE_PATH_OR_DATAFRAME, cmm="viridis"):
    if(FILE_PATH_OR_DATAFRAME.__class__.__name__ == 'str'):
        data = pd.read_csv(FILE_PATH_OR_DATAFRAME)
        data.columns = np.array([str.strip(col) for col in data.columns])
    else:
        data = FILE_PATH_OR_DATAFRAME

    x,y,z = data[x_label], data[y_label], data[metrix]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(metrix)
    ax.scatter(x,y, z, c=z, cmap = "viridis")
    plt.show()

def polyine_3d(x_label, y_label, metrix, data, occl_counted=False):

    scenes = data["scene"].unique()

    X = np.array(data[x_label].unique())

    Y= np.array(data[y_label].unique())

    data = data[data["are_occlusions_errors"] == occl_counted]
    data = data.sort_values(by=[x_label, y_label])



    z = pd.pivot_table(data, values=[metrix], columns=[x_label], index=[y_label]).values
    Z = np.nan_to_num(z, nan=2000)


    verts = []
    mins = []
    for i in range(X.shape[0]):
        current_column = Z[:, i]
        min_loc, min_val = X[np.argmin(current_column)], current_column.min()
        temp = list(zip(Y, current_column))
        verts.append(temp)
        mins.append((min_loc, min_val, X[i]))

    stop_here = 1

    poly = PolyCollection(verts, facecolors=[get_random_color() for x in X])
    poly.set_alpha(1)


    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    surf_x, surf_y = np.meshgrid(Y, X)
    surf_z = np.empty((len(X), len(Y)))
    surf_z[:, :] = data[metrix].min()
    ax.plot_surface(surf_x, surf_y, surf_z, color=(0.3,0.3,0.9, 0.6))

    ax.add_collection3d(poly, zs = X, zdir='y')

    #annotating minimums for enhanced readibility

    for x,z,y in mins:
        label = 'min: %.2f (%d, %d)' % (z, x, y)
        ax.text(x, y, z, label)


    ax.set_xlabel(y_label)
    ax.set_xlim3d(0, Y[-1])
    ax.set_ylabel(x_label)
    ax.set_ylim3d(0, X[-1])
    ax.set_zlabel(metrix)
    ax.set_zlim3d(0, 2000)
    plt.grid()
    plt.show()

    return fig, ax

def bar_3d_by_scenes(x_label, y_label, metrix, FILE_PATH_OR_DATAFRAME, occl_counted=False):
    if (FILE_PATH_OR_DATAFRAME.__class__.__name__ == 'str'):
        data = pd.read_csv(FILE_PATH_OR_DATAFRAME)
        data.columns = np.array([str.strip(col) for col in data.columns])
    else:
        data = FILE_PATH_OR_DATAFRAME

    scenes = data["scene"].unique()

    data = data[data["are_occlusions_errors"]==occl_counted]
    data = data.sort_values(by=[x_label, y_label])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = [get_random_color() for scene in scenes]

    for x_param in data[x_label].unique():
        temp_outer = data[data[x_label]==x_param]
        for i, scene in enumerate(scenes):
            temp_inner = temp_outer[(temp_outer["scene"]==scene)]
            ax.bar(temp_inner[x_label], temp_inner[metrix], zs=temp_inner[y_label], zdir="y", color=colors[i], alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(metrix)
    plt.show()

def get_random_color(alpha=0.8):
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    return (r,g,b, alpha)

def plot_3d_results(x_label,y_label,metrix, FILE_PATH_OR_DATAFRAME, steps=None):
    if(FILE_PATH_OR_DATAFRAME.__class__.__name__ == 'str'):
        data = pd.read_csv(FILE_PATH_OR_DATAFRAME)
        data.columns = np.array([str.strip(col) for col in data.columns])
    else:
        data = FILE_PATH_OR_DATAFRAME


    x = np.array(data[x_label].unique())[:, np.newaxis]
    y = np.array(data[y_label].unique())[:, np.newaxis]
    z = pd.pivot_table(data, values=[metrix], columns=[x_label], index=[y_label]).values
    z = np.nan_to_num(z, nan=1000000)
    print("Z's shape is: {0}".format(z.shape))
    x_diff_step = (x.max() - x.min()) / z.shape[1] if steps is None else steps[1]
    y_diff_step = (y.max()-y.min())/z.shape[0] if steps is None else steps[0]

    X = np.arange(x.min(), x.max()+1, x_diff_step)[:, np.newaxis]


    Y = np.arange(y.min(), y.max() +1, y_diff_step)[:, np.newaxis]
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_zlim(data[metrix].min()-100, data[metrix].max()+100)

    ax.plot_surface(X, Y, z, cmap=cm.tab20b)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(metrix)
    plt.show()
    row_with_min = data[metrix].idxmin()
    min_row = data.loc[row_with_min]
    return min_row

#it is 4d in reality
def plotly_4d_results(x_label,y_label, z_label, metrix, FILE_PATH_OR_DATAFRAME, ):
    if(FILE_PATH_OR_DATAFRAME.__class__.__name__ == 'str'):
        data = pd.read_csv(FILE_PATH_OR_DATAFRAME)
        data.columns = np.array([str.strip(col) for col in data.columns])
    else:
        data = FILE_PATH_OR_DATAFRAME

    data = data[[x_label, y_label, z_label, metrix]]
    data = data.sort_values(by=[x_label, y_label, z_label], ascending = True)
    values = np.nan_to_num(data[[metrix]].values)

    x = np.array(data[x_label].unique())[:, np.newaxis]
    y = np.array(data[y_label].unique())[:, np.newaxis]
    z = np.array(data[z_label].unique())[:, np.newaxis]

    x_step = (x.max() - x.min()) / (x.shape[0]-1)

    y_step = (y.max() - y.min()) / (y.shape[0]-1)
    z_step =(z.max() - z.min()) / (z.shape[0]-1)
    # good cmap = px.colors.diverging.Spectral
    # px.colors.sequential.Rainbow
    # px.colors.sequential.Angset

    X, Y, Z = np.mgrid[x.min():x.max()+x_step:x_step, y.min():y.max()+y_step:y_step, z.min():z.max()+z_step:z_step]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        cmin=values.min(),
        cmax=values.max(),
        opacity=0.3,  # needs to be small to see through all surfaces
        surface_count=20,
        colorscale=px.colors.diverging.Spectral
    ))
    fig.show()


def plot_disparity_3d(disparity, cmm = cm.viridis):
    x = np.arange(0, disparity.shape[0])[:, np.newaxis]
    y = np.arange(0, disparity.shape[1])[:, np.newaxis]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x,y.T, disparity, cmap = cmm)

def plot_images(imgs, titles, cmode = "gray", ncols= 4, hspace=0.5, wspace=0.5):
    assert len(imgs) == len(titles)
    n = len(imgs)
    row_number = math.ceil(n / ncols)
    fig = plt.subplots(figsize=[20, int(4*row_number)])
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    for i, img in enumerate(imgs):
        ax = plt.subplot(row_number, ncols, i + 1)
        ax.set_title("%s\n (%dx%d)" % (titles[i], img.shape[1], img.shape[0]))
        plt.imshow(img, cmode)
    return fig
if __name__ == "__main__":
    import sys
    import os
    from components.utils import middlebury_utils as mbu
    import project_helpers
    sys.path.append(os.path.join("..", ".."))

    ROOT_PATH = project_helpers.get_project_dir()
    EXPERIMENT_TITLE = "EXP_000-Baseline"

    DATASET_FOLDER = os.path.join(ROOT_PATH, "datasets", "middlebury")
    LOG_FOLDER = os.path.join(ROOT_PATH, "experiments", "logs")
    CSV_FOLDER = os.path.join(LOG_FOLDER, EXPERIMENT_TITLE + ".csv")
    SCENES = ["teddy", "cones"]
    YEAR = 2003
    loaded_imgs_and_paths = list(mbu.get_images(DATASET_FOLDER, YEAR, scene) for scene in SCENES)
    """for im, path in loaded_imgs_and_paths:
        plot_images(im, path)
    import os
    
    ROOT = os.path.join("..", "..")
    
    SELECTED_DATASET = "middlebury_2003"
    SELECTED_SCENE = "teddy"
    SELECTED_IMAGE = "2-6"

    IMG_LOAD_PATH = os.path.join(ROOT, "datasets", "middlebury", SELECTED_DATASET, SELECTED_SCENE)

    left = cv2.imread(os.path.join(IMG_LOAD_PATH, "im2.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(os.path.join(IMG_LOAD_PATH, "im6.png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(IMG_LOAD_PATH, "disp2.png"), cv2.IMREAD_GRAYSCALE)
    occl = cv2.imread(os.path.join(IMG_LOAD_PATH, "teddy_occl.png"), cv2.IMREAD_GRAYSCALE)
    coloured_left = cv2.imread(os.path.join(IMG_LOAD_PATH, "im2.png"))
    plod3d_with_img_surface(gt, surface = coloured_left, finess=5, rotation=True)"""
    """
    ========================================
    Create 2D bar graphs in different planes
    ========================================

    Demonstrates making a 3D plot which has 2D bar graphs projected onto
    planes y=0, y=1, etc.
    """

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.random.rand(20)

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



    p = "../../experiments/logs/ALG_005_EXP_001-PatchMatch-MacLean_et_al-Numba.csv"
    data = pd.read_csv(p)
    new_cols = np.array(data["kernel_size"].str.split("x").to_list()).astype(np.int8)
    data["k_h"], data["k_w"] = new_cols[:, 0], new_cols[:, 1]
    #selected_scene = data[(data["scene"]=="cones") & (data["are_occlusions_errors"]==False)]
    selected_scene = data

    # todo: print a scene by occlusion subplot figure

    #bar_3d_by_scenes("k_h", "k_w", "mse", selected_scene)
    polyine_3d("k_h", "k_w", "mse", selected_scene)
    min = data[data["mse"] == data["mse"].min()]
    print(min[["kernel_size", "mse"]])
