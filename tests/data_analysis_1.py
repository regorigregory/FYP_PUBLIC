import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

#    majority vote
#    weighted sum
#    analysis on cnn first layers
#    we trust the machine
#    it finds the best
#    is it true that all those features are equally good?
#    or some of them are rubbish?
#    finding the weights of those filters is a suboptimal process
#    what combination of filters should we use to get the best results...
#    Examine what deep learning produces...
#    Can we have any conclusion about all the filters that uses
#    parameters based on their x,y coordinates
#    what shape of patch should we use.
#    kitti dataset
#    middlebury dataset
#    path matching -> multiple feature extractions -> how to decide what shape
#    how do others do path matching...
#    some people have recently have very good results
#    more weight in the centre
#    incfrease the meta levels -> they have demonstrated it empirically, therefore you need to trust.
#    streo

def plot3d_results(x_label,y_label,metrix, FILE_PATH_OR_DATAFRAME):
    if(FILE_PATH_OR_DATAFRAME.__class__.__name__ == 'str'):
        data = pd.read_csv(FILE_PATH_OR_DATAFRAME)
        data.columns = np.array([str.strip(col) for col in data.columns])
    else:
        data = FILE_PATH_OR_DATAFRAME


    x = np.array(data[x_label].unique())[:, np.newaxis]
    y = np.array(data[y_label].unique())[:, np.newaxis]
    z = pd.pivot_table(data, values=[metrix], columns=[x_label], index=[y_label]).values
    z = np.nan_to_num(z)
    print("Z's shape is: {0}".format(z.shape))
    x_diff_step = (x.max() - x.min()) / z.shape[1]

    X = np.arange(x.min(), x.max(), x_diff_step)[:, np.newaxis]

    y_diff_step = (y.max()-y.min())/z.shape[0]

    Y = np.arange(y.min(), y.max() , y_diff_step)[:, np.newaxis]
    fig = plt.figure()

    ax = fig.gca(projection='3d')

    ax.plot_surface(X.T, Y, z, cmap=cm.tab20b)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(metrix)
    plt.show()
    row_with_min = data[metrix].idxmin()
    min_row = data.loc[row_with_min]
    return min_row




if __name__=="__main__":
    import os
    FILE_PATH = os.path.join("..", "optimization", "2", "numba_legacy", "legacy_stacked_images_5_256.csv")
    data = pd.read_csv(FILE_PATH)
    data.columns = np.array([str.strip(col) for col in data.columns])
    #data
    data["match_gap_ratio"] = (data["match"] / data["gap"])
    data["match_egap_ratio"] = (data["match"] / data["egap"])
    data["gap_egap_ratio"] = (data["gap"] / data["egap"])

    #data = data.dropna()
    data["match_gap_ratio"] = np.nan_to_num(data["match_gap_ratio"], posinf=100, neginf=-100)
    data["match_egap_ratio"]  = np.nan_to_num(data["match_egap_ratio"], posinf=100, neginf=-100)
    data["gap_egap_ratio"] = np.nan_to_num(data["gap_egap_ratio"], posinf=100, neginf=-100)

    x_label = "match"
    y_label = "match_gap_ratio"
    metrix = "avg_err"

    plot3d_results(x_label, y_label, metrix, data)
    """
    x_label = "match"
    y_label = "match_egap_ratio"
    metrix = "avg_err"

    plot3d_results(x_label, y_label, metrix,  data)

    x_label = "match"
    y_label = "gap_egap_ratio"
    metrix = "avg_err"

    plot3d_results(x_label, y_label, metrix, data)
    x_label = "match"
    y_label = "gap"
    metrix = "avg_err"


    plot3d_results(x_label, y_label, metrix,  data)


    metrix= "bad1"
    plot3d_results(x_label, y_label, metrix,  data)

    metrix= "bad10"
    plot3d_results(x_label, y_label, metrix,  data)

    x_label = "match"
    y_label= "egap"
    metrix= "avg_err"


    plot3d_results(x_label, y_label, metrix,  data)

    metrix= "bad1"
    plot3d_results(x_label, y_label, metrix,  data)

    metrix= "bad10"
    plot3d_results(x_label, y_label, metrix,  data)"""

