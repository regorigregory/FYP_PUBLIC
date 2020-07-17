import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px

def plotly_3d_results(x_label,y_label, z_label, metrix, FILE_PATH_OR_DATAFRAME):
    if(FILE_PATH_OR_DATAFRAME.__class__.__name__ == 'str'):
        data = pd.read_csv(FILE_PATH)
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


    X, Y, Z = np.mgrid[x.min():x.max()+x_step:x_step, y.min():y.max()+y_step:y_step, z.min():z.max()+z_step:z_step]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        cmin=values.min(),
        cmax=values.min()+3,
        opacity=0.3,  # needs to be small to see through all surfaces
        surface_count=20,
        colorscale=px.colors.sequential.Viridis_r
    ))
    fig.show()







if __name__ == "__main__":
    import os
    FILE_PATH = os.path.join("..", "optimization", "2", "numba_legacy", "legacy_stacked_images_5_256.csv")
    print("Does the specified file exist: {0}".format(os.path.isfile(FILE_PATH)))

    #"C:/gdrive/python_projects/FYP/test_outputs/numba_sm/3/numba_sm.csv"
    data = pd.read_csv(FILE_PATH)
    data.columns = np.array([str.strip(col) for col in data.columns])
    # data

    x_label = "match"
    y_label = "gap"
    z_label = "egap"
    metrix =  "avg_err"

    plotly_3d_results(x_label,y_label, z_label, metrix, data)
