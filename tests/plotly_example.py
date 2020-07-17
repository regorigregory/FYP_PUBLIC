import plotly.graph_objects as go
import numpy as np
import pandas as pd

if __name__ == "__main__":
    print("Plotly uses the default browser interface, therefore it should open a browser window.")
    X, Y, Z = np.mgrid[-8:8:100j, -8:8:100j, -5:5:100j]

    values = np.sin(X*Y*Z) / (X*Y*Z)

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        ))
    fig.show()