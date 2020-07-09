import pandas as pd
import ipywidgets as widgets
import numpy as np
import os
import plotly.graph_objs as go
import plotly.express as px
from ipywidgets import HBox, VBox, Button

# loading df, disparities and ground truth as well...

def load_n_clean(path_to_dataframe, gts=True, kernel_sizes=True):
    df = pd.read_csv(os.path.abspath(path_to_dataframe))
    df = df.drop_duplicates()
    df["loaded_imgs"] = [read_image_binary(path) for path in df["image_filename"]]
    if(gts):
        df["loaded_gts"] = [read_image_binary(os.path.join(os.path.dirname(path), "disp0GT.png")) for path in df["image_filename"]]
    if(kernel_sizes):
        h_n_w = np.array(df["kernel_size"].str.split("x").to_list())
        df["h"] = pd.to_numeric(h_n_w[:, 0])
        df["w"] = pd.to_numeric(h_n_w[:, 1])
        selected_scene_df = df.sort_values(by=["h", "w"])
    return df

# helper function as image widget needs this format...

def read_image_binary(path):
    with open(path, "rb") as f:
        b = f.read()
        return b

# getting a wrapped scatter plot. Optional trace dimension in colours

def get_figure_widget(df, x_label, y_label, title, use_color=False, color_dim=None):
    if (not use_color):
        fig = go.FigureWidget(px.scatter(df, x=x_label,
                                         y=y_label, hover_data=["match", "gap", "egap", "kernel_size", "kernel_spec",
                                                                "preprocessing_method", "scene"]))
    else:
        fig = go.FigureWidget(px.scatter(df, x=x_label,
                                         y=y_label, hover_data=["match", "gap", "egap",
                                                                "kernel_size", "kernel_spec", "preprocessing_method",
                                                                "scene"], color=color_dim
                                         ))

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.layout.title = title
    fig.data[0].marker.opacity = np.full(fig.data[0].x.shape, 0.5)
    return fig

# default hover function

def get_hover_function(img_widget, selected_scene_df):
    def hover_fn(trace, points, state):
        if (len(points.point_inds) > 0):
            ind = points.point_inds[0]
            img_widget.value = selected_scene_df["loaded_imgs"].iloc[ind]

    return hover_fn

# for multiple traces
# needs an array of dataframes for each trace....

def get_hover_function2(img_widget, dfs):
    def hover_fn(trace, points, state):
        # print(points)
        if (len(points.point_inds) > 0):
            ind = points.point_inds[0]
            img_widget.value = dfs[points.trace_index]["loaded_imgs"].iloc[ind]

    return hover_fn

# for displaying gt as well in a third widget...

def get_hover_function3(img_widget_disparity,img_widget_groundtruth,  dfs):
    def hover_fn(trace, points, state):
        # print(points)
        if (len(points.point_inds) > 0):
            ind = points.point_inds[0]
            img_widget_disparity.value = dfs[points.trace_index]["loaded_imgs"].iloc[ind]
            img_widget_groundtruth.value = dfs[points.trace_index]["loaded_gts"].iloc[ind]

    return hover_fn

# Binding the hover function for each fig and their traces
# However, won't work well with multiple traces as their indices would differ...
# Keeping it here so that won't break anything that is ready...

def bind_hover_function(figs, img_widget, selected_scene_df):
    for fig in figs:
        for trace in fig.data:
            trace.on_hover(get_hover_function(img_widget, selected_scene_df))



# for displaying gt as well in a third widget...

def bind_hover_function2(figs, img_widget, selected_scene_df, img_widget_groundtruth=False):
    for fig in figs:
        for trace in fig.data:
            if(img_widget_groundtruth):
                trace.on_hover(get_hover_function3(img_widget, img_widget_groundtruth, selected_scene_df))

            else:
                trace.on_hover(get_hover_function2(img_widget, selected_scene_df))

# alias. Title above is not very informative...

def bind_hover_function_for_traced(figs, img_widget, selected_scene_df, img_widget_groundtruth=False):
    bind_hover_function2(figs, img_widget, selected_scene_df, img_widget_groundtruth=img_widget_groundtruth)

def get_figure_traced(df, x_label, y_label, trace_dim, discrete_hover = False):
    trace_keys = df[trace_dim].unique()
    fig = go.Figure()
    dfs = []

    for k in trace_keys:
        temp_df = df[df[trace_dim] == k]
        dfs.append(temp_df)

        if(not discrete_hover):
            label = ["Experiment id: {4}<br>Scene: {0}<br>kernel size: {1}<br>match:{2}<br>bad4: {3}<br>" \
                     "IMG res:{5}".format(a, b, c, d, e, f)
                     for a, b, c, d, e,f in zip(temp_df["scene"], temp_df["kernel_size"], temp_df["match"], temp_df["bad4"],
                                           temp_df["experiment_id"], temp_df["img_res"])
                     ]
        else:
            label = ["Scene:{0}<br>match:{1}<br>bad4: {2}<br>" \
                         .format(a, b, c)
                     for a, b, c in
                     zip(temp_df["scene"], temp_df["match"], temp_df["bad4"]
                         )
                     ]
        fig.add_trace(go.Scatter(x=temp_df[x_label], y=temp_df[y_label], name=str(k), hovertemplate=
        '%{text}', text=label, showlegend=True))

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.update_layout(legend_title_text=trace_dim)

    return fig, dfs

def get_figure_widget_traced(df, x_label, y_label, trace_dim, discrete_hover = False):
    fig, dfs = get_figure_traced(df, x_label, y_label, trace_dim, discrete_hover = discrete_hover)
    return go.FigureWidget(fig), dfs

def get_brush_function(master_n_slaves, selected_scene_df):
    def brush_fn(trace, points, state):
        inds = np.array(points.point_inds)
        if inds.size:
            opacity = np.full(selected_scene_df.shape[0], 0.1)
            color = np.zeros(selected_scene_df.shape[0])
            color[inds] = 1
            opacity[inds] = 1

            for i, fig in enumerate(master_n_slaves):
                with fig.batch_update():
                    fig.data[0].marker.opacity = opacity/(i+1)
                    fig.data[0].marker.color = color


    return brush_fn


def bind_brush_function(master_n_slaves, selected_scene_df):
    func = get_brush_function(master_n_slaves, selected_scene_df)
    for trace in master_n_slaves[0].data:
        trace.on_selection(func)


def get_reset_brush_function(master_n_slaves):
    def reset_fn(btn):
        selected = np.full(len(master_n_slaves[0].data[0].x), 1)
        for fig in master_n_slaves:
            with fig.batch_update():
                fig.data[0].marker.opacity = selected
                fig.data[0].marker.color = selected

    return reset_fn


def get_reset_brush_button(master_n_slaves, description_passed="clear selection"):
    btn = Button(description=description_passed)
    reset_fn = get_reset_brush_function(master_n_slaves)
    btn.on_click(reset_fn)
    return btn


def get_dropdown_function(df, column, widget, figure_widget):
    def validate():
        return True
        """        if widget.value in df[column].unique():
            return True
        else:
            return False"""

    def response(change):
        if validate():
            temp_df = df[df[column] == widget.value]
            x = temp_df['kernel_size']
            y = temp_df['match']
            with figure_widget.batch_update():
                figure_widget.data[0].x = x
                figure_widget.data[0].y = y

                print("I should have been updated.")

    return response


def bind_dropdown_function(df, column, dropdown_widget, figure_widget):
    dropdown_fn = get_dropdown_function(df, column, dropdown_widget, figure_widget)
    dropdown_widget.observe(dropdown_fn, names="value")

def show_traces_legendonly(traced_fig_widget_1):
    with traced_fig_widget_1.batch_update():
        traced_fig_widget_1.for_each_trace(
            lambda trace: trace.update(visible="legendonly")
        )
def show_traces_everywhere(traced_fig_widget_1):
    with traced_fig_widget_1.batch_update():
        traced_fig_widget_1.for_each_trace(
            lambda trace: trace.update(visible=True)
        )

def get_dropdown_switch_traces_fn(traced_fig_widget_1):

    def fn(change):
        #print(change)
        if(not change.new):
            with traced_fig_widget_1.batch_update():
                traced_fig_widget_1.for_each_trace(
                    lambda trace: trace.update(visible="legendonly")
                )
        else:
            with traced_fig_widget_1.batch_update():
                traced_fig_widget_1.for_each_trace(
                    lambda trace: trace.update(visible=True))

    return fn

def bind_dropdown_switch_traces_fn(dropdown_widget, traced_fig_widget_1):
    fn = get_dropdown_switch_traces_fn(traced_fig_widget_1)
    dropdown_widget.observe(fn, names="value")



def get_dropdown_widget(options, label = "Please select", values = False):
    dropdown_widget = None
    if(values):
        dropdown_widget = widgets.Dropdown(
            description=label,
            options=[(o,v) for o,v in zip(options, values)]
        )
    else:
        dropdown_widget = widgets.Dropdown(
            description=label,
            options=options
        )
    return dropdown_widget


def get_dropdown_widget_from_df(df, column):
    dropdown_widget = widgets.Dropdown(
        description=column,
        options=df[column].unique()
    )
    return dropdown_widget




