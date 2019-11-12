###########################
##
## Plot the output of the optimisation routine
##
###########################

#

import numpy as np
import json_tricks
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.colors
from plotly.subplots import make_subplots
import glob
import dash_ui as dui

from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

import re


outputs = {}

def load_all():
    filenames = glob.glob('output/*.json')
    if len(filenames) == 0:
        raise Exception("No files match the given pattern.")

    p_experiment = re.compile('.*(experiment_[a-zA-Z0-9-]+)_')

    outputs.clear()
    for filename in filenames:
        # First get experiment name
        m = re.match(p_experiment, filename)
        experiment = m.groups()[0] if m else 'default'
        if experiment not in outputs:
            outputs[experiment] = []

        i = len(outputs[experiment])
        with open(filename) as fh:
            o = json_tricks.load(fh, preserve_order=False)
            
        outputs[experiment].append(o)

    return

load_all()




external_stylesheets = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Optimal carbon price path - dashboard"
app.css.config.serve_locally = False


controlpanel = dui.ControlPanel(_id="controlpanel")
controlpanel.create_section(
    section="ExperimentSection",
    section_title=""
)
controlpanel.create_group(
    group="ExperimentGroup",
    group_title="Choose experiments"
)
controlpanel.create_group(
    group="RefreshGroup",
    group_title=""
)
experiment_select = dcc.Dropdown(
    id="experiment-dropdown",
    options=[{
        'label': x,
        'value': x
        } for x in list(outputs.keys())
    ],
    clearable=False,
    multi=True,
    value=list(outputs.keys())[0]
)

def get_year_values():
    try:
        first = outputs[list(outputs.keys())[0]]
        return first[0]['meta']['t_values_years']
    except:
        return []

def get_param_values():
    try:
        first = outputs[list(outputs.keys())[0]]
        return list(first[0]['meta']['params'].default_params.keys())
    except:
        return []

def get_axis_options():
    options = [['temp', -1, '2100']]
    for variable in ['p', 'E']:
        for i, y in enumerate(get_year_values()):
            options.append([variable, i, '{:.0f}'.format(y)])
    return options

x_axis_select = dcc.Dropdown(
    id="x-axis-dropdown",
    options=[{
        'label': "{0} {2}".format(*x),
        'value': "{0}:{1}:{2}".format(*x)
    } for x in get_axis_options()],
    clearable=False
)

y_axis_select = dcc.Dropdown(
    id="y-axis-dropdown",
    options=[{
        'label': "{0} {2}".format(*x),
        'value': "{0}:{1}:{2}".format(*x)
    } for x in get_axis_options()],
    clearable=False
)

color_select = dcc.Dropdown(
    id="color-dropdown",
    options=[{
        'label': x,
        'value': x
    } for x in get_param_values()],
    clearable=False
)

experiment_pick = dcc.Checklist(
    id="experiment-pick",
    options=[{
        'label': name,
        'value': name
    } for name in []]
)
refresh_button = html.Button('Refresh experiment list', id='refresh-button', n_clicks=0)
controlpanel.add_element(experiment_select, "ExperimentGroup")
controlpanel.add_element(html.H6('x variable:'), "ExperimentGroup")
controlpanel.add_element(x_axis_select, "ExperimentGroup")
controlpanel.add_element(html.H6('y variable:'), "ExperimentGroup")
controlpanel.add_element(y_axis_select, "ExperimentGroup")
controlpanel.add_element(html.H6('color variable:'), "ExperimentGroup")
controlpanel.add_element(color_select, "ExperimentGroup")
controlpanel.add_element(refresh_button, "RefreshGroup")
controlpanel.add_groups_to_section("ExperimentSection", ["ExperimentGroup"])
controlpanel.add_groups_to_section("ExperimentSection", ["RefreshGroup"])

grid = dui.Grid(
    _id="grid",
    num_rows=12,
    num_cols=12,
    grid_padding=0
)

# grid.add_element(col=1, row=1, width=12, height=2, element=html.Div(children=[
#     html.H5('Experiments:'),
#     experiment_pick
# ]))
grid.add_graph(col=1, row=1, width=12, height=8, graph_id="plot")
#grid.add_graph(col=1, row=9, width=12, height=4, graph_id="economics")



app.layout = html.Div(
    dui.Layout(
        grid=grid,
        controlpanel=controlpanel
    ),
    style={
        'width': '100vw',
        'height': '100vh'
    }
)


def create_output_df (values, xvar, yvar, colorvar):
    rows = []

    values = [values] if type(values) == type('a') else values
    xvar_info = xvar.split(":")
    yvar_info = yvar.split(":")

    x_name, x_id = xvar_info[0], int(xvar_info[1])
    y_name, y_id = yvar_info[0], int(yvar_info[1])

    x_name_full = "{0} {2}".format(*xvar_info)
    y_name_full = "{0} {2}".format(*yvar_info)

    for name in values:
        if name in outputs:
            for experiment in outputs[name]:
                current_info = {
                    x_name_full: experiment[x_name][x_id],
                    y_name_full: experiment[y_name][y_id],
                    colorvar: experiment['meta']['params'].default_params[colorvar]
                }
                rows.append(current_info)
    return pd.DataFrame(rows), x_name_full, y_name_full, colorvar


@app.callback(Output('plot', 'figure'), [
    Input('experiment-dropdown', 'value'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'),
    Input('color-dropdown', 'value')
])
def update_plot(values, xvar, yvar, colorvar):
    try:
        df, x_name, y_name, colorvar = create_output_df(values, xvar, yvar, colorvar)
        fig = px.scatter(df, x=x_name, y=y_name, color=colorvar)
    except:
        fig = go.Figure()
    return fig


@app.callback(Output('experiment-dropdown', 'options'), [Input('refresh-button', 'n_clicks')])
def refresh_experiments(n):
    load_all()

    return [{
        'label': x,
        'value': x
        } for x in list(outputs.keys())
    ]


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
