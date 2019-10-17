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

from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

import re
p = re.compile('.*experiment_(.+)_')
m = re.match(p, 'output/experiment_1_damagesadsfajasdf.asdf-adsf.dsaf.json')
if m:
    print(m.groups()[0])
else:
    print('default')

outputs = {}

def load_all():
    filenames = glob.glob('output/*.json')
    if len(filenames) == 0:
        raise Exception("No files match the given pattern.")

    p_experiment = re.compile('.*(experiment_[a-zA-Z0-9]+)_')
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS * 3

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

        o['meta']['color'] = colors[i]
        o['meta']['letter'] = chr(97 + i)
        outputs[experiment].append(o)

    return

load_all()

def plot(filename_pattern):
    # Select files matching filename pattern
    filenames = glob.glob('output/'+filename_pattern+'.json')
    if len(filenames) == 0:
        raise Exception("No files match the given pattern.")

    print("Matched the following files: \n", filenames)

    # Import the content of the selected files
    outputs = []
    for filename in filenames:
        with open(filename) as fh:
            outputs.append(json_tricks.load(fh, preserve_order=False))

    # Create plots
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    return outputs


external_stylesheets = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Optimal carbon price path - dashboard"
app.css.config.serve_locally = False


controlpanel = dui.ControlPanel(_id="controlpanel")
controlpanel.create_section(
    section="StateSection",
    section_title="State Selection Section"
)
controlpanel.create_group(
    group="ExperimentGroup",
    group_title="Choose the experiment"
)
experiment_select = dcc.Dropdown(
    id="experiment-dropdown",
    options=[{
        'label': x.title(),
        'value': x
        } for x in list(outputs.keys())
    ],
    value=list(outputs.keys())[0]
)
experiment_pick = dcc.Checklist(
    id="experiment-pick",
    options=[{
        'label': name,
        'value': name
    } for name in []]
)
controlpanel.add_element(experiment_select, "ExperimentGroup")
# controlpanel.add_element(experiment_pick, "ExperimentGroup")
controlpanel.add_groups_to_section("StateSection", ["ExperimentGroup"])

grid = dui.Grid(
    _id="grid",
    num_rows=12,
    num_cols=12,
    grid_padding=0
)

grid.add_graph(col=1, row=3, width=12, height=6, graph_id="emissions-and-price")
grid.add_graph(col=1, row=9, width=12, height=4, graph_id="economics")

grid.add_element(col=1, row=1, width=12, height=2, element=html.Div(children=[
    html.H5('Experiments:'),
    experiment_pick
]))



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

plot_mode = 'lines'


def emission_traces(outputs):
    return [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['E'],
        name=o['meta']['letter'],
        legendgroup=o['meta']['title'],
        mode=plot_mode,
        line={'color': o['meta']['color']}
    ) for o in outputs] + [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['baseline'],
        name='Baseline ' + o['meta']['letter'],
        showlegend=False,
        mode=plot_mode,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color'], 'dash': 'dot'}
    ) for o in outputs]


def price_traces(outputs):
    return [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['p'],
        name=o['meta']['letter'],
        mode=plot_mode,
        showlegend=False,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color']}
    ) for o in outputs]


def damage_traces(outputs):
    return [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['damageFraction'],
        name=o['meta']['letter'],
        mode=plot_mode,
        showlegend=False,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color']}
    ) for o in outputs]


def temp_traces(outputs):
    return [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['temp'],
        name=o['meta']['letter'],
        mode=plot_mode,
        showlegend=False,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color']}
    ) for o in outputs]


def create_emission_and_price_plot(outputs):

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Emissions', 'Carbon price', 'Damage fraction', 'Temperature change (rel. to pre-industrial)'
    ), vertical_spacing=0.13)

    for trace in emission_traces(outputs):
        fig.add_trace(trace, 1, 1)

    for trace in price_traces(outputs):
        fig.add_trace(trace, 1, 2)

    for trace in damage_traces(outputs):
        fig.add_trace(trace, 2, 1)

    for trace in temp_traces(outputs):
        fig.add_trace(trace, 2, 2)

    fig.update_layout(
        margin={'b': 15, 't': 40},
        hovermode='x',
        yaxis3={'tickformat': ',.1%'},
        legend={'y': 0.0}
    )
    return fig


#Ygross, Y, consumption, investments, K*
def economic_traces(outputs, variable, dash_style, show=False):
    return [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o[variable],
        name=variable,
        mode=plot_mode,
        visible=True if show else 'legendonly',
        showlegend=(i == 0),
        legendgroup=variable,
        line={'color': o['meta']['color'], 'dash': dash_style}
    ) for i, o in enumerate(outputs)]


def create_economic_plots(outputs):

    fig = make_subplots(rows=1, cols=1, subplot_titles=('Economics'))

    for trace in economic_traces(outputs, 'Ygross', 'dot'):
        fig.add_trace(trace, 1, 1)
    for trace in economic_traces(outputs, 'Y', 'dash', True):
        fig.add_trace(trace, 1, 1)
    for trace in economic_traces(outputs, 'consumption', 'solid', True):
        fig.add_trace(trace, 1, 1)
    for trace in economic_traces(outputs, 'investments', 'dashdot'):
        fig.add_trace(trace, 1, 1)
    for trace in economic_traces(outputs, 'K', 'dot'):
        fig.add_trace(trace, 1, 1)

    fig.update_layout(margin={'b': 15, 't': 40, 'l': 200, 'r': 200}, hovermode='x')
    return fig

# def create_price_plot(outputs):
#
#
#     return go.Figure(data=traces, layout={
#         'showlegend': True,
#         'title': 'Carbon price',
#         'margin': {'b': 15, 't': 40},
#         'hovermode': 'x'
#     })

@app.callback([
    Output('experiment-pick', 'options'),
    Output('experiment-pick', 'value')
], [Input('experiment-dropdown', 'value')])
def update_experiments(value):
    options = [{
        'label': o['meta']['letter'] + ': ' + o['meta']['title'],
        'value': o['meta']['letter']
    } for o in outputs[value]]
    values = [o['meta']['letter'] for o in outputs[value]]
    return options, values




@app.callback([
    Output('emissions-and-price', 'figure'),
    Output('economics', 'figure')
], [
    Input('experiment-dropdown', 'value'),
    Input('experiment-pick', 'value')
])
def create_plots(experiment, picks):
    ### Plots: [E, baseline], [p], [Ygross, Y, consumption, investments, K*], [temp, cumEmissions*], [damageFraction]

    output_selection = []
    for o in outputs[experiment]:
        if picks == None or o['meta']['letter'] in picks:
            output_selection.append(o)
    # output_selection = [o for o in outputs[experiment] if o['meta']['letter'] in picks]


    return create_emission_and_price_plot(output_selection), create_economic_plots(output_selection)





if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
