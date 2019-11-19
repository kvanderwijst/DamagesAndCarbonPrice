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

import re


outputs = {}

def load_all():
    filenames = glob.glob('output/*.json')
    if len(filenames) == 0:
        raise Exception("No files match the given pattern.")

    p_experiment = re.compile('.*(experiment_[a-zA-Z0-9-]+)_')
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS * 300

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
        if 'shorttitle' not in o['meta']:
            o['meta']['shorttitle'] = o['meta']['letter']
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
    group_title="Choose the experiment"
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
    value=list(outputs.keys())[0]
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
controlpanel.add_element(refresh_button, "RefreshGroup")
controlpanel.add_groups_to_section("ExperimentSection", ["ExperimentGroup"])
controlpanel.add_groups_to_section("ExperimentSection", ["RefreshGroup"])

grid = dui.Grid(
    _id="grid",
    num_rows=12,
    num_cols=12,
    grid_padding=0
)

grid.add_element(col=1, row=1, width=12, height=2, element=html.Div(children=[
html.H5('Experiments:'),
experiment_pick
]))
grid.add_graph(col=1, row=3, width=12, height=6, graph_id="emissions-and-price")
grid.add_graph(col=1, row=9, width=12, height=4, graph_id="economics")



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
        name=o['meta']['shorttitle'],
        legendgroup=o['meta']['title'],
        mode=plot_mode,
        line={'color': o['meta']['color']}
    ) for o in outputs] + [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['baseline'],
        name='Baseline ' + o['meta']['shorttitle'],
        showlegend=False,
        mode=plot_mode,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color'], 'dash': 'dot'}
    ) for o in outputs]


def price_traces(outputs):
    return [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['p'],
        name=o['meta']['shorttitle'],
        mode=plot_mode,
        showlegend=False,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color']}
    ) for o in outputs]


def damage_traces(outputs):
    return [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['damageFraction'],
        name=o['meta']['shorttitle'],
        mode=plot_mode,
        showlegend=False,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color']}
    ) for o in outputs]


def temp_traces(outputs):
    traces = [go.Scatter(
        x=o['meta']['t_values_years'],
        y=o['temp'],
        name=o['meta']['shorttitle'],
        mode=plot_mode,
        showlegend=False,
        legendgroup=o['meta']['title'],
        line={'color': o['meta']['color']}
    ) for o in outputs]
    if len(outputs) > 0:
        max_temp = np.max([o['temp'][o['meta']['t_values_years'] <= 2100] for o in outputs])
    else:
        max_temp = np.nan
    return traces, max_temp

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

    traces, max_temp = temp_traces(outputs)
    for trace in traces:
        fig.add_trace(trace, 2, 2)

    fig.update_layout(
        margin={'b': 15, 't': 40},
        hovermode='x',
        yaxis3={'tickformat': ',.1%'},
        legend={'y': 0.0},
        xaxis1={'range': [2015.0, 2100]},
        xaxis2={'range': [2015.0, 2100]},
        xaxis3={'range': [2015.0, 2100]},
        xaxis4={'range': [2015.0, 2100]}, yaxis4={'range': [0.7, max_temp * 1.1]}
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

    fig = make_subplots(rows=1, cols=1, subplot_titles=['Economics'])

    for trace in economic_traces(outputs, 'Ygross', 'dash'):
        fig.add_trace(trace, 1, 1)
    for trace in economic_traces(outputs, 'Y', 'dot', True):
        fig.add_trace(trace, 1, 1)
    for trace in economic_traces(outputs, 'consumption', 'solid', True):
        fig.add_trace(trace, 1, 1)
    try:
        for trace in economic_traces(outputs, 'utility', 'dash', True):
            fig.add_trace(trace, 1, 1)
    except:
        print('No utility available')
    for trace in economic_traces(outputs, 'investments', 'dashdot'):
        fig.add_trace(trace, 1, 1)
    for trace in economic_traces(outputs, 'K', 'dash'):
        fig.add_trace(trace, 1, 1)

    fig.update_layout(margin={'b': 15, 't': 40, 'r': 350}, hovermode='x', xaxis={'range': [2015, 2100]})
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

def perform_update_experiments(value):
    options = [{
        'label': '{}: {}'.format(o['meta']['shorttitle'], o['meta']['title']),
        'value': o['meta']['letter']
    } for o in outputs[value]]
    values = [o['meta']['letter'] for o in outputs[value]]
    return options, values

# @app.callback([
#     Output('experiment-pick', 'options'),
#     Output('experiment-pick', 'value')
# ], [Input('experiment-dropdown', 'value')])
# def update_experiments(value):
#     return perform_update_experiments(value)




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



@app.callback([
    Output('experiment-dropdown', 'options'),
    Output('experiment-pick', 'options'),
    Output('experiment-pick', 'value')
], [Input('refresh-button', 'n_clicks'), Input('experiment-dropdown', 'value')],
[State('refresh-button', 'n_clicks')])
def refresh_experiments(n, value, n_old):
    load_all()
    return ([{
        'label': x,
        'value': x
        } for x in list(outputs.keys())
    ], *perform_update_experiments(value))


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
