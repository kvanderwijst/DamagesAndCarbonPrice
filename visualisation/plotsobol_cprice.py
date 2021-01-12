import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from .utils import *
from .sobol import *



def calculate_sobol_cprices(df, split_param, relative, param_columns, years, cost_and_TCRE_probs, num=600000, valid_dmg_fcts=['DICE', 'Howard Total', 'Burke (LR)']):
    
    selection = df.loc[df['damage'].isin(valid_dmg_fcts)].copy()
    selection['damage'] = pd.Categorical(selection['damage'], valid_dmg_fcts)
    selection['TCRE'] = pd.Categorical(selection['TCRE'], selection['TCRE'].unique())
    selection = selection.sort_values(split_param)

    sobol_per_year = {
        year: conditional_sobol(DiscreteSobol(
                selection.loc[
                    (selection['year'] == year),
                ['p'] + param_columns],
                cost_and_TCRE_probs
            ), split_param, num=num, plot=False, relative=relative)
        for year in tqdm(years)
    }
    return sobol_per_year



def plotsobol_cprice(sobol_per_year, use_sqrt, subplot_titles, relative, param_columns, years):
    fig = make_subplots(
        2,3,
        subplot_titles=subplot_titles,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.15,
        specs=[[{"colspan": 3},None,None],
            [{},{},{}]]
    )

    max_vals = {}
    replace_dict = {'r': 'PRTP', 'cost_level': 'Mitig. cost level', 'damage': 'Damages'}

    # mean_p = df.loc[:, ['name', 'year', 'p']].set_index(['name', 'year']).unstack('year').mean()

    for i, key in enumerate(sobol_per_year[list(sobol_per_year.keys())[0]].keys()):
        matrix = np.array([x[key] for x in sobol_per_year.values()]).T
        y_title = 'Contribution to variance'
        if not relative:
            if use_sqrt:
                matrix = np.sqrt(np.maximum(0.0, matrix))# / np.interp(years, mean_p['p'].index, mean_p['p'].values)
                y_title = 'Standard deviation (US$)'
            else:
                y_title = 'Variance'
        row_num = 1 if i == 0 else 2
        col_num = 1 if i == 0 else i
        max_vals[i] = 1.05 * matrix.sum(axis=0).max()
        for col, row, color in zip(param_columns+['Interactions'], matrix, colors_PBL):#px.colors.DEFAULT_PLOTLY_COLORS):
            if col == 'Interactions':
                color = '#CCC'
            fig.add_trace(go.Scatter(
                x=years, y=row,
                mode='lines', stackgroup=key[0],
                line={'color': color, 'width': 0}, fillcolor=color,
                name=(replace_dict[col] if col in replace_dict else col), legendgroup=col,
                showlegend=i == 0
            ), row_num, col_num)

    to_percentage = {'range': [0,1], 'tickformat': ',.0%'}

    if not relative and use_sqrt:
        max_vals[1] = 400
        max_vals[2] = 600

    if not relative:
        for i in range(2,4):
            fig.add_shape(go.layout.Shape(
                type='line', xref='paper', yref='paper', layer='below', line_color='#E5ECF6', line_width=2.5,
                x0=fig.layout['xaxis'+str(i)].domain[1], y0=fig.layout['yaxis'+str(i)].domain[1],
                x1=fig.layout['xaxis'+str(i+1)].domain[0], y1=fig.layout['yaxis'+str(i+1)].domain[1] * max_vals[i-1] / max_vals[i]))
            fig.add_shape(go.layout.Shape(
                type='line', xref='paper', yref='paper', layer='below', line_color='#E5ECF6', line_width=2.5,
                x0=fig.layout['xaxis'+str(i)].domain[1], y0=0, x1=fig.layout['xaxis'+str(i+1)].domain[1], y1=0
            ))
            
    fig.update_layout(
        yaxis1={**(to_percentage if relative else {}), 'title': y_title},
        yaxis2={**(to_percentage if relative else {'range': [0, max_vals[1]]}), 'title': y_title},
        yaxis3={**(to_percentage if relative else {'range': [0, max_vals[2]]})},
        yaxis4={**(to_percentage if relative else {'range': [0, max_vals[3]]})},
        margin={'l': 0, 't': 50, 'b': 30},
        legend={'traceorder': 'reversed', 'y': 0.5, 'font_size': 14},
        width=950,
        height=550
    )
    return fig
    