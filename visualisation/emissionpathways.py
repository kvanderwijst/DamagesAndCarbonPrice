import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from .utils import *


def select_individual(
    full_df, 
    custom, 
    SSP='SSP2',
    r='1.5%',
    cost_level='Median',
    damage='DICE',
    TCRE='0.62',
    beta='2.0', 
    elasmu='1.45'
):
    default = {
        'SSP': SSP,
        'r': r,
        'cost_level': cost_level,
        'damage': damage,
        'TCRE': TCRE,
        'beta': beta,
        'elasmu': elasmu
    }
    selection = np.ones(len(full_df), dtype='bool')
    for var, val in default.items():
        if val == 'all':
            continue
        if var in custom:
            if custom[var] == 'all':
                continue
            tmp = full_df[var].isin(custom[var])
        else:
            tmp = full_df[var] == val
        selection = np.all([selection, tmp], axis=0)
    return full_df.loc[selection & (full_df['year'] <= 2100)]


def plot_individual(df, extra_var, values='all', colorshift=0, letter='', with_costs=False, lowdamage='No damages', legend_title=None, **kwargs):
    selection = select_individual(df, {'damage': [lowdamage, 'Howard Total'], extra_var: values}, **kwargs)
    options = set(selection[extra_var]) if values == 'all' else values
    
    name_dict = {'r=': 'PRTP: ', 'damage=': 'Damage: ', '=': ': ', 'cost_level': 'Cost level'}
    
    fig = make_subplots(1,2, subplot_titles=('Price', 'Emissions', 'Cost decomposition'), horizontal_spacing=0.14)
    
    var = 'p'
    already_in_legend = []
    
    for i, var in enumerate(['p', 'E']):
        figtmp = px.line(selection,
                         x='year', y=var,
                         line_group='name', line_dash='damage', line_dash_sequence=['solid', 'dot'],
                         color=extra_var, color_discrete_sequence=colors_PBL[colorshift:])
        figtmp.update_traces(showlegend=i == 0)
        if i == 0:
            fig.add_scatter(x=[None],y=[None], mode='lines', line_color='rgba(0,0,0,0)', name='<i>Damage:</i>', legendgroup='damage')
        for trace in figtmp.data:
            splitname = trace.name.split(',')
            trace.name = replace_all(splitname[0], name_dict)
            trace.showlegend = (trace.line.dash == 'solid') & (i == 1)
            trace.legendgroup = 'all'
            fig.append_trace(trace, 1, i+1)
            
            # For legend:
            damage = splitname[1][1:]
            if damage not in already_in_legend:
                already_in_legend.append(damage)
                fig.append_trace(go.Scatter(
                    x=[None], y=[None], mode='lines', line={'dash': trace.line.dash, 'color': '#000'},
                    name=replace_all(damage, name_dict), showlegend=True, legendgroup='damage'
                ), 1, 1)
                
        if i == 0 and legend_title is not None:
            fig.add_scatter(x=[None],y=[None], mode='lines', line_color='rgba(0,0,0,0)', name=f'<i>{legend_title}:</i>', legendgroup='all')
                
    if with_costs:
        groups = selection[selection['damage'] != lowdamage].groupby('name')
        groups2 = selection[selection['damage'] == lowdamage].groupby('name')
        titles = ['{}: {}'.format(extra_var, group[extra_var].iloc[0]) for _,group in groups]
        fig2 = make_subplots(1, len(groups), shared_yaxes=True, subplot_titles=titles)
        for i, ((name, group), (name2, group2)) in enumerate(zip(groups, groups2)):
            fig2.append_trace(
                go.Scatter(x=group['year'], y=group['abatementFraction'], fillcolor=colors_PBL[2], line_width=0, name='Mitigation costs', legendgroup='mitigation', stackgroup=name, showlegend=i==0),
                1, i+1
            )
            fig2.append_trace(
                go.Scatter(x=group['year'], y=group['damageFraction'], fillcolor=colors_PBL[0], line_width=0, name='Damages', legendgroup='damages', stackgroup=name, showlegend=i==0),
                1, i+1
            )
            fig2.append_trace(
                go.Scatter(x=group2['year'], y=group2['abatementFraction'], line_color='#222', line_dash='dot', name='Mitigation costs<br>(cost-effective)', legendgroup='nodamage', showlegend=i==0),
                1, i+1
            )

        fig2.update_xaxes(range=[2020,2100])
        fig2.update_yaxes(tickformat=',.0%', rangemode='nonnegative')
        fig2.update_layout(width=750,
        margin={'l': 0, 't': 35, 'b': 30},
        height=250,legend={'y': 0.5},yaxis1={'title': 'Share of world GDP'},
        title={'text':'<b>{}.</b>'.format(letter), 'x': 0})
    
    fig.update_layout(
        legend={'y': 0.5},
        xaxis1={'range': [2020, 2100]},
        xaxis2={'range': [2020, 2100]},
        yaxis1={'title': 'Carbon price (US$)', 'range': [-127, 2862]},
        yaxis2={'title': 'Emissions (GtCO<sub>2</sub>/yr)', 'title_standoff': 0, 'range': [-23, 51.19]},
        width=750,
        margin={'l': 0, 't': 35, 'b': 30},
        height=250,
        title={'text':'<b>{}.</b>'.format(letter), 'x': 0}
    )
    
    if with_costs:
        return fig, fig2
    
    return fig