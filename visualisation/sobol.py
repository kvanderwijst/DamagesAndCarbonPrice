import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

import itertools
from abc import ABC, abstractmethod

import scipy.stats as stats
from tqdm import tqdm

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from .utils import *

###############
###############
# 1. Sobol decomposition
###############
###############

### Discrete Sobol class

class DiscreteSobol:
    
    def __init__(self, df, probs):
        self.df = df
        self.df_indexed = df.set_index(list(df.columns[1:]))
        
        # Create the param_options dictionary:
        # {
        #     'SSP': {
        #         'values': ['SSP1', 'SSP2', ...],
        #         'p': [1/5, 1/5, ...]
        #     },
        #     ...
        # }
        param_options = {}
        param_indices = {}
        for i, col in enumerate(df.columns[1:]):
            values = list(df[col].unique())
            if col in probs:
                param_options[col] = probs[col]
            else:
                param_options[col] = {
                    'values': values,
                    'p': np.ones(len(values)) / len(values)
                }
            param_indices[col] = {
                'index': i,
                'values': dict(zip(values, range(len(values))))
            }
        
        self.param_options = param_options
        self.param_indices = param_indices
        
        self.indexed_values = self.create_indexed_values()
    
    def _create_sample_params(self, num):
        return np.array([
            np.random.choice(len(x['values']), size=num, p=x['p'] / np.sum(x['p']))
            for x in self.param_options.values()
        ]).T
    
    def i_to_name(self):
        return {i: col for i, col in enumerate(self.df.columns[1:])}
    
    
    #### For quick vectorised access, create indexed values
    
    def _get_value_df(self, ordered_values):
        return self.df_indexed.loc[ordered_values].values[0]
    
    def _nmax(self):
        return np.max([len(x['values']) for x in self.param_options.values()])
    
    def _lst_to_ind(self, params):
        nmax = self._nmax()
        return np.sum([nmax ** i * params[:,i] for i in range(params.shape[1])], axis=0)
    
    def _lst_to_values(self, lst):
        return tuple(x['values'][i] for i, x in zip(lst, self.param_options.values()))
    
    def create_indexed_values(self):
        nmax = self._nmax()
        values = np.full(nmax**len(self.param_options), np.nan)

        # Loop through every combination of 1..n_i, for i in num_param_options
        for i_vect in itertools.product(*[range(len(x['values'])) for x in self.param_options.values()]):
            index = self._lst_to_ind(np.array([i_vect]))
            values[index] = self._get_value_df(self._lst_to_values(i_vect))

        return values
    
    def to_fixed_values(self, splitted_vars):
        return {
            self.param_indices[column]['index']: self.param_indices[column]['values'][option]
            for column, option in splitted_vars
        }
    
    
    #### Sobol functionality
    
    def _f(self, params):
        return self.indexed_values[self._lst_to_ind(params)]
    
    def _create_C_i(self, A, B, i):
        tmp = B.copy()
        tmp[:,i] = A[:,i]
        return tmp
    
    def sobol_indices(self, num=2500000, fixed_vals={}, relative=True, with_interaction=False, with_secondorder=False):
        
        n = len(self.param_options)

        f = self._f

        A = self._create_sample_params(num)
        B = self._create_sample_params(num)

        for i, val in fixed_vals.items():
            A[:,i] = val
            B[:,i] = val

        C_s = [self._create_C_i(A, B, i) for i in range(n)]

        fHat0_2 = np.sum(f(A) * f(B)) / num
        
        total_variance = np.sum(f(A)*f(A))/num - fHat0_2

        S_i = np.array([(np.sum(f(A)*f(C_i))/num - fHat0_2) / total_variance for C_i in C_s])
        
        if with_interaction:
            S_i = np.concatenate((S_i, [1-np.sum(S_i)]))

        S_ij = {}
        if with_secondorder:
            for i in range(n):
                for j in range(i+1, n):
                    C_ij = self._create_C_i(A, B, [i,j])
                    S_ij[(i,j)] = (np.sum( f(A)*f(C_ij) )/num - fHat0_2) / total_variance - S_i[i] - S_i[j]

        
        # S_T_i = np.array([1 - (np.sum(f(B)*f(C_i))/num - fHat0_2) / (np.sum(f(A)*f(A))/num - fHat0_2) for C_i in C_s])
        
        if relative:
            if with_secondorder:
                return S_i, S_ij
            else:
                return S_i
        else:
            if with_secondorder:
                return S_i * total_variance, {ij: value * total_variance for ij, value in S_ij.items()}
            else:
                return S_i * total_variance
    

def plot_Sobol(data, sobol_obj, relative):
    fig = go.Figure()

    all_data = np.array([x for x in data.values()])
    x_data = np.array(list(sobol_obj.param_options.keys()))
    y_data = np.array([str(x)+' ' for x in data.keys()])

    # Add interaction terms
    x_data = np.concatenate([x_data, ['interactions']])

    for i, cat in enumerate(x_data):
        fig.add_trace(go.Bar(
            y=y_data,
            x=all_data[:,i],
            name=x_data[i], orientation='h',
            marker={
                'color': colors_PBL[i] if cat != 'interactions' else '#BBB',
                'line_color': 'rgba(0,0,0,0)'
            },
            text=[('{}%'.format(int(np.round(x*100 / np.sum(all_data[j,:]))))) for j, x in enumerate(all_data[:,i])],textposition='inside',
            textfont_color='#FFF'
        ))
    
    line_props = {
        'type': 'line', 'line_color': '#999', 'line_width': 2,
        'xref': 'paper', 'yref': 'paper', 'x0': 0, 'x1': -0.2
    }
    fig.add_shape(go.layout.Shape(**line_props, y0=0, y1=0))
    fig.add_shape(go.layout.Shape(**line_props, y0=1, y1=1))
    fig.add_shape(go.layout.Shape(**line_props, y0=1 - 1/len(y_data), y1=1 - 1/len(y_data)))
    
    y_title = 'Contribution to variance' if relative else 'Variance'
    
    fig.update_layout(
        barmode='stack',
        legend={'orientation': 'h', 'traceorder': 'normal', 'y': 1.15, 'x': 0.15},
        xaxis={'title': y_title, **({'tickformat': ',.0%', 'range': [0,1]} if relative else {})},
        yaxis={'autorange': 'reversed'}
    )
    return fig




def conditional_sobol(sobol_obj, param, plot=True, param_name=None, relative=True, **kwargs):
    param_name = param.title() if param_name == None else param_name
    param_i = sobol_obj.param_indices[param]['index']
    sobol_temp = {
        '<b>Combined</b>': sobol_obj.sobol_indices(with_interaction=True, relative=relative, **kwargs)
    }

    for i, val in enumerate(sobol_obj.param_options[param]['values']):
        sobol_temp['{}:  <br>{} '.format(param_name, val)] = sobol_obj.sobol_indices(fixed_vals={param_i: i}, with_interaction=True, relative=relative, **kwargs)
    
    # return sobol_temp
    if plot:
        return plot_Sobol(sobol_temp, sobol_obj, relative=relative)
    else:
        return sobol_temp










###############
###############
# 2. Tree decomposition
###############
###############




class TreeAnalysis(ABC):
    
    def __init__(self, df, colors=None):
        self.df = df
        
        # Initialise tree
        self.tree = {'selection': self.df, 'splitted_vars': []}
        _ = self.split_node(self.tree)
        
        # Initialise colors
        cols = list(df.columns[1:]) + ['']
        
        if colors == None:
            colors = [rgb2hex(col) for col in px.colors.DEFAULT_PLOTLY_COLORS]
        
        self.node_color_dict = {name: colors[i] for i, name in enumerate(cols)}
        self.node_color_dict[None] = '#999999'
        
        self.depth_node_sizes = {0: 8000, 1: 4000, 2: 2000, 3: 1000, 4: 500, 5: 500, 6: 500}
        self.depth_label_sizes = {1: 26, 2: 26, 3: 26, 4: 0}
        self.edge_width = 3
        self.edge_alpha = 0.35
        self.filename = 'tree'
        
    def split(self, df, column, node):
        # Get unique values of column
        options = df[column].unique()
        return {
            option: {
                'selection': df[df[column] == option].drop(columns=column),
                'children': None,
                'statistics': None,
                'split_var': None,
                'splitted_vars': node['splitted_vars'] + [[column, option]]
            }
            for option in options
        }
    
    @abstractmethod
    def calc_best_split(self, df):
        pass
    
    @abstractmethod
    def end_criterium(self, value):
        pass
    
    def split_node(self, node, custom_col=None):
        # node is an element in the tree
        # Practically, it's a dictionary with elements
        # selection, children, statistics and split_var

        # Calculate best split column
        best_col, value, values = self.calc_best_split(node['selection'], node)
        if custom_col != None:
            best_col = custom_col
            value = values[custom_col]
        node['split_var'] = best_col
        node['statistics'] = {'value': value, 'values': values}

        # Return number of remaining columns
        
        isleaf = len(values) - 1 <= 0 or self.end_criterium(value)

        # Do split
        #if not isleaf:
        if not isleaf:
            node['children'] = self.split(node['selection'], best_col, node)
        return isleaf
        
    def recursive_split(self, node):
        # For every child in this node,
        # perform the split than call this 
        # function again on this node
        for child in node.values():
            isleaf = self.split_node(child)

            if not isleaf:
                self.recursive_split(child['children'])
                
    def create_tree(self):
        self.recursive_split(self.tree['children'])
        
        
        
    
    
    ##############################
    ##
    ##
    ## Plotting
    ##
    ##
    ##############################
    
    def create_nx_tree(self, G, node, parent_id, name, num_nodes_nx, info_nx, labels_nx_edges, max_depth, depth=0, currsize=1):
        curr_id = num_nodes_nx[0]
        num_nodes_nx[0] = curr_id + 1

        G.add_node(curr_id)
        nodetype = node['split_var']
        isleaf = node['split_var'] is None or depth == max_depth
        info = {
            'name': '' if node['split_var'] is None else node['split_var'],
            'statistics': node['statistics'],
            'describe': node['selection'].iloc[:,0].describe(),
            'selection': node['selection'],
            'color': self.node_color_dict[nodetype],
            'size': self.depth_node_sizes[depth],
            'size2': currsize
        }
        info_nx[curr_id] = info

        if parent_id is not None:
            G.add_edge(parent_id, curr_id)
            labels_nx_edges[depth][(parent_id, curr_id)] = name

        if node['children'] is not None and depth < max_depth:
            for name, child in node['children'].items():
                newsize = node['statistics']['values'][child['split_var']] / node['statistics']['value'] * currsize
                self.create_nx_tree(G, child, curr_id, name, num_nodes_nx, info_nx, labels_nx_edges, max_depth, depth+1, newsize)

    def rotate(self, text, center_text):
        x, y = text.get_position()
        x0, y0 = center_text.get_position()
        if y != y0:
            angle = np.arctan((y-y0)/(x-x0)) * 180 / np.pi
            text.set_rotation(angle)
        text.set_position((x0 + (x-x0)*1.06, y0 + (y-y0)*1.06))
    
    def draw_circles(self, ax, pos, n=5, color=[0,0,0,0.04]):
        center = pos[0]
        coordinates = np.array(list(pos.values())) - center

        rho = np.sqrt(np.sum(coordinates**2, axis=1))
        d_rho = np.max(rho) / (n-1)

        for i in range(n-1, -1, -1):
            d = 30
            color = [1-(n-1-i)/d,1-(n-1-i)/d,1-(n-1-i)/d,1]
            radius = d_rho * (i + 0.5)
            # Start with outer circle
            # circle = plt.Circle(center, radius, color='w' if i % 2 == 0 else color, zorder=-1)
            circle = plt.Circle(center, radius, color=color, zorder=-1)
            ax.add_artist(circle)
    
    def plot(self, max_depth=20, use_graphviz=True, draw_labels=True):
        num_nodes_nx = [0]
        info_nx, labels_nx_edges = {}, {depth: {} for depth in self.depth_label_sizes.keys()}

        G = nx.Graph()

        self.create_nx_tree(G, self.tree, None, None, num_nodes_nx, info_nx, labels_nx_edges, max_depth)

        node_colors = [info['color'] for info in info_nx.values()]
        node_sizes = np.array([info['size2'] * self.depth_node_sizes[0] for info in info_nx.values()])
        # node_sizes = np.array([info['describe']['std'] * self.depth_node_sizes[0] for info in info_nx.values()])
        # node_sizes = np.array([info['statistics']['value'] * self.depth_node_sizes[0] for info in info_nx.values()])
        node_names = np.array([info['name'] for info in info_nx.values()])

        if use_graphviz:
            pos = graphviz_layout(G, prog='twopi')
        else:
            shells = np.array([6 - len(info['selection'].columns) for info in info_nx.values()])
            shells_groups = [list(np.where(shells == i)[0]) for i in range(np.max(shells)+1)]
            pos = nx.shell_layout(G, shells_groups)

        get_descr = lambda x: '{:.2f} ± {:.2f}'.format(x['mean'], x['std']) if x['count'] > 1 else '{:.2f}'.format(x['mean'])
        node_labels = {node: get_descr(info['describe']) for node, info in info_nx.items()}

        fig, ax = plt.subplots(figsize=(26,26))

            
        for depth, labels in labels_nx_edges.items():
            if self.depth_label_sizes[depth] > 0:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=self.depth_label_sizes[depth])
        for name in np.unique(node_names):
            node_selection = list(np.where(node_names == name)[0])
            nx.draw_networkx_nodes(G, pos, nodelist=node_selection, node_size=node_sizes[node_selection], node_color=self.node_color_dict[name], label=name, ax=ax)
            #break

        nx.draw_networkx_edges(G, pos=pos, ax=ax, width=self.edge_width, alpha=self.edge_alpha)

        if draw_labels:
            text = nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
        else:
            text = None
            

        self.draw_circles(ax, pos, min(5, max_depth+1))

        if draw_labels:
            for t, size in zip(text.values(), node_sizes):
                if size < 1000: # Is leaf?
                    self.rotate(t, text[0])

        legend = plt.legend(numpoints = 1, prop={'size': 28})
        for i in range(len(legend.legendHandles)):
            legend.legendHandles[i]._sizes[0] = 1250
        ax.set_frame_on(False)
        
        
        colors = colors_PBL[:5] + ['#BBBBBB']

        radius_factor = 10

        for i, info in info_nx.items():
            d = list(info['statistics']['values'].values())
            d += [1 - sum(d)]
            d = np.minimum(np.maximum(d, 0), 1)
            center = np.array(pos[i])

            #pos -= 240
            #pos /= 10
            p1, _ = ax.pie([1], center=center, colors=[colors[np.argmax(d[:-1])]], radius=radius_factor*(1.1 * info['size'] / 6000 + 0.9))
            p2, _ = ax.pie([1], center=center, colors=['#FFFFFF'], radius=radius_factor*(info['size'] / 6000 + 0.63))
            p3, _ = ax.pie(d, center=center, colors=colors, radius=radius_factor*(info['size'] / 6000 + 0.5))
            for ps in [p1, p2, p3]:
                for p in ps:
                    p.set_zorder(100)
        
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('../Paper/img/{}.svg'.format(self.filename))
        plt.savefig('../Paper/img/{}.png'.format(self.filename), dpi=50)
        plt.show()
    
        self.info_nx = info_nx
        self.labels_nx_edges = labels_nx_edges
        self.G = G
        self.text = text
        self.pos = pos
        self.node_names = node_names

        return
     

        
##### Implementation of abstract base class
class TreeAnalysisANOVA(TreeAnalysis):
    
    def calc_value(self, df, column):
        # Get unique values of column
        options = df[column].unique()

        values = [df[df[column] == option].iloc[:,0] for option in options]

        # Perform ANOVA:
        return stats.f_oneway(*values).pvalue
    
    def calc_best_split(self, df, node):
        columns = df.iloc[:,1:].columns

        if len(columns) > 1:
            values = [self.calc_value(df, column) for column in columns]
            best_value_index = np.argmin(values)
            return columns[best_value_index], values[best_value_index], dict(zip(columns,values))
        else:
            return columns[0], np.nan, {}
    
    def end_criterium(self, value):
        return value > 0.05


class TreeAnalysisSobol(TreeAnalysis):
    
    def __init__(self, df, probs=[], N=2500000, **kwargs):
        
        # Initialise Sobol object
        self.sobol = DiscreteSobol(df, probs)
        self.N = N
        
        super().__init__(df, **kwargs)
        
    
    def calc_best_split(self, df, node):
        
        columns = df.iloc[:,1:].columns
        
        if len(columns) > 1:
            # Perform Sobol analysis with fixed values from node['splitted_vars']
            fixed_vals = self.sobol.to_fixed_values(node['splitted_vars'])

            indices = self.sobol.sobol_indices(num=self.N, fixed_vals=fixed_vals, relative=True)
            
            best_value_index = np.argmax(indices)
            
            best_column = list(self.sobol.param_options.keys())[best_value_index]
            
            return best_column, indices[best_value_index], dict(zip(self.sobol.param_options.keys(), indices))
        
        else:
            return columns[0], np.nan, {}
            
        
    def end_criterium(self, value):
        return False