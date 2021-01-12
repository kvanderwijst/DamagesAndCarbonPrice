import numpy as np
import pandas as pd

# Bugfix for Plotly default export size
import plotly.io as pio
pio.kaleido.scope.default_width = None
pio.kaleido.scope.default_height = None



colors_PBL = ['#00AEEF', '#808D1D', '#B6036C', '#FAAD1E', '#3F1464', '#7CCFF2', '#F198C1', '#42B649', '#EE2A23', '#004019', '#F47321', '#511607', '#BA8912', '#78CBBF', '#FFF229', '#0071BB']




###############
###############
# 0. Utils
###############
###############

def hex_to_rgba(value, alpha, tostring=True):
    value = value.lstrip('#')
    lv = len(value)
    lst = [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)] + [alpha]
    if tostring:
        return list_to_rgba(lst)
    return lst

def list_to_rgba(lst):
    return 'rgba({0},{1},{2},{3})'.format(*lst)


def rgb2hex(rgb):
    rgb_ints = [int(x) for x in rgb.split('(')[1].split(')')[0].split(',')]
    return '#{:02x}{:02x}{:02x}'.format(*rgb_ints)


def blend(color, alpha, bg='#FFFFFF'):
    bg = np.array(hex_to_rgba(bg, 1, False)[:3])
    lst = np.array(hex_to_rgba(color, alpha, False)[:3])
    outp = np.array(alpha * lst + (1-alpha) * bg, dtype='i4')
    return list_to_rgba(list(outp)+[1])


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


p0 = 1 / (2 * 1.64485**2)

cost_and_TCRE_probs = {
    'cost_level': {'values': ['5th perc.', 'Median', '95th perc.'], 'p': [p0, 1-2*p0, p0]},
    'TCRE': {'values': ['0.42', '0.62', '0.82'], 'p': [p0, 1-2*p0, p0]}
}


map_names = {
    'damage': {
        'damageBurkeWithLag': 'Burke (LR)', 
        'damageBurkeNoLag': 'Burke (SR)', 
        'damageHowardTotal': 'Howard Total', 
        'damageHowardNonCatastrophic': 'Howard Preferred', 
        'damageNewboldMartin2014': 'Newbold & Marten', 
        'damageDICE': 'DICE',
        'nodamage': 'No damages'
    },
    'cost_level': {'p05': '5th perc.', 'p50': 'Median', 'p95': '95th perc.'},
    'TCRE': {'0.00042': '0.42', '0.00062': '0.62', '0.00082': '0.82'},
    'r': {'0.001': '0.1%', '0.015': '1.5%', '0.03': '3%'}
}

def parse_input(filename, withInertia=True, allBeta=False, allElasmu=False, map_names=map_names):
    df = pd.read_csv(filename, dtype={'r': str, 'elasmu': str, 'beta': str, 'maxReductParam': str})
    df['TCRE'] = df['TCRE'].round(decimals=5).astype(str)
    df.loc[df['useBaselineCO2Intensity'].isna(), 'useBaselineCO2Intensity'] = False
    df['withInertia'] = df['maxReductParam'].astype(float) < 20

    for colname, mapping in map_names.items():
        # Extend mapping with unknown values:
        map_names_extended = dict(mapping)
        for value in df[colname].unique():
            if value not in map_names_extended:
                map_names_extended[value] = value
        df[colname] = pd.Categorical(df[colname].map(map_names_extended), map_names_extended.values())

    df = df.sort_values(['SSP', 'damage', 'TCRE', 'cost_level', 'r', 'beta', 'elasmu', 'maxReductParam', 'year'])
    
    df = df.loc[
        ((df['beta'] == '2.0') | allBeta) & 
        ((df['elasmu'] == '1.001') | allElasmu) & 
        (df['useBaselineCO2Intensity']) & 
        (df['withInertia'] == withInertia) &
        (df['TCRE'].isin(['0.42','0.62','0.82']))
    ].reset_index(drop=True).copy()
    
    df_indexed = df.set_index(['SSP', 'damage', 'TCRE', 'cost_level', 'r', 'beta', 'elasmu', 'year'])
    
    return df, df_indexed



