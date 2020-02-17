######
# Provides GDP, population and related data
######

import numpy as np
import numba as nb
import pandas as pd

try:
    import importlib_resources as pkg_resources
except:
    import importlib.resources as pkg_resources
import carbontaxdamages.data

f8 = nb.float64
i8 = nb.int64

f8_1d = nb.typeof(np.array([1.0]))
f8_2d = nb.typeof(np.array([[1.0]]))


# To extrapolate: take growth rate 2090-2100, linearly bring it down to growth rate of 0 in 2150
def extrapolate(input, years, extra_years, stabilising_years=50):
    # First get final change rate
    change_rate = (input[-1] - input[-2]) / (years[-1] - years[-2])
    minmax = np.maximum if change_rate > 0 else np.minimum

    t_prev = years[-1]
    val_prev = input[-1]

    new_values = []

    for t in extra_years:
        change = minmax(0.0, change_rate - change_rate * (t_prev - 2100.0) / stabilising_years)
        val = val_prev + change * (t-t_prev)

        new_values.append(val)

        val_prev = val
        t_prev = t

    return np.concatenate([input, new_values])


SSP_years =   np.linspace(2010, 2100, 10)
extra_years = np.arange(2110, 2260, 10)
extended_years = np.concatenate([SSP_years, extra_years])

##########################
##########################
## Population
##########################
##########################

# population_years = np.linspace(2010, 2100, 10)
population_data = {
    'SSP1': 1e6 * np.array([6921.797852, 7576.10498, 8061.937988, 8388.762695, 8530.5, 8492.175781, 8298.950195, 7967.387207, 7510.454102, 6957.98877]),
    'SSP2': 1e6 * np.array([6921.797852, 7671.501953, 8327.682617, 8857.175781, 9242.542969, 9459.967773, 9531.094727, 9480.227539, 9325.707031, 9103.234375]),
    'SSP3': 1e6 * np.array([6921.797852, 7746.918945, 8571.573242, 9324.817383, 10038.44043, 10671.42969, 11232.58008, 11768.17969, 12288.11035, 12793.15039]),
    'SSP4': 1e6 * np.array([6921.797852, 7660.201172, 8300.902344, 8818.331055, 9212.981445, 9459.842773, 9573.267578, 9592.255859, 9542.825195, 9456.296875]),
    'SSP5': 1e6 * np.array([6921.797852, 7584.920898, 8091.687988, 8446.890625, 8629.456055, 8646.706055, 8520.380859, 8267.616211, 7901.399902, 7447.205078])
}

population_extended = {SSP: extrapolate(data, SSP_years, extra_years) for SSP, data in population_data.items()}



def population(year, SSP):
    return np.interp(year, extended_years, population_extended[SSP])


##########################
##########################
## GDP [Trillion USD/yr]
##########################
##########################

# GDP_years = np.linspace(2010, 2100, 10)
GDP_data = {
    'SSP1': 1e-3 * np.array([68461.88281, 101815.2969, 155854.7969, 223195.5, 291301.4063, 356291.4063, 419291.1875, 475419.1875, 524875.8125, 565389.625]),
    'SSP2': 1e-3 * np.array([68461.88281, 101602.2031, 144812.9063, 188496.5938, 234213.4063, 283250.5938, 338902.5, 399688.5938, 466015.8125, 538245.875]),
    'SSP3': 1e-3 * np.array([68461.88281, 100879.5, 135704.5938, 160926.5, 180601, 197332.2031, 215508.2969, 234786.4063, 255673.7969, 279511.1875]),
    'SSP4': 1e-3 * np.array([68461.88281, 103664.2969, 147928.2969, 191229.5, 229272.7969, 262242.4063, 292255.1875, 317266.6875, 339481.8125, 360479.5]),
    'SSP5': 1e-3 * np.array([68461.88281, 105689.8984, 174702.2969, 271955.0938, 378443.0938, 492278.5938, 617695.3125, 749892.8125, 889666.3125, 1034177.])
}

GDP_extended = {SSP: extrapolate(data, SSP_years, extra_years) for SSP, data in GDP_data.items()}
# go.Figure([go.Scatter(x=SSP_years, y=GDP_data['SSP3']),go.Scatter(x=extended_years, y=GDP_extended['SSP3'])]).show()

def GDP(year, SSP):
    return np.interp(year, extended_years, GDP_extended[SSP])


##########################
##########################
## Baseline emissions
##########################
##########################

# baseline_emissions_years = np.linspace(2010, 2100, 10)
baseline_emissions_data = {
    'SSP1': 1e-3 * np.array([35488.81804,40069.00391,42653.23405,43778.4961,42454.75782,41601.92839,39217.53158,33392.29395,28618.4139,24612.91358]),
    'SSP2': 1e-3 * np.array([35612.61459,43478.01205,49474.44434,52913.73829,55991.09929,57621.59506,60866.66667,64443.06316,67837.07162,72492.60678]),
    'SSP3': 1e-3 * np.array([35710.44369,46649.57129,53669.32943,56344.23796,61127.10743,61307.10352,63838.6862,66926.7142,70980.4362,76477.13477]),
    'SSP4': 1e-3 * np.array([35594.47461,42463.96029,48287.80144,50825.20638,50388.90886,48402.19857,47702.92872,47037.72234,44735.16667,44358.89291]),
    'SSP5': 1e-3 * np.array([35982.45736,46630.57553,60289.7181,72707.2142,86440.4922,94748.90105,105555.3053,110432.9245,112423.4447,111910.0397])
}

baseline_emissions_extended = {SSP: extrapolate(data, SSP_years, extra_years) for SSP, data in baseline_emissions_data.items()}

def baseline_emissions(year, SSP):
    return np.interp(year, extended_years, baseline_emissions_extended[SSP])


##########################
##########################
## Damage functions
##########################
##########################

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageHowardTotalProductivity(T):
    return 1.1450 * T**2 / 100.0

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageHowardTotal(T):
    return 1.0038 * T**2 / 100.0

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageHowardNonCatastrophic(T):
    return 0.7438 * T**2 / 100.0

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageTol2009(T):
    return (-2.46*T + 1.1*T**2) / 100.0

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageNewboldMartin2014(T):
    return (1.1 * T + (T > 3)*(3.6-1.1)*(T-3)) / 100.0

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageDICE(T): # DICE-2013R damage function
    return 0.267 * T**2 / 100.

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageTol2014(T):
    return (0.28 * T + 0.16 * T**2) / 100.

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageBurkePooled(T):
    shift = 0.8
    return 0.0 * T + (T > 0.8) * ( 0.326657 * (T-shift) - 0.033932 * (T-shift)**2 )

@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def damageBurkeDiff(T):
    shift = 0.8
    return 0.0 * T + (T > 0.8) * ( 0.1845 * (T-shift) - 0.012336 * (T-shift)**2 )


@nb.njit([ f8(f8), f8_1d(f8_1d) ], fastmath=True)
def nodamage(T):
    return 0.0 * T

damages = {
    "damageHowardTotalProductivity": damageHowardTotalProductivity,
    "damageHowardTotal": damageHowardTotal,
    "damageHowardNonCatastrophic": damageHowardNonCatastrophic,
    "damageTol2009": damageTol2009,
    "damageNewboldMartin2014": damageNewboldMartin2014,
    "damageDICE": damageDICE,
    "damageTol2014": damageTol2014,
    "damageBurkePooled": damageBurkePooled,
    "damageBurkeDiff": damageBurkeDiff,
    "nodamage": nodamage
}



###########################
##
## Import calibrated values of gamma (abatement costs)
##
###########################

gamma_SSP_combined = True

calibration_filename = "calibrated_gamma_SSP_combined.csv" if gamma_SSP_combined else "calibrated_gamma.csv"

gamma_values_df = pd.read_csv(pkg_resources.open_text(carbontaxdamages.data, calibration_filename)).fillna('').set_index(['SSP', 'beta', 'rho', 'cost_percentile'])

def gamma_val(SSP, beta, rho, cost_level):
    try:
        gamma = gamma_values_df.loc['' if gamma_SSP_combined else SSP, beta, rho, cost_level]['gamma']
    except:
        raise Exception("This combination of parameters has not been calibrated yet: ", SSP, beta, rho, cost_level)
    return gamma



###########################
##
## Minimum discount rate: pure per capita economic growth rate
##
###########################

def annual_growth_rates(data, years):
    return ((data[1:] - data[:-1]) / data[:-1]) / (years[1:] - years[:-1])

def growth_rates_GDP_per_capita(SSP, years):
    GDP_per_capita = GDP(years, SSP) / population(years, SSP)
    return annual_growth_rates(GDP_per_capita, years)

growth_rates_extended = {SSP: growth_rates_GDP_per_capita(SSP, np.concatenate([extended_years, [extended_years[-1]+10]])) for SSP in GDP_data.keys()}

def growth_rate(year, SSP):
    return np.interp(year, extended_years, growth_rates_extended[SSP])


#
