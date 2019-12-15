############################
##
## Automated calibration
##
############################

import numpy as np
import pandas as pd
try:
    import plotly.graph_objs as go
    import plotly.express as px
except:
    pass
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.optimize

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output
from carbontaxdamages.economics import *

####
# Calibration:
####

#### Step 1: import cost points from Fig 6.23 AR5 WG3
costs_ar5 = pd.read_csv("../AR5fig6.23.csv").rename(columns={'X-Value': 'x', 'Y-Value': 'y'})
costs_ar5['y'] /= 100 # Costs are in percentage points
costs_ar5_GE = costs_ar5[costs_ar5['ModelType'] == 'GE']

## First, perform simple linear regression without constraints:
aStar, bStar = smf.quantreg('y ~ x', costs_ar5_GE).fit(q=0.5).params[['x', 'Intercept']]

## Then, perform constrained linear regression such that every line intersects
## the y=0 line at the same point
def constrained(x):
    return x + bStar / aStar

model = smf.quantreg('y ~ constrained(x) - 1', costs_ar5_GE) # "-1" removes the intercept variable
slopes = {q: model.fit(q=q).params['constrained(x)'] for q in [0.05, 0.1, 0.16, 0.5, 0.84, 0.9, 0.95]}

# Create f(x) = a x + b for the 16th, 50th and 84th percentiles
def f(a):
    return lambda x: a * constrained(x)
linear_costs = {'p{:02.0f}'.format(q*100): f(slope) for q, slope in slopes.items() if q in [0.05, 0.5, 0.95]}

# Plot:

# x_test = np.linspace(0,0.7,100)
# go.Figure([go.Scatter(x=costs_ar5_GE['x'], y=costs_ar5_GE['y'], mode='markers')] + [
#     go.Scatter(x=x_test, y=linear_cost(x_test), name=q) # result.params['Intercept'
#     for q, linear_cost in linear_costs.items()
# ])


#### Step 2: calculate values of gamma iteratively for different scenarios
#
# Scenario's:
# - SSP: [SSP1, SSP2, SSP3, SSP4, SSP5]
# - rho: [0.65, 0.82, 0.95]
# - beta: [2.0, 3.0]
#
# To do this, calculate NPV consumption loss (npv(C_BL - C) / npv(C_BL)) for a
# range of relative carbon budgets: [0.2, 0.3, 0.4, 0.5]
#
# Then, assuming linearity of NPV CL in gamma, calculate new value of gamma.
# Repeat this process until convergence (or just once)
#
# This is repeated for each percentile of the costs (16th, 50th and 84th)


def baseline_consumption(SSP, t_values_years):
    # Baseline consumption is (100-21)% of baseline GDP
    return (1-0.21) * GDP(t_values_years, SSP)

def npv(array, t_values, r=0.05):
    return np.sum(np.exp(-r * t_values) * array)

def consumption_loss(output, SSP, r=0.05):
    consumptionBL = baseline_consumption(SSP, output['meta']['t_values_years'])
    consumption = output['consumption']

    t_values = output['meta']['t_values']

    return npv(consumptionBL - consumption, t_values, r) / npv(consumptionBL, t_values, r)

def do_carbonbudget_runs(SSP, rho, beta, gamma, r=0.05):
    NPVs = []
    carbonbudgets = []
    if SSP == "":
        SSP_lst = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    else:
        SSP_lst = [SSP]
    for SSP in SSP_lst:
        for cb in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
            output = full_run_structured(Params(
                damage='nodamage', progRatio=rho,
                beta=beta,
                K_values_num=20, CE_values_num=1000, p_values_num=500,
                SSP=SSP, carbonbudget=cb, relativeBudget=True,
                discountRateFromGrowth=False, r=r,
                useCalibratedGamma=False, gamma=gamma,
                runname="calibration %SSP rho %rho beta %beta gamma %gamma r %r CB %carbonbudget",
                shortname="%SSP %damage %TCRE %cost_level %r"
            ))
            consumption_loss_NPV = consumption_loss(output, SSP)
            NPVs.append(consumption_loss_NPV)
            carbonbudgets.append(cb)
    return NPVs, carbonbudgets

def mult_factor(percentile):
    return lambda x, factor: linear_costs[percentile](x) / factor

def calc_mult_factor(carbonbudgets, NPVs, percentile):
    # Calculate multiplication factor
    factor, pcov = scipy.optimize.curve_fit(mult_factor(percentile), carbonbudgets, NPVs)
    return factor[0]


all_consumption_losses = pd.DataFrame({'cb': [], 'NPV': [], 'SSP': [], 'rho': [], 'beta': [], 'cost_level': []})


def update_gamma(SSP, rho, beta, target_percentile, current_gamma, iteration=1, max_iterations=1):
    # Step (a): do a run for the current scenario, and all carbon budgets,
    # and directly calculate the NPV of consumption loss
    NPVs, carbonbudgets = do_carbonbudget_runs(SSP, rho, beta, current_gamma)
    # Step (b): compare this to target consumption losses
    factor = calc_mult_factor(carbonbudgets, NPVs, target_percentile)
    # Step (c): update gamma
    gamma = factor * current_gamma
    print('Old:', current_gamma, 'New:', gamma)
    if iteration < max_iterations:
        return update_gamma(SSP, rho, beta, target_percentile, gamma, iteration+1, max_iterations=max_iterations)
    else:
        # Save consumption losses for future reference
        global all_consumption_losses
        all_consumption_losses = all_consumption_losses.append(pd.DataFrame({
            'cb': carbonbudgets, 'NPV': NPVs,
            'SSP': SSP, 'rho': rho, 'beta': beta, 'cost_level': target_percentile
        }))
        all_consumption_losses.to_csv('calibration_consumption_losses.csv', index=False)
        return gamma



df_gammas = pd.DataFrame(columns=['SSP', 'rho', 'beta', 'cost_percentile', 'gamma'])
i = 0

separate_SSP_calibration = False
if separate_SSP_calibration:
    SSPs = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    SSP_name = 'SSP_seperate'
else:
    # Each SSP/carbon budget combination will be calculated in the calibration step, instead of separately
    SSPs = ['']
    SSP_name = 'SSP_combined'

# for SSP in :
for SSP in SSPs:
    for rho in [0.65, 0.82, 0.95]:
        for beta in [2.0, 3.0]:
            for cost_percentile, cost_level in [['p05', 'p05'], ['p50', 'p50'], ['p95', 'p95']]:
                # Current value of gamma:
                current_gamma = gamma_val(SSP if SSP != '' else 'SSP2', beta, rho, cost_level)
                print(SSP, rho, beta, cost_percentile, cost_level, current_gamma)
                new_gamma = update_gamma(SSP, rho, beta, cost_percentile, current_gamma)
                df_gammas.loc[i] = [SSP, rho, beta, cost_percentile, new_gamma]
                df_gammas.to_csv('calibrated_gamma_{}.csv'.format(SSP_name), index=False)
                i += 1



# calc_mult_factor(test[:,0], test[:,1], 'p50')
#
# test = np.array([[o['meta']['params'].default_params['carbonbudget'], consumption_loss(o, 'SSP2')] for o in outputs])
# x_test = np.linspace(0.1, 0.6, 50)
# go.Figure([
#     go.Scatter(y=[consumption_loss(x, 'SSP5') for x in _[0]], x=_[1]),
#     go.Scatter(x=x_test, y=linear_costs['p16'](x_test)),
#     go.Scatter(x=x_test, y=linear_costs['p50'](x_test)),
#     go.Scatter(x=x_test, y=linear_costs['p84'](x_test))
# ])







#
