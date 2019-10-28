######
# Numba implementation of optimal control problem
######


#### Import packages
import numpy as np
import pandas as pd
idx = pd.IndexSlice
import plotly.graph_objs as go
import plotly.express as px

import scipy.optimize

import json_tricks

def write_json(outp, name):
	with open(name+'.json', 'w') as outfile:
		json_tricks.dump(outp,outfile)

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output





######
# Calibration
######



##############################
########## Analyse calibration
##############################

with open("output/calibration/calibration.json", 'r') as fh:
    calibration_read = json_tricks.load(fh, preserve_order=False)


calibration_df = pd.DataFrame(calibration_read)
calibration_df["rho"] = calibration_df["rho"].astype(str)
calibration_df["beta"] = calibration_df["beta"].astype(str)
px.scatter(calibration_df, x="carbonbudget", y="consumptionLoss2", color="SSP", facet_col="beta", facet_row="rho")

##### Fit exponential function to data
#calibration_df[(calibration_df["SSP"] == "SSP1") & (calibration_df["beta"] == "2.0") & (calibration_df["rho"] == "0.65")][["carbonbudget", "consumptionLoss1"]].values
calibration_indexed = calibration_df.set_index(['SSP', 'beta', 'rho', 'carbonbudget'])[['consumptionLoss2']]


def f(x, a, b):
    return 0.001 * a * (np.exp(-b*(x-1)) - 1)

calibration_coefficients = {}
for SSP in ['SSP1','SSP2','SSP3','SSP4','SSP5']:
    for beta in ['2.0', '3.0']:
        for rho in ['0.65', '0.82', '0.95']:
            data = calibration_indexed.loc[idx[SSP, beta, rho, :]].reset_index().values
            coeff, _ = scipy.optimize.curve_fit(f, data[:,0], data[:,1])
            calibration_coefficients[(SSP, beta, rho)] = coeff

calibration_coefficients_df = pd.DataFrame(calibration_coefficients).T.rename_axis(index=['SSP', 'beta', 'rho']).rename(columns={0: 'a', 1: 'b'})
# calibration_coefficients_df.head()




###### Calibrate with AR5 consumption loss data
AR5_data = pd.read_csv("../AR5fig6.23.csv")
# Y-Values are in percentage points, still need to divide by 100:
AR5_data['Y-Value'] /= 100.0

# General equilibrium only
AR5_data_GE = AR5_data[AR5_data['ModelType'] == 'GE'][['X-Value', 'Y-Value']].values


def mult_factor(coeffs):
    return lambda x, factor: factor * f(x, *coeffs)

def calc_mult_factor(SSP, beta, rho):
    # Get coefficients
    coeffs = calibration_coefficients_df.loc[SSP, beta, rho].values

    # Calculate multiplication factor
    factor, pcov = scipy.optimize.curve_fit(mult_factor(coeffs), AR5_data_GE[:,0], AR5_data_GE[:,1])

    return factor




def find_high_low_factors(SSP, beta, rho, best_factor, low=0.16, high=0.84):

    # First define functions that calculate the percentage of points above the exponential function:
    def points_below(SSP, beta, rho, factor):
        coeffs = calibration_coefficients_df.loc[SSP, beta, rho].values
        x_data = AR5_data_GE[:,0]
        y_data = AR5_data_GE[:,1]
        y_prediction = mult_factor(coeffs)(x_data, factor)
        return np.mean(y_data < y_prediction)

    # Since scipy.optimize only accepts scalar function, use a lambda function:
    def points_below_scalar(SSP, beta, rho, goal):
        return lambda factor: points_below(SSP, beta, rho, factor) - goal

    # Perform root finding
    result_low  = scipy.optimize.root_scalar(points_below_scalar(SSP, beta, rho, low), x0=best_factor, bracket=[0.1, 100])
    result_high = scipy.optimize.root_scalar(points_below_scalar(SSP, beta, rho, high), x0=best_factor, bracket=[0.1, 100])

    if (not result_low.converged) or (not result_high.converged):
        raise("Didn't converge", SSP, beta, rho)

    return result_low.root, result_high.root



#############
#############
## Calculate gamma:
#############
#############


gamma_values = {}
for SSP in ['SSP1','SSP2','SSP3','SSP4','SSP5']:
    for beta in ['2.0', '3.0']:
        for rho in ['0.65', '0.82', '0.95']:
            coeffs = calibration_coefficients_df.loc[SSP, beta, rho].values
            factor = calc_mult_factor(SSP, beta, rho)[0]
            factor_low, factor_high = find_high_low_factors(SSP, beta, rho, factor)

            gamma_basis = 1500
            gamma_values[(SSP, beta, rho)] = {
                'best': gamma_basis * factor,
                'low': gamma_basis * factor_low,
                'high': gamma_basis * factor_high,
            }

gamma_values_df = pd.DataFrame(gamma_values).T.rename_axis(['SSP', 'beta', 'rho']).reset_index()
gamma_values_df.to_csv("carbontaxdamages/data/gamma_values2.csv", index=False)



def test_plot(SSP, beta, rho):
    coeffs = calibration_coefficients_df.loc[SSP, beta, rho].values
    xvals = np.linspace(0.05,0.65, 100)

    # Calculate best-fit factor
    factor = calc_mult_factor(SSP, beta, rho)

    # Low and high factors:
    factor_low, factor_high = find_high_low_factors(SSP, beta, rho, factor)

    go.Figure([
        go.Scatter(x=AR5_data_GE[:,0], y=AR5_data_GE[:,1], mode='markers', name='AR5 data'),
        go.Scatter(x=xvals, y=factor * f(xvals, *coeffs), name='Best fit'),
        go.Scatter(x=xvals, y=factor_low * f(xvals, *coeffs), name='Low'),
        go.Scatter(x=xvals, y=factor_high * f(xvals, *coeffs), name='High')
    ]).show()
test_plot('SSP5', '3.0', '0.95')

######
# Without damages: calibrate, using beta=2.0
# Results:
#
# beta=2.0, rho=0.65: gamma=4683.5
# beta=2.0, rho=0.82: gamma=1500.0
# beta=2.0, rho=0.95: gamma=600.3
#
######



















#
