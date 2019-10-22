######
# Numba implementation of optimal control problem
######


#### Import packages
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import json_tricks

def write_json(outp, name):
	with open(name+'.json', 'w') as outfile:
		json_tricks.dump(outp,outfile)

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output





######
# Calibration
######


def gamma_val(beta=2.0, rho=0.82):
    if beta == 2.0:
        if rho == 0.65: return 4683.5
        if rho == 0.82: return 1500.0
        if rho == 0.95: return 600.3
    raise Exception("Gamma not yet calibrated for these parameters")


def npv(array, t_values, r=0.05):
    return np.sum(np.exp(-r * t_values) * array)


# For a value of beta, create a list of consumption losses vs cumulative emissions (% baseline)
# Variations: various SSPs, either npv(C) / npv(C_baseline) or npv(C_baseline - C), various progratios
# And calibrate with min emission level?

calibration = []
for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
    outputBL = full_run_structured(Params(carbonbudget=0, damage="nodamage", SSP=SSP, K_values_num=20))
    consumptionBL = outputBL['consumption']

    for cb in np.linspace(0.1, 0.6, 6):
        gamma = 1500.0 # First approximation, will be fine-tuned later
        rho = 0.82
        beta = 2.0
        output = full_run_structured(Params(
            carbonbudget=cb, relativeBudget=True,
            SSP=SSP, K_values_num=30,
            gamma=gamma, progRatio=rho, beta=beta
        ))

        t_values = output['meta']['t_values']
        consumption = output['consumption']
        consumptionLoss1 = npv((consumptionBL - consumption) / consumptionBL, t_values)
        consumptionLoss2 = (npv(consumptionBL - consumption, t_values)) / npv(consumptionBL, t_values)

        calibration.append({
            'SSP': SSP,
            'carbonbudget': cb,
            'consumptionLoss1': consumptionLoss1,
            'consumptionLoss2': consumptionLoss2,
            'rho': rho,
            'beta': beta,
            'gamma': gamma
        })

write_json(calibration3p0_0p82, "output/calibration1")

######
# Without damages: calibrate, using beta=2.0
# Results:
#
# beta=2.0, rho=0.65: gamma=4683.5
# beta=2.0, rho=0.82: gamma=1500.0
# beta=2.0, rho=0.95: gamma=600.3
#
######
