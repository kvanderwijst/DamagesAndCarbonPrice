######
# Numba implementation of optimal control problem
######


#### Import packages
import numpy as np
import pandas as pd


import json_tricks

def write_json(outp, name):
	with open(name+'.json', 'w') as outfile:
		json_tricks.dump(outp,outfile)

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output





######
# Calibration
######


def npv(array, t_values, r=0.05):
    return np.sum(np.exp(-r * t_values) * array)


# For a value of beta, create a list of consumption losses vs cumulative emissions (% baseline)
# Variations: various SSPs, either npv(C) / npv(C_baseline) or npv(C_baseline - C), various progratios
# And calibrate with min emission level?

calibration = []
for SSP in ['SSP2']: #['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
    outputBL = full_run_structured(Params(carbonbudget=0, damage="nodamage", SSP=SSP, K_values_num=20))
    consumptionBL = outputBL['consumption']

    for rho in [0.65, 0.82, 0.95]:
        for beta in [2.0, 3.0]:
            for cb in np.linspace(0.1, 0.6, 6):
                gamma = 1500.0 # First approximation, will be fine-tuned later
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

write_json(calibration, "output/calibration2")