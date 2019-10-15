######
# Numba implementation of optimal control problem
######

import numpy as np

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output

# Possible damages:     damageHowardTotalProductivity, damageHowardTotal, damageHowardNonCatastrophic,
#                       damageTol2009, damageNewboldMartin2014, damageDICE, damageTol2014, nodamage

######
# Baseline run
######

# outputBL = full_run_structured(Params(carbonbudget=0.0, damage="nodamage", K_values_num=20, runname="Baseline"))
# consumptionBL = outputBL['consumption']
# t_values = outputBL['meta']['t_values']
# t_values_years = outputBL['meta']['t_values_years']
# plot_output_all(outputBL)

######
# Calibrated values
######

def gamma_val(beta=2.0, rho=0.82):
    if beta == 2.0:
        if rho == 0.65: return 4683.5
        if rho == 0.82: return 1500.0
        if rho == 0.95: return 600.3
    raise Exception("Gamma not yet calibrated for these parameters")


#####
# Experiments
#####


## Experiment 1
output_experiment1 = [
    full_run_structured(Params(
        carbonbudget=1000,
        damage="nodamage",
        K_values_num=30,
        CE_values_num=800,
        minEmissions=-10,
        SSP=SSP,
        gamma=gamma_val(2.0, 0.82), beta=2.0, progRatio=0.82,
        runname="Experiment 1 %damage %carbonbudget GtCO2 rho %progRatio beta %beta %SSP"
    ))
for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']]
for output in output_experiment1:
    export_output(output)

output_experiment1[0]['investments'] / output_experiment1[0]['Y']


## Experiment 2
output_experiment2 = full_run_structured(Params(
    carbonbudget=1000,
    damage="nodamage",
    K_values_num=30,
    CE_values_num=800,
    p_values_num=500,
    E_values_num=50,
    maxReductParam=1.5,
    gamma=gamma_val(2.0, 0.82), beta=2.0, progRatio=0.82,
    runname="Experiment 2 %damage %carbonbudget GtCO2 rho %progRatio beta %beta %SSP inertia %maxReductParam Gt/yr"
))

plot_output_all(output_experiment2)



import json_tricks
with open("output/nodamage_1000_gtco2_rho_0.82_beta_2.0_ssp1.json") as fp:
    test = json_tricks.load(fp, preserve_order=False)















#
