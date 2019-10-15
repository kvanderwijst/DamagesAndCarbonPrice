######
# Numba implementation of optimal control problem
######


#### Import packages
import numpy as np

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, plot_output_all





######
# Calibration
######

paramsBL = Params(carbonbudget=0.0, damage="nodamage", K_values_num=20)
outputBL = full_run_structured(paramsBL)
consumptionBL = outputBL['consumption']
t_values = outputBL['t_values']
#plot_output(p_pathBL, E_pathBL, restBL, t_values)


def gamma_val(beta=2.0, rho=0.82):
    if beta == 2.0:
        if rho == 0.65: return 4683.5
        if rho == 0.82: return 1500.0
        if rho == 0.95: return 600.3
    raise Exception("Gamma not yet calibrated for these parameters")


######
# Without damages: calibrate, using beta=2.0
# Results:
#
# beta=2.0, rho=0.65: gamma=4683.5
# beta=2.0, rho=0.82: gamma=1500.0
# beta=2.0, rho=0.95: gamma=600.3
#
######

calibration2p0_0p65 = {}
for cb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    gamma = 3.1223 * 1500
    pMax = gamma
    params = Params(carbonbudget=cb, progRatio=0.65, gamma=gamma, p_values_max=pMax, damage="nodamage", K_values_num=20, minEmissions=-300)
    calibration2p0_0p65[cb] = full_run(params)

calibration2p0_0p95 = {}
for cb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    gamma = 0.4002 * 1500
    pMax = 3.5*gamma if cb < 0.15 else 2.5*gamma
    params = Params(carbonbudget=cb, progRatio=0.95, gamma=gamma, p_values_max=pMax, damage="nodamage", K_values_num=20)
    calibration2p0_0p95[cb] = full_run(params)


#pyo.iplot([go.Scatter(y=calibration2p0_0p95[0.6][0])])


calibration2p0_gamma1500 = {}
for cb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    params = Params(carbonbudget=cb, damage="nodamage", K_values_num=20, gamma=1500)
    calibration2p0_gamma1500[cb] = full_run(params)

def npv(array, t_values, r=0.05):
    return np.sum(np.exp(-r * t_values) * array)

def calibrate(runs, consumptionBL):
    for cb, output in runs.items():
        t_valuesIn = output[3]
        # First calculate consumption losses
        consumption = output[2][:,7]
        consumptionLoss = npv((consumptionBL - consumption) / consumptionBL, t_valuesIn)
        print(cb, consumptionLoss)

calibrate(calibration2p0, consumptionBL)
calibrate(calibration2p0_gamma1500, consumptionBL)
calibrate(calibration2p0_0p65, consumptionBL)
calibrate(calibration2p0_0p95, consumptionBL)





#############
##
## Experiments
##
#############

params_experiment1 = Params(
    carbonbudget=0.3,
    damage="nodamage",
    K_values_num=30,
    CE_values_num=800,
    minEmissions=-10,
    gamma=gamma_val(2.0, 0.82), beta=2.0, progRatio=0.82
)
output_experiment1 = full_run_structured(params_experiment1)

plot_output_all(output_experiment1, False)

params_experiment2 = Params(
    carbonbudget=0.3,
    damage="damage1",
    K_values_num=50,
    CE_values_num=1500,
    minEmissions=-10,
    gamma=gamma_val(2.0, 0.82), beta=2.0, progRatio=0.82
)
output_experiment2 = full_run_structured(params_experiment2)
plot_output_all(output_experiment2, False)



params_experiment3 = Params(
    carbonbudget=0.3,
    damage="damage1",
    K_values_num=20,
    CE_values_num=500,
    E_values_num=70,
    # minEmissions=-10,
    maxReductParam=1,
    gamma=gamma_val(2.0, 0.82), beta=2.0, progRatio=0.82
)
output_experiment3 = full_run_structured(params_experiment3)
plot_output_all(output_experiment2, False)

# npv((consumptionBL - output_experiment2['consumption']) / consumptionBL, t_values)
