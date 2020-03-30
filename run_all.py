######
# Numba implementation of optimal control problem
######

import numpy as np
import sys

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output


experiments = {}

#####
# Define experiments
#####

beta = 2.0
elasmu = 1.45
withInertia = True

def experiment_all():

    print("Running experiment 'all'")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        for damage in ['nodamage', 'damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']: # ['nodamage', 'damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']
            if damage == 'nodamage':
                TCRE_values = np.array([0.62]) * 1e-3
            else:
                TCRE_values = np.array([0.62 - 2*0.12, 0.62, 0.62 + 2*0.12]) * 1e-3
            for TCRE in TCRE_values:
                for cost_level in ['p05', 'p50', 'p95']:
                    print(SSP, damage, TCRE, cost_level)
                    for r in [0.001, 0.015, 0.03]:# [0.001, 0.015, 0.03]
                        output = full_run_structured(Params(
                            damage=damage, beta=beta,
                            K_values_num=25 if withInertia else 50, CE_values_num=1500 if withInertia else 2000, p_values_num=1000, p_values_max_rel=0.6,
                            E_values_num=30 if withInertia else 2, maxReductParam=2.2 if withInertia else 500,
                            cost_level=cost_level, SSP=SSP, carbonbudget=1344, elasmu=elasmu,
                            TCRE=TCRE, T = 180, t_values_num = int(180/5)+1,
                            r=r,
                            runname="Experiment all %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions budget %carbonbudget GtCO2 elasmu %elasmu beta %beta carbint %maxReductParam",
                            shortname="%SSP %damage %TCRE %cost_level %r"
                        ))
                        export_output(output, plot=False)

experiments['all'] = experiment_all


####################### Cost benefit runs ###########################

def experiment_allCBA():
    print("Running experiment 'allCBA'")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        for damage in ['damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']:
            for TCRE in np.array([0.62 - 2*0.12, 0.62, 0.62 + 2*0.12]) * 1e-3:
                for cost_level in ['p05', 'p50', 'p95']:
                    print(SSP, damage, TCRE, cost_level)

                    for r in [0.001, 0.015, 0.03]:
                        p_values_max_rel = 0.6
                        output = full_run_structured(Params(
                            damage=damage, beta=beta,
                            K_values_num=20 if withInertia else 25, CE_values_num=1000 if withInertia else 1500, p_values_num=800 if withInertia else 1000, p_values_max_rel=p_values_max_rel,
                            E_values_num=30 if withInertia else 2, maxReductParam=2.2 if withInertia else 500,
                            cost_level=cost_level, SSP=SSP, carbonbudget=0, r=r,
                            elasmu=elasmu,
                            TCRE=TCRE,
                            T = 180, t_values_num = int(180/5)+1,
                            useCalibratedGamma=True,
                            runname="Experiment allCBA %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget 2020 beta %beta elasmu %elasmu carbint %maxReductParam",
                            shortname="CBA %SSP %damage %TCRE %cost_level %r"
                        ))
                        export_output(output, plot=False)

experiments['allCBA'] = experiment_allCBA



#####
# Run experiments
#####

to_be_run = sys.argv[1:]
to_be_run = [exp for exp in to_be_run if exp in experiments.keys()]
if len(to_be_run) == 0:
    to_be_run = experiments.keys()
print("Running experiments:", to_be_run)

for name in to_be_run:
    experiments[name]()
