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

def experiment_all(beta=2.0, PRTP_elasmu=[(0.001, 1.001), (0.015, 1.001), (0.03, 1.001)], withInertia=True, filename='all'):

    print("Running experiment 'all'")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        for damage in ['nodamage', 'damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']: # ['nodamage', 'damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']
            if damage == 'nodamage':
                TCRE_values = np.array([0.62]) * 1e-3
            else:
                TCRE_values = np.array([0.42, 0.62, 0.82]) * 1e-3
            for TCRE in TCRE_values:
                for cost_level in ['p05', 'p50', 'p95']:
                    print(SSP, damage, TCRE, cost_level)
                    for PRTP, elasmu in PRTP_elasmu:
                        output = full_run_structured(Params(
                            damage=damage, beta=beta,
                            K_values_num=25 if withInertia else 50, CE_values_num=2000, p_values_num=1000, p_values_max_rel=0.6,
                            E_values_num=40 if withInertia else 2, maxReductParam=2.2 if withInertia else 500,
                            cost_level=cost_level, SSP=SSP, carbonbudget=1344, elasmu=elasmu,
                            TCRE=TCRE, T = 180, t_values_num = int(180/5)+1,
                            r=PRTP,
                            runname=f"Experiment {filename} %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions budget %carbonbudget GtCO2 elasmu %elasmu beta %beta carbint %maxReductParam",
                            shortname="%SSP %damage %TCRE %cost_level %r"
                        ))
                        export_output(output, plot=False)

experiments['all'] = experiment_all
experiments['allDrupp'] = lambda: experiment_all(PRTP_elasmu=[(0.0, 0.5), (0.0, 1.5), (0.02, 2.5)], withInertia=False, filename='allDrupp')
experiments['allBeta'] = lambda: experiment_all(beta=3.0)
# experiments['allInertia'] = lambda: experiment_all(withInertia=True)


####################### Cost benefit runs ###########################

def experiment_allCBA(
    beta=2.0, 
    PRTP_elasmu=[(0.001, 1.001), (0.015, 1.001), (0.03, 1.001)], 
    withInertia=True, 
    filename='allCBA',
    TCRE_values=[0.42, 0.62, 0.82],
    minEmissions=-20
):
    print("Running experiment 'allCBA'")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        for damage in ['damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']:
            for TCRE in np.array(TCRE_values) * 1e-3:
                for cost_level in ['p05', 'p50', 'p95']:
                    print(SSP, damage, TCRE, cost_level)

                    for PRTP, elasmu in PRTP_elasmu:
                        p_values_max_rel = 0.6
                        output = full_run_structured(Params(
                            damage=damage, beta=beta,
                            K_values_num=15 if withInertia else 25, CE_values_num=800 if withInertia else 1500, p_values_num=600 if withInertia else 1000, p_values_max_rel=p_values_max_rel,
                            E_values_num=25 if withInertia else 2, maxReductParam=2.2 if withInertia else 500,
                            cost_level=cost_level, SSP=SSP, carbonbudget=0, r=PRTP,
                            elasmu=elasmu,
                            TCRE=TCRE,
                            minEmissions=minEmissions,
                            T = 180, t_values_num = int(180/5)+1,
                            useCalibratedGamma=True,
                            runname=f"Experiment {filename} %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget 2020 beta %beta elasmu %elasmu carbint %maxReductParam",
                            shortname="CBA %SSP %damage %TCRE %cost_level %r"
                        ))
                        export_output(output, plot=False)

experiments['allCBA'] = lambda: experiment_allCBA()
experiments['allCBADrupp'] = lambda: experiment_allCBA(PRTP_elasmu=[(0.00, 0.5), (0.00, 1.5), (0.02, 2.5)], withInertia=False, filename='allCBADrupp')
experiments['allCBABeta'] = lambda: experiment_allCBA(beta=3.0)
experiments['allCBAminEmissions'] = lambda: experiment_allCBA(TCRE_values=[0.62], minEmissions=0, filename='CBAminEmissions')
# experiments['allCBAInertia'] = lambda: experiment_allCBA(withInertia=True)




#####
# Sensitivity runs
#####

def experiment_sensitivity():
    for SSP, damage, TCRE, cost_level, r, maxReductParam, extra in [
        # Net negative emissions
        ['SSP2', 'damageHowardTotal', 0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'nodamage',          0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],

        ['SSP2', 'damageHowardTotal', 0.62*1e-3, 'p05', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'nodamage',          0.62*1e-3, 'p05', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'damageHowardTotal', 0.62*1e-3, 'p95', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'nodamage',          0.62*1e-3, 'p95', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],

        ['SSP1', 'damageHowardTotal', 0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP1', 'nodamage',          0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP3', 'damageHowardTotal', 0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP3', 'nodamage',          0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP5', 'damageHowardTotal', 0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP5', 'nodamage',          0.62*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        
        ['SSP2', 'damageHowardTotal', 0.42*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'nodamage',          0.42*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'damageHowardTotal', 0.82*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'nodamage',          0.82*1e-3, 'p50', 0.015, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        
        ['SSP2', 'damageHowardTotal', 0.62*1e-3, 'p50', 0.001, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
        ['SSP2', 'nodamage',          0.62*1e-3, 'p50', 0.003, 2.2, {'minEmissions': 0, 'elasmu': 1.001}],
    ]:
        print('Extra: ', extra)
        # Cost-effectiveness

        output = full_run_structured(Params(
            damage=damage,
            K_values_num=25, CE_values_num=2000, p_values_num=1000, p_values_max_rel=0.6,
            E_values_num=40, maxReductParam=maxReductParam,
            cost_level=cost_level, SSP=SSP, carbonbudget=1344,
            TCRE=TCRE, T = 180, t_values_num = int(180/5)+1,
            r=r,
            **extra,
            runname="Experiment extra %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions budget %carbonbudget GtCO2 elasmu %elasmu beta %beta carbint %maxReductParam",
            shortname="%SSP %damage %TCRE %cost_level %r"
        ))
        export_output(output, plot=False)
        
        # Cost-benefit
        # p_values_max_rel = 0.6
        # output = full_run_structured(Params(
        #     damage=damage,
        #     K_values_num=20, CE_values_num=1000, p_values_num=800, p_values_max_rel=p_values_max_rel,
        #     E_values_num=30, maxReductParam=maxReductParam,
        #     cost_level=cost_level, SSP=SSP, carbonbudget=0, r=r,
        #     TCRE=TCRE,
        #     **extra,
        #     T = 180, t_values_num = int(180/5)+1,
        #     useCalibratedGamma=True,
        #     runname="Experiment extraCBA %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget 2020 beta %beta elasmu %elasmu carbint %maxReductParam",
        #     shortname="CBA %SSP %damage %TCRE %cost_level %r"
        # ))
        # export_output(output, plot=False)


experiments['sensitivity'] = experiment_sensitivity



######
# Drupp et al pairs
######
def experiment_drupp(beta=2.0, withInertia=True):
    print("Running experiment 'Drupp'")
    SSP = 'SSP2'
    damage = 'damageHowardTotal'
    TCRE = 0.62 * 1e-3
    cost_level = 'p50'
    for PRTP, elasmu in [(0.0, 0.5), (0.0, 1.5), (0.02, 2.5)]:
    # for PRTP, elasmu in [(0.0, 0.0), (0.0, 0.11), (0.0, 0.2), (0.0, 0.5), (0.0, 0.66), (0.0, 0.8), (0.0, 1.0), (0.0, 1.3), (0.0, 1.5), (0.0, 2.0), (0.0, 3.0), (0.0, 5.0), (0.0, 1.2), (0.0, 0.9), (0.0, 1.4), (0.0, 0.25), (0.0, 1.1), (0.01, 1.5), (0.01, 1.8), (0.01, 0.11), (0.01, 0.2), (0.01, 0.5), (0.01, 0.6), (0.01, 0.7), (0.01, 0.9), (0.01, 1.0), (0.01, 1.3), (0.01, 2.0), (0.01, 2.5), (0.01, 3.0), (0.01, 5.0), (0.01, 1.7), (0.01, 4.0), (0.02, 0.2), (0.02, 0.34), (0.02, 0.5), (0.02, 0.8), (0.02, 1.0), (0.02, 1.25), (0.02, 1.3), (0.02, 2.0), (0.02, 2.5), (0.02, 0.9), (0.03, 0.0), (0.03, 0.25), (0.03, 1.0), (0.03, 1.5), (0.04, 0.5), (0.04, 1.0), (0.04, 1.5), (0.05, 0.2), (0.05, 0.5), (0.06, 4.0), (0.07, 0.7), (0.08, 1.0)]:
        if elasmu == 1.0:
            elasmu = 1.001 # Approximate log
        p_values_max_rel = 0.6
        output = full_run_structured(Params(
            damage=damage, beta=beta,
            K_values_num=20 if withInertia else 25, CE_values_num=1000 if withInertia else 1500, p_values_num=800 if withInertia else 1000, p_values_max_rel=p_values_max_rel,
            E_values_num=30 if withInertia else 2, maxReductParam=2.2 if withInertia else 500,
            cost_level=cost_level, SSP=SSP, carbonbudget=0, r=PRTP,
            elasmu=elasmu,
            TCRE=TCRE,
            T = 180, t_values_num = int(180/5)+1,
            useCalibratedGamma=True,
            runname="Experiment druppthree %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget 2020 beta %beta elasmu %elasmu carbint %maxReductParam",
            shortname="CBA %SSP %damage %TCRE %cost_level %r %elasmu"
        ))
        export_output(output, plot=False)

experiments['drupp'] = lambda: experiment_drupp(withInertia=True)
experiments['druppNoInertia'] = lambda: experiment_drupp(withInertia=False)

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
