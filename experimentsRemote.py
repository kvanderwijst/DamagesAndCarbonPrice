######
# Numba implementation of optimal control problem
######

import numpy as np
import sys

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output

# Possible damages:     damageHowardTotalProductivity, damageHowardTotal, damageHowardNonCatastrophic,
#                       damageTol2009, damageNewboldMartin2014, damageDICE, damageTol2014, nodamage



experiments = {}

#####
# Define experiments
#####

##### Part 1: cost-effectiveness without damages

# Experiment 1a: three discount rates
def experiment_1a():
    print("Running experiment 1a")
    for r in [0.025, 0.05, 0.075]:
        output = full_run_structured(Params(
            r=r,
            carbonbudget=1750,
            damage='nodamage',
            K_values_num=50, CE_values_num=2500, p_values_num=1500,
            SSP='SSP2', useCalibratedGamma=True,
            cost_level='best',
            runname="Experiment 1a-discountrate budget 2C %SSP %damage r=%r %cost_level costs",
            shortname="r=%r"
        ))
        export_output(output, plot=False)

experiments['1a'] = experiment_1a


# Experiment 1b: high/medium/low mitigation costs
def experiment_1b():
    print("Running experiment 1b")
    for cost_level in ['low', 'best', 'high']:
        output = full_run_structured(Params(
            cost_level=cost_level,
            carbonbudget=1750,
            damage='nodamage',
            K_values_num=50, CE_values_num=2500, p_values_num=1500,
            SSP='SSP2', useCalibratedGamma=True,
            runname="Experiment 1b-costs budget 2C %SSP %damage r=%r %cost_level costs",
            shortname="cost_level=%cost_level"
        ))
        export_output(output, plot=False)

experiments['1b'] = experiment_1b


# Experiment 1c: SSPs
def experiment_1c():
    print("Running experiment 1c")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        output = full_run_structured(Params(
            SSP=SSP,
            carbonbudget=1750,
            damage='nodamage',
            K_values_num=50, CE_values_num=2500, p_values_num=1500,
            cost_level='best',
            useCalibratedGamma=True,
            runname="Experiment 1c-SSPs budget 2C %SSP %damage r=%r %cost_level costs",
            shortname="%SSP"
        ))
        export_output(output, plot=False)

experiments['1c'] = experiment_1c


# Experiment 1d: Inertia
def experiment_1d():
    print("Running experiment 1d")
    for maxReductParam in [1.2, 2.2, 3.2]:
        output = full_run_structured(Params(
            maxReductParam=maxReductParam,
            SSP='SSP2',
            carbonbudget=1750,
            damage='nodamage',
            K_values_num=30, CE_values_num=1500,
            E_values_num=80,
            cost_level='best',
            useCalibratedGamma=True,
            runname="Experiment 1d-Inertia maxReduct %maxReductParam GtCO2 p. yr. budget 2C %SSP %damage r=%r %cost_level costs",
            shortname="maxYearlyReduct=%maxReductParam"
        ))
        export_output(output, plot=False)

experiments['1d'] = experiment_1d

# Experiment 1e: Technological learning
def experiment_1e():
    print("Running experiment 1e")
    for rate_LBD, p_max_rel in zip([0.65, 0.82, 0.95], [0.3, 1.5, 3]):
        for useExog in [True, False]:
            name = "LoT " + str(rate_LBD) + " (%exogLearningRate/yr)" if useExog else "LBD %progRatio"
            output = full_run_structured(Params(
                progRatio=rate_LBD,
                useCalibratedExogLearningRate=useExog,
                carbonbudget=1750, # GtCO2,
                p_values_max_rel=p_max_rel,
                K_values_num=50, CE_values_num=2500, p_values_num=1500,
                cost_level='best', SSP='SSP2', damage='nodamage',
                useCalibratedGamma=True,
                runname="Experiment 1e-learning " + name + " budget 2C %SSP %damage r=%r %cost_level costs",
                shortname=name
            ))
            export_output(output, plot=False)

experiments['1e'] = experiment_1e



# Experiment 1f: min. emission level
def experiment_1f():
    print("Running experiment 1f")
    for min_level in [0, -10, -20]: # GtCO2/yr net emissions
        output = full_run_structured(Params(
            minEmissions=min_level,
            damage='nodamage',
            K_values_num=50, CE_values_num=2500, p_values_num=1500,
            cost_level='best', SSP='SSP2', carbonbudget=1750,
            useCalibratedGamma=True,
            runname="Experiment 1f-minemissions %minEmissions budget 2C %SSP %damage r=%r %cost_level costs",
            shortname="%minEmissions GtCO2/yr"
        ))
        export_output(output, plot=False)

experiments['1f'] = experiment_1f






##### Part 2: with damages and carbon budget





# Experiment 2a: different damage functions
def experiment_2a():
    print("Running experiment 2a")
    for damage in ['nodamage', 'damageDICE', 'damageTol2009', 'damageHowardTotal']:
        output = full_run_structured(Params(
            damage=damage,
            K_values_num=50, CE_values_num=2500, p_values_num=1500,
            cost_level='best', SSP='SSP2', carbonbudget=1750,
            useCalibratedGamma=True,
            runname="Experiment 2a-damages %damage %minEmissions budget 2C %SSP r=%r %cost_level costs",
            shortname="%damage"
        ))
        export_output(output, plot=False)

experiments['2a'] = experiment_2a




# Experiment 2b: different mitigation costs
def experiment_2b():
    print("Running experiment 2b")
    for cost_level in ['low', 'best', 'high']:
        for damage in ['nodamage', 'damageDICE', 'damageHowardTotal']:
            output = full_run_structured(Params(
                damage=damage,
                K_values_num=50, CE_values_num=2500, p_values_num=1500,
                cost_level=cost_level, SSP='SSP2', carbonbudget=1750,
                useCalibratedGamma=True,
                runname="Experiment 2b-costlevel %damage %cost_level costs %minEmissions budget 2C %SSP r=%r",
                shortname="%damage %cost_level"
            ))
            export_output(output, plot=False)

experiments['2b'] = experiment_2b



# Experiment 2c: TCRE
def experiment_2c():
    print("Running experiment 2c")
    for TCRE in np.array([0.62 - 2*0.12, 0.62, 0.62 + 2*0.12]) * 1e-3:
        for damage in ['nodamage', 'damageDICE', 'damageHowardTotal']:
            output = full_run_structured(Params(
                damage=damage,
                K_values_num=50, CE_values_num=2500, p_values_num=1500,
                cost_level='best', SSP='SSP2', carbonbudget=1750, TCRE=TCRE,
                useCalibratedGamma=True,
                runname="Experiment 2c-TCRE %damage TCRE %TCRE degC per TtCO2 %cost_level costs %minEmissions budget 2C %SSP r=%r",
                shortname="%damage TCRE %TCRE"
            ))
            export_output(output, plot=False)

experiments['2c'] = experiment_2c



# Experiment 2d: SSPs
def experiment_2d():
    print("Running experiment 2d")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        for damage in ['nodamage', 'damageDICE', 'damageHowardTotal']:
            output = full_run_structured(Params(
                damage=damage,
                K_values_num=50, CE_values_num=2500, p_values_num=1500,
                cost_level='best', SSP=SSP, carbonbudget=1750,
                useCalibratedGamma=True,
                runname="Experiment 2d-SSPs %damage %cost_level costs %minEmissions budget 2C %SSP r=%r",
                shortname="%damage %SSP"
            ))
            export_output(output, plot=False)

experiments['2d'] = experiment_2d



# Experiment 2e: discount rates
def experiment_2e():
    print("Running experiment 2e")
    for r in [0.025, 0.05, 0.075]:
        for damage in ['nodamage', 'damageDICE', 'damageHowardTotal']:
            output = full_run_structured(Params(
                damage=damage,
                K_values_num=50, CE_values_num=2500, p_values_num=1500,
                cost_level='best', SSP='SSP2', carbonbudget=1750, r=r,
                useCalibratedGamma=True,
                runname="Experiment 2e-discountrate %damage r=%r %cost_level costs %minEmissions budget 2C %SSP",
                shortname="%damage r=%r"
            ))
            export_output(output, plot=False)

experiments['2e'] = experiment_2e


def experiment_all():

    print("Running experiment 'all'")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        for damage in ['nodamage', 'damageDICE', 'damageHowardTotal']:
            for TCRE in np.array([0.62 - 2*0.12, 0.62, 0.62 + 2*0.12]) * 1e-3:
                for cost_level in ['low', 'best', 'high']:
                    print(SSP, damage, TCRE, cost_level)
                    for r in [0.025, 0.05, 0.075]:

                        output = full_run_structured(Params(
                            damage=damage,
                            K_values_num=100, CE_values_num=2500, p_values_num=2000,
                            cost_level=cost_level, SSP=SSP, carbonbudget=1750, r=r,
                            TCRE=TCRE,
                            useCalibratedGamma=True,
                            runname="Experiment all %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions budget 2C",
                            shortname="%SSP %damage %TCRE %cost_level %r"
                        ))
                        export_output(output, plot=False)

experiments['all'] = experiment_all


####################### Cost benefit runs ###########################

def experiment_allCBA():
    print("Running experiment 'allCBA'")
    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        # for damage in ['damageDICE', 'damageTol2009', 'damageHowardTotal']:
        for damage in ['damageDICE']:
            # for TCRE in np.array([0.62 - 2*0.12, 0.62, 0.62 + 2*0.12]) * 1e-3:
            for TCRE in np.array([0.62, 0.62 + 2*0.12]) * 1e-3:
                for cost_level in ['low', 'best', 'high']:
                    print(SSP, damage, TCRE, cost_level)

                    # for r, discountRateFromGrowth in [[0.03, False], [0.05, False], [0.07, False], [0.0, True]]:
                    for r, discountRateFromGrowth in [[0.0, True]]:
                        p_values_max_rel = 0.25 if (damage == 'damageDICE' and not discountRateFromGrowth) else 0.5
                        output = full_run_structured(Params(
                            damage=damage,
                            K_values_num=75, CE_values_num=2500, p_values_num=2000, p_values_max_rel=p_values_max_rel,
                            cost_level=cost_level, SSP=SSP, carbonbudget=0, r=r,
                            TCRE=TCRE,
                            discountRateFromGrowth=discountRateFromGrowth,
                            T = 185, t_values_num = int(185/5)+1,
                            useCalibratedGamma=True,
                            runname="Experiment allCBA2 %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget",
                            shortname="CBA %SSP %damage %TCRE %cost_level %r"
                        ))
                        export_output(output, plot=False)

experiments['allCBA'] = experiment_allCBA



def experiment_cbaSSP5():
    for p_values_num in [1000]:
        for p_values_max_rel in [0.5]:
            for K_values_num in [150]:

                output = full_run_structured(Params(
                    damage='damageTol2009',
                    K_values_num=K_values_num, CE_values_num=5000, p_values_num=p_values_num, p_values_max_rel=p_values_max_rel,
                    cost_level='best', SSP='SSP5', carbonbudget=0, r=0.05,
                    TCRE=0.62e-3,
                    T = 185+50, t_values_num = int(235/5)+1,
                    useCalibratedGamma=True,
                    runname="Experiment cbaSSP5 %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget %p_values_num %p_values_max_rel %K_values_num %CE_values_num %T",
                    shortname="%p_values_num %p_values_max_rel K=%K_values_num CE=%CE_values_num %T"
                ))
                export_output(output, plot=False)

experiments['cbaSSP5'] = experiment_cbaSSP5


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



# Tests:

# from carbontaxdamages.economics import *
# gamma_val('SSP2', 2.0, 0.95, 'best') * 3
