from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output

output = full_run_structured(Params(
    T = 85,
    t_values_num = int(85/5)+1,
    damage='damageHowardTotal',
    K_values_num=15, CE_values_num=1500, p_values_num=500,
    end_of_run_inertia=2085, E_values_num=100,
    cost_level='p05', SSP='SSP5', carbonbudget=1750, elasmu=1.45,
    TCRE=0.00086,
    maximise_utility=True, r=0.03,
    useCalibratedGamma=True,
    runname="Experiment test1 %SSP 2120 %damage TCRE %TCRE cost %cost_level r=%r %minEmissions budget 2C utility %maximise_utility elasmu %elasmu",
    shortname="%SSP %damage %TCRE %cost_level %r"
))
export_output(output, plot=True)
