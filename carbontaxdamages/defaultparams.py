######
# Default parameters
######

from collections import namedtuple


class Params:
    default_params = None

    obj = None

    def __init__(self, **kwargs):

        self.default_params = dict(

            beta = 2.0,
            gamma = 1500.0,
            useCalibratedGamma = False,
            cost_level = 'best',

            carbonbudget = 1000, # GtCO2
            relativeBudget = False,

            progRatio = 0.82,
            exogLearningRate = 0.0,

            # If True, use progRatio to calculate corresponding exog learning rate.
            # The endogenous learning (LBD) is then ignored
            useCalibratedExogLearningRate = False,
            minEmissions = -20, # Default is at most 20 GtCO2/yr net negative emissions
            maxReductParam = 0.05 +100,
            end_of_run_inertia = 2200,
            maxReductParamPositive = 1.0,

            CE_values_num = 800,
            E_values_num = 2,
            E_min_rel = -3, E_max_rel = 2.5,

            K_values_num = 50,
            K_min = 0, K_max = 3000,

            p_values_num = 1000,
            p_values_max_rel = 1.5,

            T = 85,
            t_values_num = int(85/5)+1,

            T0 = 0.909, # Temperature in 2010
            TCRE = 0.62e-3, # degC / GtCO2

            r = 0.015,
            elasmu = 1.45,
            discountConsumptionFixed = False, # Only used when maximise_utility is False

            maximise_utility = True,

            SSP = 'SSP2',
            K_start = 223.0,

            fastmath = True,

            damage = "nodamage",

            runname = "default",
            shortname = "default"
        )

        for key, value in kwargs.items():
            if key not in self.default_params:
                raise KeyError("Key " + str(key) +" not a valid argument")
            self.default_params[key] = value

        ## Create namedtuple
        ParamsObj = namedtuple('ParamsObj', sorted(self.default_params))
        self.obj = ParamsObj(**self.default_params)

    def __repr__(self):
        return "Params("+str(self.default_params)+")"

    def dict(self):
        return self.default_params
