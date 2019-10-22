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

            carbonbudget = 1000, # GtCO2
            relativeBudget = False,

            progRatio = 0.82,
            minEmissions = -300,
            maxReductParam = 0.05 +100,

            CE_values_num = 800,
            E_values_num = 2,
            E_min_rel = -3, E_max_rel = 2.5,

            K_values_num = 50,
            K_min = 0, K_max = 3000,

            p_values_num = 1000,
            p_values_max = 1.5 * 1500.0,

            T = 85,
            t_values_num = int(85/5),

            T0 = 0.909, # Temperature in 2010
            TCRE = 0.62e-3, # degC / GtCO2

            r = 0.05,

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
