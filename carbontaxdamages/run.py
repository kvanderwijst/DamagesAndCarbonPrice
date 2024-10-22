


import numpy as np
import numba as nb
import scipy.optimize
from .interp import *
from .economics import *
from .defaultparams import *
from tqdm import tqdm
import time
import re
import json_tricks

import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.tools as pyt


f4 = nb.float32
f8 = nb.float64
i4 = nb.int32
i8 = nb.int64

f8_1d = nb.typeof(np.array([1.0]))
f8_2d = nb.typeof(np.array([[1.0]]))
f8_3d = nb.typeof(np.array([[[1.0]]]))


def full_run(params_input):

    #### Make parameters available as object
    params = params_input.obj
    # ... and a few as variables
    T = params.T
    damage = damages[params.damage]
    if params.damage == 'damageBurkeWithLag' or params.damage == 'damageBurkeNoLag':
        # Burke damage functions are SSP specific
        damage = damage[params.SSP]
    if params.damage == 'damageGeneral':
        damage = damage(params.damage_coeff)
        params_input.default_params['damage'] = '{}{:.7f}'.format(params.damage,params.damage_coeff)
        
    fastmath = params.fastmath

    t_values = np.linspace(0, T, params.t_values_num)
    t_values_years = params.start_year + t_values
    dt = t_values[1] - t_values[0]

    ##########################
    ##########################
    ## Baseline functions
    ##########################
    ##########################


    SSP_population = params.SSP if params.SSP_population == 'same' else params.SSP_population
    SSP_GDP = params.SSP if params.SSP_GDP == 'same' else params.SSP_GDP
    SSP_emissions = params.SSP if params.SSP_emissions == 'same' else params.SSP_emissions


    population_values = population(t_values_years, SSP_population)
    GDP_values = GDP(t_values_years, SSP_GDP)
    B_values = baseline_emissions(t_values_years, SSP_emissions)
    B_cumulative_values = dt * np.cumsum(B_values)

    CO2_intensity = B_values / GDP_values

    if params.useBaselineCO2Intensity:
        @nb.njit([ f8(i8, f8) ], fastmath=fastmath)
        def B(t_i, Y):
            if Y <= 0:
                Y = GDP_values[t_i]
            return CO2_intensity[t_i] * Y
    else:
        @nb.njit([ f8(i8, f8) ], fastmath=fastmath)
        def B(t_i, Y):
            return B_values[t_i]

    ### TODO:
    @nb.njit([ f8(i8) ], fastmath=fastmath)
    def B_cumulative(t_i):
        return B_cumulative_values[t_i]


    if params.relativeBudget:
        cumulative_until_2100 = np.trapz(
            baseline_emissions(t_values_years[t_values_years <= 2100], SSP_emissions),
            x=t_values_years[t_values_years <= 2100]
        )
        absoluteBudget = cumulative_until_2100 * params.carbonbudget
        absoluteBudgetOld = cumulative_until_2100 * params.carbonbudgetOld
        params_input.default_params['absoluteBudget'] = absoluteBudget
        params_input.default_params['absoluteBudgetOld'] = absoluteBudgetOld
    else:
        absoluteBudget = params.carbonbudget
        absoluteBudgetOld = params.carbonbudgetOld

    # @nb.njit([ f8(i8) ], fastmath=fastmath)
    # def B(t_i):
    #     t = t_values[t_i]
    #     return params.emissions2010 * (1.0 + t/50.0 - 0.2 * (t/50.0)**2)
    # @nb.njit([ f8(i8) ], fastmath=fastmath)
    # def B_cumulative(t_i):
    #     t = t_values[t_i]
    #     return params.emissions2010 * (t + t**2 / 100.0 - 0.2 / (3.0 * 2500.0) * t**3)


    ##########################
    ##########################
    ## Declare variables and sizes
    ##########################
    ##########################

    # Obtain calibrated value of gamma:
    if params.useCalibratedGamma:
        gamma = gamma_val(SSP_GDP, params.beta, params.progRatio, params.cost_level)
        params_input.default_params['gamma'] = gamma
    else:
        gamma = params.gamma


    # Obtain corresponding exogenous learning rate:
    # NOTE: only works in cost-effectiveness setting, with carbon budget
    if params.useCalibratedExogLearningRate:
        total_abatement = B_cumulative(-1) - absoluteBudget
        exogLearningRate = (total_abatement / 38.8 + 1) ** (-np.log2(params.progRatio)/T) - 1
        exogLearningRate = np.round(exogLearningRate, 5)
        progRatio = 1.0
        params_input.default_params['exogLearningRate'] = exogLearningRate
        params_input.default_params['progRatio'] = progRatio
    else:
        progRatio = params.progRatio
        exogLearningRate = params.exogLearningRate

    CE_min, CE_max = 0, B_cumulative(-1)
    CE_factor = (params.CE_values_num - 1.0) / (CE_max - CE_min)
    CE_values = np.linspace(CE_min, CE_max, params.CE_values_num)
    dCE = CE_values[1] - CE_values[0]

    E_max, E_min = params.E_max_rel * B(0,-1), params.E_min_rel * B(0,-1)
    E_factor = (params.E_values_num - 1.0) / (E_max - E_min)
    E_values = np.linspace(E_min, E_max, params.E_values_num)
    dE = E_values[1] - E_values[0]

    K_min, K_max = params.K_min, params.K_max
    K_factor = (params.K_values_num - 1.0) / (K_max - K_min)
    K_values = np.linspace(K_min, K_max, params.K_values_num)
    dK = K_values[1] - K_values[0]

    ### NOTE: x, y, z order is CE, E, K
    ### Data should be accessed as array[CE_i, E_i, K_i]
    CE_values_3d, E_values_3d, K = np.meshgrid(CE_values, E_values, K_values, indexing='ij')

    p_values_max = params.p_values_max_rel * gamma
    p_values = np.linspace(0, p_values_max, params.p_values_num)

    TFP_values = np.zeros_like(GDP_values)

    if params.maximise_utility or params.discountConsumptionFixed:
        r_values = np.ones_like(GDP_values) * params.r
        if params.discountConsumptionFixed:
            params_input.default_params['r'] = 'fixed{:.3f}'.format(params.r)
    else:
        r_values = params.r + params.elasmu * growth_rate(t_values_years, SSP_GDP)
        params_input.default_params['r'] = 'ramsey{:.2f}_{:.3f}'.format(params.elasmu, params.r)

    J = np.zeros((params.t_values_num+1, params.CE_values_num, params.E_values_num, params.K_values_num))
    pStar = np.zeros_like(J)


    ##########################
    ##########################
    ## System functions
    ##########################
    ##########################

    # @nb.njit(f8(f8,f8), fastmath=fastmath)
    # def MACinvLinear(p, factor):
    #     return p / (100.0 * factor)
    #
    # @nb.njit(f8(f8,f8), fastmath=fastmath)
    # def MACintegralLinear(mEnd, factor):
    #     return 100.0 * mEnd * mEnd * 0.5 / factor

    @nb.njit(f8(f8,f8,f8,f8), fastmath=fastmath)
    def MACinvGeneral(p, factor, beta, gamma):
        return (p / (gamma * factor)) ** (1.0 / beta)

    @nb.njit(f8(f8,f8,f8,f8), fastmath=fastmath)
    def MACintegralGeneral(mEnd, factor, beta, gamma):
        return mEnd**(1+beta) * factor * gamma / (1+beta)



    @nb.njit(f8(f8,f8), fastmath=fastmath)
    def MACinv(p, factor):
        return MACinvGeneral(p, factor, params.beta, gamma)

    @nb.njit(f8(f8,f8), fastmath=fastmath)
    def MACintegral(mEnd, factor):
        return MACintegralGeneral(mEnd, factor, params.beta, gamma)

    @nb.njit([ f8(i8,f8) ], fastmath=fastmath)
    def learningFactor(t_i, CE):
        Q = (B_cumulative(t_i) - CE) / 38.8
        LBD_factor = (np.maximum(0.0, Q) + 1.0) ** np.log2(progRatio)
        LoT_factor = 1.0 / (1.0 + exogLearningRate) ** t_values[t_i]
        return LBD_factor * LoT_factor

    @nb.njit([ f8(i8,f8,f8,f8) ], fastmath=fastmath)
    def abatementCosts(t_i, CE, p, Y):
        factor = learningFactor(t_i, CE)
        return B(t_i, Y) * MACintegral(MACinv(p, factor), factor)

    # @nb.njit([ f8(i8,f8,f8) ], fastmath=fastmath)
    # def R(t_i, CE, p):
    #     return abatementCosts(t_i, CE, p) / ((1+params.r)**t)

    @nb.njit([ nb.types.UniTuple(f8,2)(i8,f8,f8,f8,f8) ], fastmath=fastmath)
    def f(t_i, CE, E, p, Y):
        factor = learningFactor(t_i, CE)

        E_next = B(t_i, Y) * (1 - MACinv(p, factor))
        E_next = np.maximum(E_next, params.minEmissions)
        E_next = np.maximum(E_next, E - dt * params.maxReductParam)

        CE_next = CE + dt * E_next
        return E_next, CE_next



    ##########################
    ##########################
    ## Economic module
    ##########################
    ##########################


    alpha = 0.3             # Power in Cobb-Douglas production function

    @nb.njit([ f8(f8,f8,f8,f8) ])
    def calc_GDP_gross(TFP, L, K, alpha):
        return TFP * (L/1e3)**(1-alpha) * K**alpha

    @nb.njit(nb.types.UniTuple(f8,12)(f8,i8,f8,f8,f8, f8,nb.boolean, f8_1d))
    def economicModule(t, t_i, CE, E, K, p, calibrate, TFP_values):
        savingsRate = 0.21      #
        dk = 0.1                # Depreciation of capital

        L = population_values[t_i] * 1e-6
        if calibrate:
            TFP = GDP_values[t_i] / ((L/1e3)**(1-alpha) * K**alpha)
        else:
            TFP = TFP_values[t_i]

        Y_gross = calc_GDP_gross(TFP, L, K, alpha)

        # Total abatement costs
        abatement = 1e-3 * abatementCosts(t_i, CE, p, Y_gross)

        # Current temperature and damages
        temperature = params.T0 + params.TCRE * CE

        if calibrate:
            damageFraction = 0.0
        else:
            damageFraction = (damage(temperature) - damage(params.T0))
        Y = Y_gross * (1-damageFraction)

        # Abatement costs are removed as 21% from investments, rest from consumption
        investments_gross = savingsRate * Y

        if savingsRate * abatement > investments_gross: # Shit, the world goes bankrupt
            investments = 0.0
            consumption = 0.0
            utility = 0.0
            NPV = 0.0
            K_next = 0.0

        else:

            investments = investments_gross - savingsRate * abatement
            consumption = Y - investments_gross - (1-savingsRate) * abatement

            r = r_values[t_i]

            discount_factor = 1.0 / (1 + r)**t

            # Utility (equal to log of per capita consumption):
            # (DICE uses population in millions, consumption in )
            utility = ((consumption * 1000 / L) ** ( (1-params.elasmu) ) - 1) / (1-params.elasmu) - 1

            if params.maximise_utility:
                NPV = utility * L * discount_factor
            else:
                NPV = consumption * discount_factor


            K_next = (1-dk)**dt * K + dt * investments

        if calibrate:
            return TFP, K_next, temperature, Y_gross, damageFraction, investments, consumption, TFP, Y, utility, L, r

        return NPV, K_next, temperature, Y_gross, damageFraction, investments, consumption, TFP, Y, utility, L, r


    ### Calibrate the TFP such that GDP matches SSP GDP
    def calibrateTFP():
        K = params.K_start
        for t_i, t in enumerate(t_values):
            TFP, K, *rest = economicModule(t, t_i, 0.0, 0.0, K, 0.0, calibrate=True, TFP_values=TFP_values)
            TFP_values[t_i] = TFP


    # pyo.iplot([go.Scatter(y=TFP_values)])


    ##########################
    ##########################
    ## Other functions
    ##########################
    ##########################


    def reset():
        J[:] = np.zeros_like(J)
        pStar[:] = np.zeros_like(J)

    def setPenalty(abs_budget):
        too_high = CE_values_3d > abs_budget
        J[-1, too_high] = 500. * (CE_values_3d[too_high] - abs_budget)**3


    #### Calculate limits
    # def minPossibleCE_CE(CE_t, t, t_i):
    #     CE = CE_t
    #     for s_i in range(t_i, t_values_num):
    #         s = t_values[s_i]
    #         CE += B(s) * (1 - MACinv(p_values_max, learningFactor(s, CE)))
    #     return CE / B_cumulative(T)
    #
    # def minPossibleCE(t, t_i, CB):
    #     if minPossibleCE_CE(B_cumulative(t), t, t_i) < CB:
    #         return np.nan
    #     return scipy.optimize.brentq(lambda x: (minPossibleCE_CE(x, t, t_i) - CB), a=0, b=B_cumulative(T))
    #
    # limits = [minPossibleCE(t, t_i, 0.5) for t_i, t in enumerate(t_values)]


    ##########################
    ##########################
    ## Optimal control
    ##########################
    ##########################

    @nb.njit(f8(f8,f8,f8,f8_3d), fastmath=fastmath)
    def getValue(CE, E, K, array):
        return trilinear_interpolate(
            CE, E, K,
            CE_min, dCE, CE_factor, params.CE_values_num,
            E_min, dE, E_factor, params.E_values_num,
            K_min, dK, K_factor, params.K_values_num,
            array
        )

    @nb.njit(nb.types.UniTuple(f8,2)(f8,i8,i8,i8,i8,f8_3d, f8_1d), fastmath=fastmath)
    def calcOptimalPolicy_single(t, t_i, CE_i, E_i, K_i, J_t_next, TFP_values):
        CE = CE_values[CE_i]
        E = E_values[E_i]
        K = K_values[K_i]

        if CE > 1.2 * B_cumulative(t_i) and CE > 10:
            return np.nan, np.nan

        i = 0
        currValue = 0.0
        best_p = 0.0

        if t >= (params.budgetYear - params.start_year) and params.noPositiveEmissionsAfterBudgetYear and absoluteBudget > 0:
            # Emissions cannot be positve after budget year
            p_min = gamma * learningFactor(t_i, CE)
            p_values_selection = np.concatenate((np.array([p_min]), p_values[p_values > p_min]))
        else:
            p_values_selection = p_values

        for p in p_values_selection[::-1]:
            NPV_curr, K_next, _, _, _, _, _, _, _, _, _, _ = economicModule(t, t_i, CE, E, K, p, False, TFP_values)
            Y = calc_GDP_gross(TFP_values[t_i], population_values[t_i] * 1e-6, K, alpha)
            E_next, CE_next = f(t_i, CE, E, p, Y)
            J_next = -NPV_curr + getValue(CE_next, E_next, K_next, J_t_next)


            if (
                i == 0 or
                (J_next < currValue and (
                    CE_next < absoluteBudget or t < (params.budgetYear - params.start_year) or absoluteBudget <= 0
                ))
            ):
                currValue = J_next
                best_p = p
            i += 1

        return currValue, best_p


    @nb.njit(nb.types.UniTuple(f8_3d,2)(f8,i8,f8_3d,f8_3d,f8_3d, f8_1d), parallel=True, fastmath=fastmath)
    def calcOptimalPolicy(t, t_i, J_t, J_t_next, pStar_t, TFP_values):
        for CE_i in nb.prange(params.CE_values_num):
            CE = CE_values[CE_i]
            if absoluteBudget > 0 and t > (params.budgetYear - params.start_year) and CE > absoluteBudget:
                penalty = 500. * (CE - absoluteBudget)**3
            else:
                penalty = 0.0
            for E_i in range(params.E_values_num):
                for K_i in range(params.K_values_num):
                    J_val, p_val = calcOptimalPolicy_single(t, t_i, CE_i, E_i, K_i, J_t_next, TFP_values)
                    J_t[CE_i, E_i, K_i] = J_val + penalty
                    pStar_t[CE_i, E_i, K_i] = p_val
        return J_t, pStar_t

    def backwardInduction():
        with tqdm(reversed(list(enumerate(t_values))), total=params.t_values_num) as pbar:
            for t_i, t in pbar:
                J_t, pStar_t = calcOptimalPolicy(t, t_i, J[t_i], J[t_i+1], pStar[t_i], TFP_values)
                J[t_i,:,:,:] = J_t
                pStar[t_i,:,:,:] = pStar_t

    def forward():
        CE, E, K = 0, B(0,-1), params.K_start
        p_path = []
        E_path = []
        rest_all = []
        for t_i, t in enumerate(t_values):
            p = getValue(CE, E, K, pStar[t_i])
            p_path.append(p)
            # Next CE:
            Y = calc_GDP_gross(TFP_values[t_i], population_values[t_i] * 1e-6, K, alpha)
            E, CE = f(t_i, CE, E, p, Y)
            NPV, K, *rest = economicModule(t, t_i, CE, E, K, p, False, TFP_values=TFP_values)
            Y = rest[-4]
            E_path.append(E)
            rest_all.append([B(t_i,-1), CE, K] + rest)
        print(CE, CE / B_cumulative(-1))
        return p_path, E_path, np.array(rest_all)



    calibrateTFP()

    reset()
    if absoluteBudgetOld > 0:
        setPenalty(absoluteBudgetOld)
    # setPenalty(absoluteBudget)


    backwardInduction()

    #pyo.plot([go.Heatmap(z=pStar[:,1,:].T)], filename='output/pStar.html')

    p_path, E_path, rest = forward()

    return p_path, E_path, rest, t_values, J, pStar, B, B_cumulative, t_values_years

def full_run_structured(params, *args, **kwargs):
    p_path, E_path, rest, t_values, J, pStar, B, B_cumulative, t_values_years = full_run(params, *args, **kwargs)
    return {
        'p': p_path,
        'E': E_path,
        'baseline': rest[:,0],
        'cumEmissions': rest[:,1],
        'K': rest[:,2],
        'temp': rest[:,3],
        'Ygross': rest[:,4],
        'damageFraction': rest[:,5],
        'investments': rest[:,6],
        'consumption': rest[:,7],
        'TFP': rest[:,8],
        'Y': rest[:,9],
        'utility': rest[:,10],
        'population': rest[:,11],
        'r': rest[:,12],
        'meta': {
            # 'B': B,
            't_values': t_values,
            't_values_years': t_values_years,
            # 'B_cumulative': B_cumulative,
            'params': params
        }
    }



###################
##
## Plot functions
##
###################




# def plot_output(p_path, E_path, rest, t_values):
#     pyo.plot([go.Scatter(x=t_values, y=path, name=name) for path, name in zip([
#         p_path,
#         E_path
#     ] + list(rest.T),
#         ["Carbon price", "Emissions", "Baseline", "Cumulative emissions",
#             "Capital K", "temperature", "Y_gross", "damageFraction", "investments",
#             "consumption", "TFP", "Y"]
#     )], filename='output/paths.html')



def write_json(outp, name):
	with open(name+'.json', 'w') as outfile:
		json_tricks.dump(outp,outfile)

def replaceFlags(params, name):
    params_regex = re.compile(r'%([\w_]+)')
    params_flags = params_regex.findall(name)
    for flag in params_flags:
        if flag in params.default_params:
            name = name.replace('%'+flag, str(params.default_params[flag]))
    return name

def parseRunName(params):
    runname = replaceFlags(params, params.obj.runname)
    shortname = replaceFlags(params, params.obj.shortname)

    filename = "output/" + runname.replace(' ', '_').replace('/','').replace('\\','').lower().replace('%time', '_' + str(int(time.time())))
    runname = runname.replace('%time', '')

    return runname, shortname, filename

def plot_output(filename, inline=False):
    # Import JSON
    with open(filename+".json", 'r') as fh:
        o = json_tricks.load(fh, preserve_order=False)
    plot_fct = pyo.iplot if inline else pyo.plot
    plot_fct(
        go.Figure(
            data=[go.Scatter(x=o['meta']['t_values_years'], y=path, name=name)
                    for name, path in o.items() if name != 'meta'
            ],
            layout=go.Layout(title=o['meta']['title'])
        ),
        filename=filename+'.html'
    )
    return

def export_output(o, save=True, plot=True, inline=False):
    runname, shortname, filename = parseRunName(o['meta']['params'])
    o['meta']['title'] = runname
    o['meta']['shorttitle'] = shortname


    if save:
        write_json(o, filename)
    if plot:
        plot_output(filename, inline)
