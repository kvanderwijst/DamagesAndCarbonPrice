# Integrated Assessment Model with full literature range for cost-effectiveness and cost-benefit scenarios


## Directly using output data

The main data used in the paper is contained in the files:
 - [`output/experiment_allcarbonbudget.csv`](output/experiment_allcarbonbudget.csv)
 - [`output/experiment_allcba.csv`](output/experiment_allcba.csv)

for respectively the carbon budget and the cost-benefit runs. These files are used to create the figures of the paper using the notebooks `2. Cost-effectiveness visualisation.ipynb` and `3. Cost-benefit visualisation.ipynb`.

## Recreating output files

To recreate all the output files for of the experiments, run:
```
$ python run_all.py all allCBA
```
using Python 3 (the keywords `all` and `allCBA` refer to which set of scenarios you want to run: `all` is for all the cost-effectiveness runs, `allCBA` for all the cost-benefit runs).

The above command creates a set of JSON-files containing the output of the IAM. To transform these to a more useful CSV-file, use the notebook `1. Data preparation.ipynb`.

The core of the model is the `carbontaxdamages`-folder:
 - `defaultparams.py`: the default parameters, which can be changed programmatically when running the model with the above command,
 - `economics.py`: the socio-economic data and damage function definitions
 - `run.py`: the main optimisation and Cobb-Douglas routine (mainly the function `full_run(...)`, the rest are export functions). In this function, a number of sub-functions are defined:
   - `economicModule(...)`: given the value of the state variables and a carbon price at time `t`, calculates the discounted utility at the next time step using the Cobb-Douglas production function
   - `calcOptimalPolicy_single(...)`: given the value of the state variables at time `t` and the optimal discounted utility coming from times `t+1...T` from the backpropagation algorithm, loops over all possible values of the carbon price to calculate the carbon price maximising the discounted utility at time `t`.
   - `calcOptimalPolicy(...)`: calls `calcOptimalPolicy_single(...)` for each value of the state variables at time `t`.
   - `backwardInduction(...)`: runs `calcOptimalPolicy(...)` for each timestep from `T` to `0` backwards in time.
   - `forward()`: once the backward induction step is finished, the model is run forward in time, starting at the initial state variable values to obtain the optimal carbon price path.

The model mainly uses the Numba Python package, compiling Python code to low-level highly efficient code. It is then run in parallel using Numba's automatic parallelisation.

## Rerun the calibration

The MAC curve is calibrated to AR5 consumption loss data. The calibrated values can be found in `carbontaxdamages/data/calibrated_gamma_SSP_combined.csv`. To recreate these values, run `calibrate.py` in the main folder.

Similarly, the resulting damage function using the Burke et al (2015) data can be found in `carbontaxdamages/data/Burke_damages.csv`, and can be recreated using `make_Burke_damage_functions.py` in the main folder.
