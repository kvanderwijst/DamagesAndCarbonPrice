# Integrated Assessment Model with full literature range for cost-effectiveness and cost-benefit scenarios


## Directly using output data

The main data used in the paper is contained in the files `output/experiment_allcarbonbudget.csv` and `experiment_allcba.csv` for respectively the carbon budget and the cost-benefit runs. These files are used to create the figures of the paper using the notebooks `2. Cost-effectiveness visualisation.ipynb` and `3. Cost-benefit visualisation.ipynb`.

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
 - `run.py`: the main optimisation and Cobb-Douglas routine.

## Rerun the calibration

The MAC curve is calibrated to AR5 consumption loss data. The calibrated values can be found in `carbontaxdamages/data/calibrated_gamma_SSP_combined.csv`. To recreate these values, run `calibrate.py` in the main folder.

Similarly, the resulting damage function using the Burke et al (2015) data can be found in `carbontaxdamages/data/Burke_damages.csv`, and can be recreated using `make_Burke_damage_functions.py` in the main folder.
