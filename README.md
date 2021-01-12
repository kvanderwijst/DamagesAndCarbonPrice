# Integrated Assessment Model with full literature range for cost-effectiveness and cost-benefit scenarios

To recreate all the output files for of the experiments, run:
```
$ python run_all.py all allCBA
```
using Python 3 (the keywords `all` and `allCBA` refer to which set of scenarios you want to run: `all` is for all the cost-effectiveness runs, `allCBA` for all the cost-benefit runs).

The core of the model is the `carbontaxdamages`-folder, containing the default parameters, the socio-economic data, the main optimisation and Cobb-Douglas routine (`run.py`) and some extra necessary functions.


## Directly using output data

The main data used in the paper, created using the above command, is contained in the files `experiment_all.csv` and `experiment_allcba.csv` for respectively the cost-minimising and the cost-benefit runs. These files are used to create the figures of the paper using the notebooks `2. Cost-effectiveness visualisation.ipynb` and `3. Cost-benefit visualisation.ipynb`.
