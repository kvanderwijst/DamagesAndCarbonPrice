# Integrated Assessment Model with full literature range for cost-effectiveness and cost-benefit scenarios

To create all the output files for of the experiments, run:
```$ python run_all.py all allCBA
```
(the keywords `all` and `allCBA` refer to which set of scenarios you want to run: `all` is for all the cost-effectiveness runs, `allCBA` for all the cost-benefit runs).

The core of the model is the `carbontaxdamages`-folder, containing the default parameters, the socio-economic data, the main optimisation and Cobb-Douglas routine (`run.py`) and some extra necessary functions.
