# verification
Scripts and tools for computing verification metrics.

- `scripts` contains callable end-user scripts
- `pincast_verif` contains reusable components that are used by above scripts
- `datelists` contains lists of timestamps used for experiments
- `config` contains example configuration files and you can put yours in there too.

## How to use this code ? 

1. First and foremost: Clone the repository and make sure all the dependencies are installed.
2. Install the package locally with running`pip install -e .` in your conda/virtual environment from the repository root folder. 
3. Setup your Pysteps configuration locally as instructed in https://pysteps.readthedocs.io/en/stable/user_guide/set_pystepsrc.html
4. Proceed further now ...

### I want to make predictions.

*Existing `PredictionBuilder` instances do not use GPU!* 

1. Build your own prediction builder object inheriting `PredictionBuilder` or use one defined in `pincast_verif/prediction_builder_instances` if your use case is covered there.
2. Create/find and use a script calling that builder such as `scripts/run_pysteps_predictions.py` or a jupyter notebook depending on your needs.
3. With a script such as `scripts/run_pysteps_predictions.py`, you will get out an HDF5 file containing predictions usable for metrics calculation.

### I want to calculate metrics.

1. If some metric you want to use is lacking from `pincast_verif/metrics`, take example on the other metrics and add it there, to `pincast_verif/metrics/__init__.py`, and as a case in the `get_metric()` function under `pincast_verif/score_tools.py`.
2. Run `scripts/calculate_metrics.py` with a YAML configuration containing your metric parameters. 
3. Intermediate results are saved as "contingency table" dictionaries potentially saved to disk in `.npy` format.
4. Information on samples calculated and samples with missing data is saved in a "done" pandas dataFrame potentially saved to disk in `.csv` format.  
5. Final metrics are saved to disk in `.npy` format, with accompanying metric descriptor text files.  

### I want to plot metrics. 

1. Run `scripts/plot.yaml` with appropriate YAML configuration and potentially modified to plot your custom metrics, or call the plot method of your metric in some other way, like in a jupyter notebook. 
2. Plots will be saved to disk in specified folders.
