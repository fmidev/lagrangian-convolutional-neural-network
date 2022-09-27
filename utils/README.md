# Utils

This repository contains utility functions to be used when implementing deep learning models.

Currently, the following utilities exist:

* `config.py`:
  * `load_config`: load configurations from yaml-file as `attrdict.AttrDict`
* `logging.py`:
  * `setup_logging`: setup logging with `logging` module. Required parameters, given in dictionary, are `level` for logging level,  `format` for log message format, `dateformat` for date format in log message, and `filename` for output file name (if not given, logging outputs to `std`)
* `radar_image_plots.py`:
  * `plot_1h_plus_1h_timeseries`: Plots a timeseries of dBZ figures for the given model. Assumes that the model takes 5 input images and outputs 15.
  * `plot_dbz_image`: Plots individual dBZ image.
  * `debug_training_plot`: Plot a debug plot of model input, output and targets.
* `verification_score_plots.py`:
  * `plot_cont_scores_against_leadtime`: Plot continuous scores against leadtime. Scores are in `pandas.DataFrame` where both the scores and `Leadtime` are given as columns.
  * `plot_cat_scores_against_leadtime`: Plot categorical scores against leadtime with multiple thresholds. Scores are in `pandas.DataFrame` where both the scores and `Leadtime` are given as columns. Additionally, thresholds need to be given `Threshold` column.
  * `plot_fss_against_leadtime`: Plot FSS scores, for possibly several thresholds, against leadtime. Scores are in `pandas.DataFrame` where both the scores and `Leadtime` are given as columns. Additionally, thresholds need to be given `Threshold` column and spatial scales in `Scale` column.
