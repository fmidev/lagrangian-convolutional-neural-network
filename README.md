# L-CNN model development and analysis

This repository contains the source code for the article [_Advection-free Convolutional Neural Network for Convective Rainfall Nowcasting_](https://doi.org/10.1109/JSTARS.2023.3238016) by Jenna Ritvanen, Bent Harnist, Miguel Aldana, Terhi Mäkinen, and Seppo Pulkkinen, published in _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_.

[![DOI](https://zenodo.org/badge/541975069.svg)](https://zenodo.org/badge/latestdoi/541975069)

For documentation of the L-CNN method, refer to the article. For questions related to the code, contact Jenna Ritvanen (jenna.ritvanen[at]fmi.fi).

The workflow for replicating the results in the article is roughly:

1. [Create the training, validation and test datasets in Lagrangian coordinates.](#creating-lagrangian-datasets)
2. [Train the L-CNN model.](#training-the-l-cnn-model)
3. [Create nowcasts for the L-CNN model.](#creating-nowcasts-for-the-l-cnn-model)
4. Create the reference model nowcasts
   1. Train the RainNet model and run nowcasts for it. See the accompanying repository [fmidev/rainnet](https://github.com/fmidev/rainnet) for details.
   2. Create nowcasts for the LINDA and extrapolation nowcast models. See [`verification/replication.md`](verification/replication.md) for instructions and [`config/p25-extrap-linda-whole-domain-optical-flow`](config/p25-extrap-linda-whole-domain-optical-flow) for configuration files.
5. [Run verification statistics computation and visualize the results.](#running-verification-results)

## Scripts

| Script                                     | Description                                                                                                                                                                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `transform_fmi_composite_to_lagrangian.py` | Script for transforming FMI radar composite to Lagrangian coordinates. Reads parameters from `lagrangian_transform_datasets.yaml` and `lagrangian_transform_params.yaml` configuration files.                                    |
| `train_model.py`                           | Script for training the L-CNN model. Dataset parameters read from `lagrangian_datasets.yaml` and model parameters from `lcnn.yaml`.                                                                                              |
| `predict_model.py`                         | Script for producing predictions from the `datelists/fmi_rainy_days_bbox_predict.txt` datelist. Dataset parameters read from `lagrangian_datasets.yaml` and model parameters from `lcnn.yaml`.                                   |
| `plot_example_nowcasts.py`                 | Script for plotting example nowcasts. Input parameters nowcast time and config file name.                                                                                                                                        |
| `plot_example_nowcasts_gif.py`             | Script for plotting example nowcasts as GIF animations. Input parameters nowcast time and config file name. Config file has same structure as `plot_example_nowcasts.py`, but output filename and figure size are not respected. |

## Creating Lagrangian datasets

The Lagrangian-transformed datasets are created with the script `transform_fmi_composite_to_lagrangian.py`. The configuration for the transformation is given in two configuration files [`lagrangian_transform_params.yaml`](config/datatransform-5-6/lagrangian_transform_params.yaml) for the transformation parameters and [`lagrangian_transform_datasets.yaml`](config/datatransform-5-6/lagrangian_transform_datasets.yaml) for the dataset configuration.

The script is run with

```bash
python transform_fmi_composite_to_lagrangian.py <config-sub-path> <dataset-split> --nworkers <workers>
```

where `<config-sub-path>` is a sub-directory of the `config` directory where the configuration files are located, `<dataset-split>` is the name of the dataset to be generated (that will be injected to the `{split}` placeholder in the `date_list` variable in `lagrangian_transform_datasets.yaml`), and `<workers>` indicates the number of dask processes used to run the transformation.

## Training the L-CNN model

The L-CNN model is trained with the `train_model.py` script. The script is going to read 5 configuration files:

- [`lcnn.yaml`](config/lcnn-rmse-train-30lt/lcnn.yaml) for the model and training parameters
- [`lagrangian_datasets.yaml`](config/lcnn-rmse-train-30lt/lagrangian_datasets.yaml) for the dataset parameters
- [`log_nowcast_callback.yaml`](config/lcnn-rmse-train-30lt/log_nowcast_callback.yaml) for logging parameters
- [`nowcast_metrics_callback.yaml`](config/lcnn-rmse-train-30lt/nowcast_metrics_callback.yaml) for verification metrics calculated during training
- [`output.yaml`](config/lcnn-rmse-train-30lt/output.yaml) for random logging outputs

The training will automatically use the dataset list with `train` injected to `{split}` placeholder in the `date_list` variable in `lagrangian_datasets.yaml`.

In SLURM-based machines, the training can be run with e.g.

```bash
#!/bin/bash
#SBATCH --account=<project_no>
#SBATCH --job-name=lcnn_diff_rmse_30lt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=output_%j.txt
#SBATCH --error=errors_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

srun python train_model.py <config-sub-path> &> lcnn-train-rmse-30lt.out

seff $SLURM_JOBID
```

where `<config-sub-path>` is a sub-directory of the `config` directory where the configuration files are located.

Note that the packages in [`environment.yml`](environment.yml) should be installed, if they are not installed in the module environment.

The training will logged in Tensorboard logger that can be viewed through the dashboard. The logs are located at `logs/train_{configpath}`.

## Creating nowcasts for the L-CNN model

The L-CNN model is run with the `predict_model.py` script. The script is going to read 5 configuration files:

- [`lcnn.yaml`](config/lcnn-rmse-train-30lt/lcnn.yaml) for the model and training parameters
- [`lagrangian_datasets.yaml`](config/lcnn-rmse-train-30lt/lagrangian_datasets.yaml) for the dataset parameters
- [`log_nowcast_callback.yaml`](config/lcnn-rmse-train-30lt/log_nowcast_callback.yaml) for logging parameters
- [`nowcast_metrics_callback.yaml`](config/lcnn-rmse-train-30lt/nowcast_metrics_callback.yaml) for verification metrics calculated during training
- [`output.yaml`](config/lcnn-rmse-train-30lt/output.yaml) for random logging outputs

In SLURM-based machines, the training can be run with e.g.

```bash
#!/bin/bash
#SBATCH --account=<project_no>
#SBATCH --job-name=lcnn_diff_rmse_30lt_pred
#SBATCH --partition=fmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=output_%j.txt
#SBATCH --error=errors_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

srun python predict_model.py <path-to-model-checkpoint>.ckpt <config-sub-path> -l <dataset-split>  &> lcnn-rmse-pred.out

seff $SLURM_JOBID
```

where `<config-sub-path>` is a sub-directory of the `config` directory where the configuration files are located, and `<dataset-split>` is the name of the dataset to be run (that will be injected to the `{split}` placeholder in the `date_list` variable in `lagrangian_datasets.yaml`).

## Running verification results

The verification results need to be run in the `verification` subfolder.

First, calculating the metrics is run with the [`verification/scripts/calculate_metrics.py`](verification/scripts/calculate_metrics.py) script using the configuration file [`calculate_metrics.yaml`](verification/config/lcnn-article-21082022/calculate_metrics.yaml).

Second, the results are visualized with the [`verification/scripts/plot.py`](verification/scripts/plot.py) script using the configuration file [`plot.yaml`](verification/config/lcnn-article-21082022/plot_article_figs.yaml).

Note that the name of these configuration files does not matter, as the full path is given to the script.

An example SLURM job for calculating the metrics is given below. Note that in this example the `calculate_metrics.py` script has been copied to the `verification` directory to solve some issue with paths in the SLURM environment. When running the script on regular systems, this was not necessary.

```bash
#!/bin/bash
#SBATCH --account=<project_no>
#SBATCH --job-name=lcnn_rmse_30lt_verif
#SBATCH --partition=fmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=output_%j.txt
#SBATCH --error=errors_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL

# cd verification

module load pytorch

export PYTHONPATH="${PYTHONPATH}:$(pwd -P)"

srun python calculate_metrics.py <config-path>/calculate_metrics.yaml  &> lcnn-mssim-30lt-verif.out

seff $SLURM_JOBID
```

where `<config-path>/calculate_metrics.yaml` is the full path to the configuration file.

Finally, the results can be visualized with

```bash
python scripts/plot.py <config-path>/plot.yaml
```

where `<config-path>/plot.yaml` is the full path to the configuration file.

## Plotting example nowcasts

Case figures of nowcasts can be plotted with

```bash
python plot_example_nowcasts.py plot_example_nowcasts.yaml <YYYYMMDDHHMM>
```

and GIF animations with

```bash
python plot_example_nowcasts_gif.py plot_erad_gifs.yaml <YYYYMMDDHHMM>
```

where the date placeholder should be replaced with the desired time. Note that the configuration files are the same, but the GIF script does not respect the output filename and figure size parameters.
