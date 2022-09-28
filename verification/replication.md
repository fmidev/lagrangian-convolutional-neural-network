## Replication Instructions

## LINDA/Lagrangian Persistence nowcasts

### Steps to follow

- Create a configuration folder for computing nowcasts named arbitrarily, e.g. `arbitrary_folder_name` under `config`.
- Make a copy of `config/example/prediction_builders/advection_prediction.yaml` in that folder for each model to compute nowcasts for, each configuration separately, if many are run.
  - For example: `LINDA_config_1.yaml`, `LINDA_config_2.yaml`, `LP.yaml`
- modify each of these according to your needs, taking care that:
  - `datelist_path`, `data_source_name`, `input`, `preprocessing`, `postprocessing` are matching for comparable results
  - `save` parameters should most likely be the same, and set according to your choices.
  - `hdf5_path` should be different for each model.
  - `nowcast` parameters should differ for each model. Here the actual nowcast method and its parameters are defined.
- Run `python scripts/run_pysteps_swap_predictions.yaml {config_folder}` where `{config_folder}` indicates your arbitrarily name folder name.
- Prediction HDF5 archives should appear where you set them to be saved. Check for the integrity of the nowcasts.
