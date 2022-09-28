"""
    This script will run nowcasting predictions
    for advection based deterministic methods implemented in pysteps, with multiple different configurations
    
    Working (tested) prediction types: 
    - extrapolation
    - S-PROG
    - LINDA
    - ANVIL
    - STEPS

    Usage requires: 
    1) Having a workable pysteps installation with
    .pystepsrc configured. 
    2) (Optionally) modifying your AdvectionPrediction class to
    satisfy requirements.
    3) Setting configuration files for each prediction experiment
    to be run, putting them in the folder passed as an argument
"""
import argparse
import sys
import os
from attrdict import AttrDict
import yaml
from typing import Sequence
import argparse
from pathlib import Path 

import h5py
from tqdm import tqdm

from pincast_verif import AdvectionSwapPrediction
from pincast_verif import io_tools

#temp for except handling
import dask


def run(builders : Sequence[AdvectionSwapPrediction]) -> None:
    
    date_paths = [builder.date_path for builder in builders]
    if any(path != date_paths[0] for path in date_paths):
        raise ValueError("The datelists used must be the same for all runs,\
                        Please check that the paths given match.")

    print(date_paths)
    timesteps = io_tools.read_file(date_paths[0])
    output_dbs = [h5py.File(builder.hdf5_path, 'a') 
                  for builder in builders]

    for t in tqdm(timesteps):
        for i, builder in enumerate(builders):
            print(f"sample {t} ongoing...")
            group_name = builder.save_params.group_format.format(
                timestamp = io_tools.get_neighbor(time=t, distance=builder.input_params.num_next_files),
                method = builder.nowcast_params.nowcast_method
            )
            group = output_dbs[i].require_group(group_name)
            if len(group.keys()) == builder.nowcast_params.n_leadtimes:
                continue
            print(f"Running predictions for {builder.nowcast_params.nowcast_method} method.")
            sys.stdout = open(os.devnull, 'w')
            #try:
            nowcast = builder.run(t)
            #except: # indexError, or other error (mainly with LINDA and/or dask)
            #    sys.stdout = sys.__stdout__
            #    continue
            sys.stdout = sys.__stdout__
            builder.save(nowcast=nowcast,group=group,
                    save_parameters=builder.save_params)

    for db in output_dbs:
        db.close()

def load_config(path : str):
    with open(path, "r") as f:
        config = AttrDict(yaml.safe_load(f))
    return config


if __name__ == "__main__" : 

    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config", type=str, help=
                           "Configuration folder path, contains \
                           one YAML configuration file per forecast \
                           type that is to be computed."
                           )
    args = argparser.parse_args()

    config_dir = Path("config") / args.config
    config_filenames = config_dir.glob("*.yaml")
    configurations = [load_config(filename) for filename in config_filenames]
    predictor_builders = [AdvectionSwapPrediction(config = config)
                         for config in configurations]
    run(builders=predictor_builders)
