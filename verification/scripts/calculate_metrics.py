'''
Bent Harnist 2022 (FMI)

Version 0.2

Script for calculating prediction skill metrics for wanted methods / models
and saving them to npy files for fast access. Predictions and measurements are read
from an hdf5 file and advancement of calculations is saved
in the "done" csv file, enabling us to continue calculations if they are once stopped
without having to redo them all.
'''

import argparse
import yaml
import logging

from attrdict import AttrDict
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from pincast_verif.metrics_calculator import MetricsCalculator



def run(config: AttrDict, config_path : str):
    
    metrics_calculator = MetricsCalculator(config=config, config_path=config_path)
    log_fn = config.logging_path.format(id=config.exp_id)
    logging.basicConfig(filename=log_fn, level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logging.captureWarnings(True)

    if config.debugging: 
        import random
        random.seed(12345)
        samples_left = random.sample(population=metrics_calculator.timestamps, k=config.debugging)
    else:
        samples_left = metrics_calculator.samples_left 
    
    # Main Loop
    if config.accumulate:
        # No parallelization
        if config.n_chunks == 0:
            for sample in tqdm(samples_left):
                done_df, outdict = metrics_calculator.accumulate(
                    sample=sample,
                    done_df=metrics_calculator.done_df,
                    outdict=metrics_calculator.outdict)
                metrics_calculator.update_state(done_df=done_df,outdict=outdict)
                metrics_calculator.save_contingency_tables()
                metrics_calculator.save_done_df()
        # parallelized code
        # does not allow for state saving in-between samples or chunks
        else:
            pbar = ProgressBar()
            pbar.register()
            metrics_calculator.parallel_accumulation()
            metrics_calculator.save_contingency_tables()
            metrics_calculator.save_done_df()

    metrics_data = metrics_calculator.compute()
    metrics_calculator.save_metrics(metrics_dict=metrics_data)


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(
       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config_path", type=str, help="Configuration file path")
    args = argparser.parse_args()

    with open(args.config_path, "r") as f:
        config = AttrDict(yaml.safe_load(f))

    run(config, args.config_path)

