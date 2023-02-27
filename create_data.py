"""Create a timeseries of Lagrangian input data for the given timestamp.

Usage:
    python create_data.py <timestamp> <config> --nworkers <nworkers> --datelist-path <datelist_path>

Example:
-------
    python create_data.py 202003011200 lcnn-predict-realtime --nworkers 1 --datelist-path datelists/realtime.txt

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import sh

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("timestamp", type=str, help="Nowcast timestamp (YYYYMMDDHH)")
    argparser.add_argument("config", type=str, help="Configuration folder sub-path")
    argparser.add_argument("--nworkers", type=int, default=1, help="Number of workers")
    argparser.add_argument(
        "--datelist-path",
        type=str,
        help="Path to datelist file. The split specified in the config file is assumed to be 'realtime'.",
        default="datelists/realtime.txt",
    )
    args = argparser.parse_args()

    # Create datetime file
    timestamp = pd.Timestamp(datetime.strptime(args.timestamp, "%Y%m%d%H%M"))
    datelist_path = Path(args.datelist_path).resolve()
    times = pd.date_range(end=timestamp, freq="5T", periods=5)
    times.to_series().to_csv(datelist_path, index=False, header=False)

    # Create input data
    # python transform_fmi_composite_to_lagrangian.py lcnn-predict-realtime realtime --nworkers 1
    proc = sh.python(
        "transform_fmi_composite_to_lagrangian.py",
        args.config,
        "realtime",
        "--nworkers",
        args.nworkers,
    )
