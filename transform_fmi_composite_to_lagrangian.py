"""Apply transformation to Lagrangian coordinates."""
import argparse
import numpy as np
import dask
from pathlib import Path
import logging
from scipy.ndimage import uniform_filter

from datasets import FMIComposite
import utils


def worker(idx, dataset, lconf):
    past_data, future_data, _ = dataset[idx]
    dates = dataset.get_window(idx)
    R = np.vstack([past_data.detach().numpy(), future_data.detach().numpy()]).squeeze()
    # Remove nan values
    R[~np.isfinite(R)] = 0

    # Transform dBZ to mm/h
    a_r = lconf.rainrate_conversion.a
    b_r = lconf.rainrate_conversion.b

    # First to linear
    R = 10 ** (R * 0.1)
    # Then to mm/h
    R = (R / a_r) ** (1 / b_r)  # fixed

    # Remove non-precipitation values
    R[R < lconf["precip_threshold_mmh"]] = 0

    # Common time index (in 1-based indices)
    # For Lagrangian transform
    t0_lagr = past_data.shape[0]
    # After Lagrangian transform, when first fields are discarded
    if lconf["oflow_params"]["update_advfield"]:
        t0_ = t0_lagr - lconf["oflow_params"]["oflow_history_length"] + 1
        dates_ = dates[lconf["oflow_params"]["oflow_history_length"] - 1 :]
    else:
        t0_ = t0_lagr
        dates_ = dates

    R_lagrangian, mask_adv, advfields = utils.transform_to_lagrangian(
        R,
        t0_lagr,
        dates,
        advfield_length=lconf["oflow_params"]["oflow_history_length"],
        update_advfield=lconf["oflow_params"]["update_advfield"],
        optflow_method=lconf["oflow_params"]["oflow_method"],
        oflow_kwargs=lconf["oflow_params"][lconf["oflow_params"]["oflow_method"]],
        extrap_kwargs=lconf["oflow_params"]["extrap_kwargs"],
    )

    if (lconf["output"]["display_freq"] > 0) and (
        idx % lconf["output"]["display_freq"] == 0
    ):
        R_euler = utils.transform_to_eulerian(R_lagrangian, t0_, dates_, advfields)

        utils.plot_lagrangian_fields(
            dates_,
            t0_,
            R,
            R_lagrangian,
            R_euler,
            advfields,
            mask_adv,
            outpath=lconf["output"]["fig_path"],
            min_dbz=lconf["precip_threshold_dbz"],
        )
    R_lagrangian[:, ~mask_adv] = np.nan

    # Fill possible negative values with moving mean
    if np.any(R_lagrangian < 0):
        for i in range(R_lagrangian.shape[0]):
            mean_ = uniform_filter(R_lagrangian[i], size=3, mode="constant")
            R_lagrangian[i][np.where(R_lagrangian[i] < 0)] = mean_[
                np.where(R_lagrangian[i] < 0)
            ]

        # If any negative remain, set to 0
        R_lagrangian[R_lagrangian < 0] = 0

    # Transform mm/h back to dBZ for storing
    R_lagrangian = a_r * R_lagrangian ** (b_r)  # to z
    R_lagrangian = 10 * np.log10(R_lagrangian)  # to dBZ

    utils.save_lagrangian_fields_h5_with_advfields(
        R_lagrangian, dates_, t0_, advfields, lconf["output"]
    )

    del R, R_lagrangian, past_data, future_data, advfields


def main(dataset, lconf):
    # Iterate over dataset and calculate Lagrangian transform
    res = []
    n_items = len(dataset)
    for idx in range(n_items):
        # Don't run existing files
        dates = dataset.get_window(idx)

        if lconf["oflow_params"]["update_advfield"]:
            # Consider the extra fields in common time
            common_time = dates[
                dataset.num_frames_input
                + lconf["oflow_params"]["oflow_history_length"]
                - 2
            ]
        else:
            common_time = dates[dataset.num_frames_input - 1]

        fn = Path(
            lconf["output"]["path"].format(
                year=common_time.year,
                month=common_time.month,
                day=common_time.day,
            )
        ) / Path(
            lconf["output"]["filename"].format(
                commontime=common_time,
            )
        )

        if fn.exists():
            continue

        res.append(
            dask.delayed(worker)(
                idx,
                dataset,
                lconf,
            )
        )

    logging.info(f"Running {len(res)} datasets!")

    scheduler = "processes" if args.nworkers > 1 else "single-threaded"
    res = dask.compute(*res, num_workers=args.nworkers, scheduler=scheduler)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument("split", type=str, help="split")
    argparser.add_argument(
        "--nworkers",
        type=int,
        default=1,
        help="Number of workers",
    )

    args = argparser.parse_args()

    confpath = Path("config") / args.config
    dsconf = utils.load_config(confpath / "lagrangian_transform_datasets.yaml")[
        "FMIComposite"
    ]
    lconf = utils.load_config(confpath / "lagrangian_transform_params.yaml")
    logconf = utils.load_config(confpath / "output.yaml")

    if lconf["oflow_params"]["update_advfield"]:
        dsconf["input_block_length"] += (
            lconf["oflow_params"]["oflow_history_length"] - 1
        )

    dataset = FMIComposite(split=args.split, **dsconf)

    utils.setup_logging(logconf.logging)

    main(dataset, lconf)
