"""Utility functions for Lagrangian transform."""
import numpy as np
import h5py
from datetime import datetime
from functools import partial
from pathlib import Path
from pysteps import motion
from pysteps.extrapolation import semilagrangian
from matplotlib import pyplot as plt
from pysteps.visualization.motionfields import quiver
import hdf5plugin
import dask
import pandas as pd
import xarray as xr

def transform_to_eulerian(
    input_fields,
    t0,
    dates,
    advfields,
    extrap_kwargs={},
    prediction_mode=False,
    n_workers=1,
):
    """Transform the input time series to Eulerian coordinates using the given advection field.

    Parameters
    ----------
    input_fields : np.ndarray
        Input precipitation fields, with shape (N, height, width).
    t0 : int
        Number of the "common time" that all other fields are extrapolated to
        (1-based; from 1 ... N).
    dates : list-like
        List of dates corresponding to each field.
    advfields : dict
        Advection fields with structure (startdate, enddate): advfield.
    extrap_kwargs : dict
        Keyword arguments passed to pysteps.extrapolation.semilagragian.extrapolate.
    prediction_mode : bool
        If we're predicting, observed motion fields from timesteps after common_time
        are not used.

    Returns
    -------
    np.ndarray
        Output precipitation fields in Lagrangian coordinates.
    np.ndarray
        Advection mask.
    dict
        Dictionary of advection fields with structure (startdate, enddate): advfield

    """
    n_fields = input_fields.shape[0]
    output_fields = np.empty(input_fields.shape)
    # common_time = dates[t0 - 1]
    input_fields[~np.isfinite(input_fields)] = 0

    def extrapolate_backwards(i, field, advfields):
        # Extrapolate backwards
        # Get advection field
        if len(advfields) == 1:
            advfield = (-1) * list(advfields.values())[0].copy()
        else:
            # Here, get correct advection field based on times
            # if advection field changes
            try:
                # We need advection field that starts with t (since t < t0)
                field_k = [k for k in advfields.keys() if k[0] == dates[i]][0]
                advfield = advfields[field_k].copy()
            except KeyError:
                raise KeyError(
                    f"Correct advection field for time {dates[i]} doesn't exist!"
                )

        output_field = semilagrangian.extrapolate(
            field,
            advfield,
            t0 - i - 1,
            **extrap_kwargs,
        )[-1]
        return i, output_field

    def extrapolate_forwards(i, field, advfields):
        # Extrapolate forwards
        if prediction_mode:
            # Look for the longest motion field interval ending at common_time
            possible_keys = [k for k in advfields.keys() if k[1] == dates[t0 - 1]]
            if len(possible_keys) == 1:
                key = possible_keys[0]
            else:
                i_key = np.argmax(np.diff([*sorted(possible_keys)], axis=0))
                key = sorted(advfields.keys())[i_key]
            advfield = advfields[key]
        elif len(advfields) == 1:
            advfield = list(advfields.values())[0].copy()
        else:
            # Here, get correct advection field based on times
            try:
                # Here we need adv.field that ends at t (since t > t0)
                field_k = [k for k in advfields.keys() if k[1] == dates[i]][0]
                advfield = advfields[field_k].copy()
            except (IndexError, KeyError):
                raise KeyError(
                    f"Correct advection field for time {dates[i]} doesn't exist!"
                )
        output_field = semilagrangian.extrapolate(
            field,
            advfield,
            abs(t0 - i - 1),
            **extrap_kwargs,
        )[-1]
        return i, output_field

    delayed = []
    for i in range(t0 - 1):
        # Extrapolate backwards in time from common time to original time
        delayed.append(
            dask.delayed(extrapolate_backwards)(i, input_fields[i, :], advfields)
        )

    # Field at t0 is not transformed
    output_fields[t0 - 1, :] = input_fields[t0 - 1, :]

    for i in range(t0, n_fields):
        # Extrapolate forwards in time from common time to original time
        # This can also be predicting!
        delayed.append(
            dask.delayed(extrapolate_forwards)(i, input_fields[i, :], advfields)
        )
    scheduler = "processes" if n_workers > 1 else "single-threaded"
    res = dask.compute(*delayed, num_workers=n_workers, scheduler=scheduler)

    for r in res:
        i, field = r
        output_fields[i, :] = field

    # return output_fields, mask_adv, advfields
    return output_fields


def transform_to_lagrangian(
    input_fields,
    t0,
    dates,
    compute_mask=True,
    advfield_length=5,
    update_advfield=False,
    optflow_method="lucaskanade",
    oflow_kwargs={},
    extrap_kwargs={},
):
    """Transform the input time series to Lagrangian coordinates using the given advection field.

    Parameters
    ----------
    input_fields : np.ndarray
        Input precipitation fields, with shape (N, height, width).
    t0 : int
        Number of the "common time" that all other fields are extrapolated to
        (1-based; from 1 ... N).
    dates : list-like
        List of dates corresponding to each field.
    compute_mask : boolean
        Whether to compute advection mask.
    advfield_length : int
        Number of fields used to calculate advection field.
    update_advfield : boolean
        Whether to update advection field. If True, the advection field is calculated
        using advfield_length fields.
        Note that in that case first (advfield_length - 1) fields are discarded.
    oflow_kwargs : dict
        Keyword arguments passed to pysteps.motion methods.
    extrap_kwargs : dict
        Keyword arguments passed to pysteps.extrapolation.semilagragian.extrapolate.

    Returns
    -------
    np.ndarray
        Output precipitation fields in Lagrangian coordinates.
    np.ndarray
        Advection mask.
    dict
        Dictionary of advection fields with structure (startdate, enddate): advfield

    """
    n_fields = input_fields.shape[0]
    output_fields = np.empty(input_fields.shape)

    if compute_mask:
        mask_adv = np.isfinite(input_fields[-1])
    else:
        mask_adv = None

    calc_oflow = partial(motion.get_method(optflow_method), **oflow_kwargs)

    advfields = {}

    if not update_advfield:
        advfield = calc_oflow(input_fields[t0 - advfield_length : t0, ...])
        advfields[(dates[t0 - advfield_length], dates[t0 - 1])] = advfield.copy()
        start_timestep = 0
    else:
        # Add a motion field for the whole input data to be used in prediction
        advfield = calc_oflow(input_fields[advfield_length - 1 : t0, ...])
        advfields[(dates[advfield_length - 1], dates[t0 - 1])] = advfield.copy()
        # If advection field is updated, the first fields are used only to calculate
        # the advection fields
        start_timestep = advfield_length - 1

    for i in range(start_timestep, t0 - 1):
        # Extrapolate forwards
        if update_advfield:
            # Pick correct window for advection fields calculation
            time_slice = slice(i - advfield_length + 1, i + 1)
            date_window = dates[time_slice]
            advfields[tuple(date_window)] = calc_oflow(input_fields[time_slice, ...])
            advfields[tuple(date_window[::-1])] = calc_oflow(
                np.flip(input_fields[time_slice, ...], axis=0)
            )

            # Since here t < t0, we need advection field from t-n ... t
            advfield = advfields[tuple(date_window)]

        output_fields[i, :] = semilagrangian.extrapolate(
            input_fields[i, :],
            advfield,
            t0 - i - 1,
            **extrap_kwargs,
        )[-1]
        if compute_mask:
            mask_adv = np.logical_and(mask_adv, np.isfinite(output_fields[i, :]))

    # Field at t0 is not transformed
    output_fields[t0 - 1, :] = input_fields[t0 - 1, :]

    if not update_advfield:
        # To extrapolate backwards, we need to invert motion field
        advfield *= -1

    for i in range(t0, n_fields):
        # Extrapolate backwards
        if update_advfield:
            # Pick correct window the same as before, but now we're
            # extrapolating backwards in time so we need to flip the
            # time axis
            time_slice = slice(i - advfield_length + 1, i + 1)
            date_window = dates[time_slice]
            advfields[tuple(date_window)] = calc_oflow(input_fields[time_slice, ...])
            advfields[tuple(date_window[::-1])] = calc_oflow(
                np.flip(input_fields[time_slice, ...], axis=0)
            )
            # Since here t > t0, we need advection field from t ... t-n
            advfield = advfields[tuple(date_window[::-1])]

        output_fields[i, :] = semilagrangian.extrapolate(
            input_fields[i, :],
            advfield,
            abs(t0 - i - 1),
            **extrap_kwargs,
        )[-1]
        if compute_mask:
            mask_adv = np.logical_and(mask_adv, np.isfinite(output_fields[i, :]))

    if update_advfield:
        # Discard first input fields
        output_fields = output_fields[start_timestep:, ...]

    return output_fields, mask_adv, advfields


def plot_lagrangian_fields(
    dates, t0, orig_fields, lagr_fields, euler_fields, advfields, mask, outpath, min_dbz
):
    """Plot Lagrangian precipitation fields.

    Parameters
    ----------
    dates : list-line
        Dates corresponding to the precipitation fields.
    orig_fields : np.ndarray
        Original precipitation fields.
    lagr_fields : np.array
        Lagrangian precipitation fields.
    advfield : np.ndarray
        Motion field as returned by pysteps.motion.lucaskanade.dense_lucaskanade.
    mask : np.ndarray
        Advection mask.
    outpath : str
        Path to directory where output figures are saved.
    min_dbz : float
        Minimum dBZ value.

    """
    if orig_fields.shape[0] != lagr_fields.shape[0]:
        diff = orig_fields.shape[0] - lagr_fields.shape[0]
        orig_fields = orig_fields[diff:, ...]

    ncols = orig_fields.shape[0]
    nrows = 3

    lagr_fields[:, ~mask] = np.nan

    vmin = 0
    vmax = 20

    # lagr_fields[lagr_fields < min_dbz] = np.nan
    # orig_fields[orig_fields < min_dbz] = np.nan
    # euler_fields[euler_fields < min_dbz] = np.nan

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(2 * ncols, 8),
    )

    for i in range(ncols):

        # Original field
        axes[0, i].imshow(orig_fields[i, ...], vmin=vmin, vmax=vmax)
        # quiver(advfield, ax=axes[0, i])
        axes[0, i].set_title(f"{dates[i]:%Y-%m-%d %H:%M}", fontsize="xx-small")

        if i < t0 - 1:
            if len(advfields) > 1:
                field_k = [k for k in advfields.keys() if k[1] == dates[i]][0]
            else:
                field_k = list(advfields.keys())[0]
            advfield_l = advfields[field_k].copy()
            advfield_l[1, ...] *= -1
            axes[1, i].set_title(
                f"{field_k[0]:%H:%M} ... {field_k[1]:%H:%M}", fontsize="xx-small"
            )

            if len(advfields) > 1:
                field_k = [k for k in advfields.keys() if k[0] == dates[i]][0]
            else:
                field_k = list(advfields.keys())[0]

            advfield_e = advfields[field_k].copy()
            advfield_e[1, ...] *= -1
            axes[2, i].set_title(
                f"{field_k[0]:%H:%M} ... {field_k[1]:%H:%M}", fontsize="xx-small"
            )
        elif i == t0 - 1:  # common time
            advfield_l = np.zeros_like(advfield_l)
            advfield_e = np.zeros_like(advfield_e)
        else:
            if len(advfields) > 1:
                field_k = [k for k in advfields.keys() if k[0] == dates[i]][0]
            else:
                field_k = list(advfields.keys())[0]
            advfield_l = advfields[field_k].copy()
            advfield_l[1, ...] *= -1
            axes[1, i].set_title(
                f"{field_k[0]:%H:%M} ... {field_k[1]:%H:%M}", fontsize="xx-small"
            )

            if len(advfields) > 1:
                field_k = [k for k in advfields.keys() if k[1] == dates[i]][0]
            else:
                field_k = list(advfields.keys())[0]
            advfield_e = advfields[field_k].copy()
            advfield_e[1, ...] *= -1
            axes[2, i].set_title(
                f"{field_k[0]:%H:%M} ... {field_k[1]:%H:%M}", fontsize="xx-small"
            )

        # Lagrangian field
        axes[1, i].imshow(lagr_fields[i, ...], vmin=vmin, vmax=vmax)
        quiver(advfield_l, ax=axes[1, i])

        # Eulerian field
        axes[2, i].imshow(euler_fields[i, ...], vmin=vmin, vmax=vmax)
        quiver(advfield_e, ax=axes[2, i])

    for ax in axes.flat:
        ax.grid(zorder=100)

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        outpath / f"lagrangian_{dates[0]:%Y%m%d%H%M}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def read_advection_fields_from_h5(filename):
    """Read advection fields from HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the file.
    """

    advfields = {}
    with h5py.File(filename, "r") as f:
        for k, ds in f["advection_fields"].items():
            scale = ds["what"].attrs["gain"]
            offset = ds["what"].attrs["offset"]
            nodata = ds["what"].attrs["nodata"]
            undetect = ds["what"].attrs["undetect"]

            startdate = datetime.strptime(
                ds["what"].attrs["startdate"].decode("UTF-8"), "%Y%m%d%H%M"
            )
            enddate = datetime.strptime(
                ds["what"].attrs["enddate"].decode("UTF-8"), "%Y%m%d%H%M"
            )

            data = scale * ds["data"][:].astype(float) + offset
            data[ds["data"][:] == undetect] = 0
            data[ds["data"][:] == nodata] = 0

            advfields[(startdate, enddate)] = data
    return advfields


def read_advection_fields_from_nc(filename):
    """Read advection fields from NetCDF file.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    dict
        Dictionary of advection fields, with keys being the start and end times

    """
    advfields = {}
    ds = xr.open_dataset(filename)
    for endtime, ds_ in ds.groupby("endtime"):
        endtime = pd.Timestamp(endtime).to_pydatetime()
        for starttime, ds__ in ds_.groupby("starttime"):
            starttime = pd.Timestamp(starttime).to_pydatetime()
            motion_field = np.stack([ds__.U, ds__.V])
            advfields[(starttime, endtime)] = motion_field
    return advfields

def save_lagrangian_fields_h5_with_advfields(R, dates, t0, advfields, outputconf):
    """Save Lagrangian precipitation fields into HDF5 files.

    Parameters
    ----------
    R : np.ndarray
        Lagrangian precipitation fields.
    dates : list-like
        Dates corresponding to the precipitation fields.
    t0 : int
        Number of the date that fields are interpolated to (1-based; same as for
        `transform_to_lagrangian`.
    outputconf : dict
        Needs to contain 'path' and 'filename' fields, where 'filename' should have
        placeholder for 'commontime' for the common time of the Lagragian transform.
        'path'

    """
    N_BITS = outputconf["n_bits"]
    min_dbz = outputconf["min_val_dBZ"]
    max_dbz = outputconf["max_val_dBZ"]
    min_advfield = outputconf["min_val_advfield"]
    max_advfield = outputconf["max_val_advfield"]
    n_fields = R.shape[0]

    # Set negative values to zero
    R[R < min_dbz] = min_dbz
    R[R > max_dbz] = max_dbz

    # The date that fields are extrapolated to
    enddate = dates[t0 - 1]

    fn = (
        Path(
            outputconf["path"].format(
                year=enddate.year,
                month=enddate.month,
                day=enddate.day,
            )
        )
        / Path(outputconf["filename"].format(commontime=enddate))
    )
    fn.parents[0].mkdir(parents=True, exist_ok=True)
    with h5py.File(fn, "w") as f:
        # Write precipitation fields
        for i in range(n_fields):
            dname = f"{dates[i]:%Y%m%d%H%M}"
            ds = f.require_group(dname)

            # Calculate scale and offset
            offset = min_dbz
            scale = (max_dbz - min_dbz) / (2 ** N_BITS - 2)

            packed = np.round((R[i, ...] - offset) / scale)
            packed[np.isnan(packed)] = 2 ** N_BITS - 1

            # Write dataset and attributes
            data = ds.create_dataset(
                "data",
                data=packed,
                dtype=f"uint{N_BITS}",
                # **hdf5plugin.Blosc(
                #     cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
                # ),
                compression="gzip",
                compression_opts=9,
            )
            data.attrs["CLASS"] = np.string_("IMAGE")
            data.attrs["IMAGE_VERSION"] = np.string_("1.2")

            ds_what = ds.require_group("what")
            ds_what.attrs["gain"] = scale
            ds_what.attrs["offset"] = offset
            ds_what.attrs["undetect"] = 0.0
            ds_what.attrs["nodata"] = 2 ** N_BITS - 1
            ds_what.attrs["date"] = np.string_(f"{dates[i]:%Y%m%d%H%M}")

        # Write advection fields
        ds_top = f.require_group("advection_fields")
        for (startdate, enddate), advfield in advfields.items():
            dname = f"{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}"
            ds = ds_top.require_group(dname)

            # Calculate scale and offset
            if np.nanmin(advfield) < min_advfield or np.nanmax(advfield) > max_advfield:
                offset = np.nanmin(advfield)
                scale = (np.nanmax(advfield) - offset) / (2 ** N_BITS - 2)
            else:
                offset = min_advfield
                scale = (max_advfield - min_advfield) / (2 ** N_BITS - 2)

            packed = np.round((advfield - offset) / scale)
            packed[np.isnan(packed)] = 2 ** N_BITS - 1

            # Write dataset and attributes
            data = ds.create_dataset(
                "data",
                data=packed,
                dtype=f"uint{N_BITS}",
                # **hdf5plugin.Blosc(
                #     cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
                # ),
                compression="gzip",
                compression_opts=9,
            )
            data.attrs["CLASS"] = np.string_("IMAGE")
            data.attrs["IMAGE_VERSION"] = np.string_("1.2")

            ds_what = ds.require_group("what")
            ds_what.attrs["gain"] = scale
            ds_what.attrs["offset"] = offset
            ds_what.attrs["undetect"] = 0.0
            ds_what.attrs["nodata"] = 2 ** N_BITS - 1
            ds_what.attrs["startdate"] = np.string_(f"{startdate:%Y%m%d%H%M}")
            ds_what.attrs["enddate"] = np.string_(f"{enddate:%Y%m%d%H%M}")
