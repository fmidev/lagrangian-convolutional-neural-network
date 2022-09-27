"""
Helper functions to read and write predictions, and
otherwise process the IO required for the verification of models.
"""

from datetime import datetime, timedelta
import numpy as np
import h5py


def arr_compress_uint8(
    dBZ_array: np.ndarray, missing_val: np.uint8 = 255
) -> np.ndarray:
    masked = np.ma.masked_where(~np.isfinite(dBZ_array), dBZ_array)
    max_value_dBZ = -32 + 0.5 * 254  # not 255 to ensure no real value gets lost!
    mask_big_values = dBZ_array[...] >= max_value_dBZ
    arr = ((2.0 * masked) + 64).astype(np.uint8)
    arr[arr.mask] = missing_val
    arr[mask_big_values] = 254
    return arr.data


def arr_reconstruct_uint8(
    uint8_array: np.ndarray, missing_val: np.uint8 = 255, mask_val: float = np.nan
):
    mask = uint8_array == missing_val
    arr = uint8_array.astype(np.float64)
    arr[mask] = mask_val
    arr = (arr - 64) / 2.0
    return arr


def write_image(group: h5py.Group, ds_name: str, data: np.ndarray, what_attrs: dict):
    dataset = group.create_dataset(
        ds_name, data=data, dtype="uint8", compression="gzip", compression_opts=9
    )
    dataset.attrs["CLASS"] = np.string_("IMAGE")
    dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")

    ds_what = group.require_group("what")
    ds_what.attrs["quantity"] = what_attrs["quantity"]
    ds_what.attrs["gain"] = what_attrs["gain"]
    ds_what.attrs["offset"] = what_attrs["offset"]
    ds_what.attrs["undetect"] = what_attrs["undetect"]
    ds_what.attrs["nodata"] = what_attrs["nodata"]


def read_image(group: h5py.Group) -> np.ndarray:
    img_uint8 = np.array(group["data"]).squeeze()
    img = arr_reconstruct_uint8(img_uint8)
    return img


def get_neighbor(time, distance: int, delta_min: int = 5):
    if isinstance(time, str):
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        neigh_time = time + timedelta(minutes=(delta_min * distance))
        return neigh_time.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(time, datetime):
        return time + timedelta(minutes=(delta_min * distance))

    else:
        raise ValueError("Time is not in a valid format (str/datetime)")


def _get_sample_names(sample0: str, leadtimes: list) -> list:
    #  sample0 corresponds to the timestamp of the first sample in the series of
    # samples used for calculating predictions.
    # offset : 4 prediction sample by default, corresponds to the number of samples loaded for pred.
    # time0 = datetime.strptime(sample0, '%Y-%m-%d %H:%M:%S')
    names = []
    for lt in leadtimes:
        time_lt = get_neighbor(sample0, distance=lt)
        names.append(time_lt)
    return names


def load_observations(db_path: str, sample: str, leadtimes: list) -> tuple:
    obs_path_fmt = "{sample}/measurements/"
    observations = []
    timestamps = _get_sample_names(sample, leadtimes)
    with h5py.File(db_path, "r") as db:
        for timestamp in timestamps:
            try:
                obs_grp = db[obs_path_fmt.format(sample=timestamp)]
                observations.append(read_image(obs_grp))
            except:
                return None
    return np.array(observations)


def load_predictions(prediction_methods: dict, sample: str, leadtimes: list) -> tuple:
    pred_path_fmt = "{sample}/{method}/{leadtime}/"
    pred_dict = dict()
    for method in prediction_methods:
        with h5py.File(prediction_methods[method]["path"], "r") as db:
            pred_dict[method] = []
            for lt in leadtimes:
                try:
                    pred_path = pred_path_fmt.format(
                        sample=sample, method=method, leadtime=lt
                    )
                    pred_grp = db[pred_path]
                    pred_dict[method].append(read_image(pred_grp))
                except:
                    return method
    return {method: np.array(pred_dict) for method, pred_dict in pred_dict.items()}


def dBZ_to_rainrate(
    data_dBZ: np.ndarray, zr_a: float = 223, zr_b: float = 1.53
) -> np.ndarray:
    data = 10 ** (data_dBZ * 0.1)  # dB - inverse transform -> Z
    data = (data / zr_a) ** (1 / zr_b)  # Z -> R
    return data


def rainrate_to_dBZ(
    R: np.ndarray,
    zr_a: float = 223,
    zr_b: float = 1.53,
    thresh: float = 0.1,
    zerovalue=-32,
) -> np.ndarray:
    zeros = R < thresh
    Z = zr_a * R ** zr_b  # R -> Z
    Z = 10 * np.log10(Z)  # Z -> dBZ
    Z[zeros] = zerovalue  # fill values under threshold with zerovalues
    return Z


def dBZ_list_to_rainrate(data: list, zr_a: float = 223, zr_b: float = 1.53):
    out = []
    for i in data:
        out.append(dBZ_to_rainrate(i, zr_a, zr_b))
    return out


def read_file(rainy_days_path, start_idx: int = 0):
    with open(rainy_days_path, "r") as f:
        rain_days = f.readlines()
        rain_days = [r.rstrip() for r in rain_days]
        rain_days = rain_days[start_idx:]
    return rain_days


def chunk_list(list, n_chunks):
    for i in range(0, len(list), n_chunks):
        yield list[i : i + n_chunks]
