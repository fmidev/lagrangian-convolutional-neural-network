import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import Any, List
import numpy as np
from datetime import timedelta
import h5py
from pathlib import Path


class LagrangianHDF5Writer(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        filename: str,
        group_format: str,
        filter_dbz: float = -32.0,
        what_attrs: dict = {},
        where_attrs: dict = {},
        how_attrs: dict = {},
        write_leadtimes_separately: bool = False,
        write_interval: str = "batch",
        **kwargs,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.filename = filename
        self.what_attrs = what_attrs
        self.how_attrs = how_attrs
        self.where_attrs = where_attrs
        self.group_format = group_format
        self.write_leadtimes_separately = write_leadtimes_separately
        self.filter_dbz = filter_dbz

    def write_on_batch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        _, _, batch_indices_, *_ = batch
        batch_indices_ = batch_indices_.tolist()

        for bi, b_idx in enumerate(batch_indices_):
            common_time = trainer.datamodule.predict_dataset.get_common_time(b_idx, in_datetime=True)
            if not self.write_leadtimes_separately:
                fn = (
                    Path(
                        self.output_dir.format(
                            year=common_time.year,
                            month=common_time.month,
                            day=common_time.day,
                        )
                    )
                    / Path(self.filename.format(common_time=common_time))
                )
                fn.parents[0].mkdir(parents=True, exist_ok=True)
                with h5py.File(fn, "a") as f:
                    where = f.require_group("where")
                    write_attrs(where, self.where_attrs)

                    how = f.require_group("how")
                    write_attrs(how, self.how_attrs)

                    # Write base what attributes
                    what = f.require_group("what")
                    base_what = {
                        "date": np.string_(f"{common_time:%Y%m%d}"),
                        "time": np.string_(f"{common_time:%H%M%S}"),
                        "object": np.string_("COMP"),
                        "source": np.string_("NOD:fi;ORG:fi"),
                    }
                    write_attrs(what, base_what)

                    group_name = self.group_format.format(common_time=common_time)
                    group = f.require_group(group_name)

                    for i in range(prediction.shape[1]):
                        date = common_time + timedelta(
                            minutes=(i + 1)
                            * trainer.datamodule.predict_dataset.timestep
                        )
                        dname = f"{i + 1}"
                        ds_group = group.require_group(dname)

                        what_attrs = self.what_attrs.copy()
                        what_attrs["validtime"] = np.string_(
                            f"{date:%Y-%m-%d %H:%M:%S}"
                        )

                        packed = arr_compress_uint8(
                            prediction[bi, i, ...].detach().cpu().numpy(),
                            nodata_val=what_attrs["nodata"],
                            undetect_val=what_attrs["undetect"],
                            filter_dbz=self.filter_dbz,
                        )
                        write_image(
                            group=ds_group,
                            ds_name="data",
                            data=np.flipud(packed),
                            what_attrs=what_attrs,
                        )
            else:
                for i in range(prediction.shape[1]):
                    leadtime = (i + 1) * trainer.datamodule.predict_dataset.timestep
                    date = common_time + timedelta(minutes=leadtime)
                    fn = Path(
                        self.output_dir.format(
                            year=common_time.year,
                            month=common_time.month,
                            day=common_time.day,
                        )
                    ) / Path(
                        self.filename.format(
                            common_time=common_time, validtime=date, leadtime=leadtime
                        )
                    )
                    fn.parents[0].mkdir(parents=True, exist_ok=True)
                    with h5py.File(fn, "a") as f:
                        where = f.require_group("where")
                        write_attrs(where, self.where_attrs)

                        how = f.require_group("how")
                        write_attrs(how, self.how_attrs)

                        # Write base what attributes
                        what = f.require_group("what")
                        base_what = {
                            "date": np.string_(f"{common_time:%Y%m%d}"),
                            "time": np.string_(f"{common_time:%H%M%S}"),
                            "object": np.string_("COMP"),
                            "source": np.string_("NOD:fi;ORG:fi"),
                        }
                        write_attrs(what, base_what)

                        group_name = self.group_format.format(common_time=common_time)
                        group = f.require_group(group_name)

                        what_attrs = self.what_attrs.copy()
                        what_attrs["validtime"] = np.string_(
                            f"{date:%Y-%m-%d %H:%M:%S}"
                        )

                        packed = arr_compress_uint8(
                            prediction[bi, i, ...].detach().cpu().numpy(),
                            nodata_val=what_attrs["nodata"],
                            undetect_val=what_attrs["undetect"],
                            filter_dbz=self.filter_dbz,
                        )
                        write_image(
                            group=group,
                            ds_name="data",
                            data=packed,
                            what_attrs=what_attrs,
                        )

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        pass


def arr_compress_uint8(
    dBZ_array: np.ndarray,
    nodata_val: np.uint8 = 255,
    undetect_val: np.uint8 = 0,
    filter_dbz: float = -32.0,
) -> np.ndarray:
    masked = np.ma.masked_where(~np.isfinite(dBZ_array), dBZ_array)
    masked = np.ma.masked_less(masked, filter_dbz)
    max_value_dBZ = -32 + 0.5 * 254  # not 255 to ensure no real value gets lost!
    mask_big_values = dBZ_array[...] >= max_value_dBZ
    arr = ((2.0 * masked) + 64).astype(np.uint8)
    arr[arr.mask] = nodata_val
    arr[mask_big_values] = 254
    arr[masked < 0] = undetect_val
    return arr.data


def write_image(group: h5py.Group, ds_name: str, data: np.ndarray, what_attrs: dict):
    try:
        del group[ds_name]
    except:
        pass
    dataset = group.create_dataset(
        ds_name, data=data, dtype="uint8", compression="gzip", compression_opts=9
    )
    dataset.attrs["CLASS"] = np.string_("IMAGE")
    dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")

    ds_what = group.require_group("what")
    for k, val in what_attrs.items():
        ds_what.attrs[k] = val


def write_attrs(group: h5py.Group, attrs: dict):
    for k, val in attrs.items():
        group.attrs[k] = val
