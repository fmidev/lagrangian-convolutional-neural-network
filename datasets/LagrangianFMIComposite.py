"""PyTorch Dataset and LightningDataModule definitions for FMI data."""
import gzip
from pathlib import Path
from datetime import timedelta, datetime
import logging

import h5py
import numpy as np
import pandas as pd
from skimage.measure import block_reduce
import torch
from matplotlib.pyplot import imread
from torch.utils.data import Dataset


class LagrangianFMIComposite(Dataset):
    """Dataset for FMI composite in PGM files."""

    def __init__(
        self,
        split="train",
        db=None,
        date_list=None,
        path=None,
        filename=None,
        importer=None,
        input_block_length=None,
        prediction_block_length=None,
        len_date_block=None,
        timestep=None,
        bbox=None,
        image_size=None,
        bbox_image_size=None,
        input_image_size=None,
        upsampling_method=None,
        normalization_method="log",
        transform_to_grayscale=True,
        apply_differencing=False,
        predicting=False,
        log_unit_diff_cutoff=None,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        split : {'train', 'test', 'valid'}
            The type of the dataset: training, testing or validation.
        db :
            H5py File object to access hdf5 file containing data.
        date_list : str
            Defines the name format of the date list file. The string is expected
            to contain the '{split}' keyword, where the value of the 'split'
            argument is substituted.
        path : str
            Format of the data path. May contain the tokens {year:*}, {month:*},
            {day:*}, {hour:*}, {minute:*}, {second:*} that are substituted when
            going through the dates.
        filename : str
            Format of the data file names. May contain the tokens {year:*},
            {month:*}, {day:*}, {hour:*}, {minute:*}, {second:*} that are
            substituted when going through the dates.
        importer : {'pgm_gzip', 'hdf5'}
            The importer to use for reading the files.
        input_block_length : int
            The number of frames to be used as input to the models.
        prediction_block_length : int
            The number of frames that are predicted and tested against the
            observations.
        timestep : int
            Time step of the data (minutes).
        bbox : str
            Bounding box of the data in the format '[x1, x2, y1, x2]'.
        image_size : str
            Shape of the original images without bounding box in the format
            '[width, height]'.
        image_size : str
            Shape of the images after the bounding box in the format
            '[width, height]'.
        bbox_image_size : str
            Size of the image after clipping to bbox.
        input_image_size : str
            Shape of the input images supplied to the models in the format
            '[width, height]' after upsampling.
        upsampling_method : {'average'}
            The method to use for upsampling the input images.
        normalization_method : str
            Normalization method used to to transform data.
        transform_to_grayscale : bool
            Whether to transform data to grayscale.
        apply_differencing : bool
            Whether to apply differencing to the data.

        """
        assert date_list is not None, "No date list for FMI composites given!"
        assert path is not None, "No path for FMI composites given!"
        assert filename is not None, "No filename format for FMI composites given!"
        assert importer is not None, "No importer for FMI composites given!"

        # Inherit from parent class
        super().__init__()

        # Get data times
        self.date_list = pd.read_csv(
            date_list.format(split=split), header=None, parse_dates=[0]
        )
        self.date_list.set_index(self.date_list.columns[0], inplace=True)

        self.path = path
        self.filename = filename

        # Get correct importer function
        if importer == "pgm_gzip":
            self.importer = read_pgm_composite
        elif importer == "hdf5":
            self.importer = read_hdf5_db
            assert db is not None, "No HDF5 database file given!"
            self.db = db
        elif importer == "lagrangian_h5":
            self.importer = read_lagrangian_h5
        else:
            raise NotImplementedError(f"Importer {importer} not implemented!")

        self.upsampling_method = upsampling_method

        self.image_size = image_size
        self.bbox_image_size = bbox_image_size
        self.input_image_size = input_image_size

        if normalization_method not in ["log", "log_unit", "none", "log_unit_diff"]:
            raise NotImplementedError(
                f"data normalization method {normalization_method} not implemented"
            )
        else:
            self.normalization = normalization_method

        if bbox is None or self.image_size == self.bbox_image_size:
            self.use_bbox = False
        else:
            self.use_bbox = True
            self.bbox_x_slice = slice(bbox[0], bbox[1])
            self.bbox_y_slice = slice(bbox[2], bbox[3])

        self.num_frames_input = input_block_length
        self.num_frames_output = prediction_block_length
        self.num_frames = input_block_length + prediction_block_length
        self.len_date_block = len_date_block
        self.common_time_index = self.num_frames_input - 1

        self.transform_to_grayscale = transform_to_grayscale
        self.apply_differencing = apply_differencing
        self.log_unit_diff_cutoff = log_unit_diff_cutoff

        # Get windows
        self.timestep = timestep
        self.date_list_pdt = self.date_list.index.to_pydatetime()
        self.windows = self.make_windows()

        # If we're predicting now
        self.predicting = predicting

    def __len__(self):
        """Mandatory property for Dataset."""
        return self.windows.shape[0]

    def __getitem__(self, idx):
        """Mandatory property for fetching data."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        window = self.windows.iloc[idx].dt.to_pydatetime()
        common_time = window[self.common_time_index]
        data = np.empty((self.num_frames, *self.input_image_size))

        # Check that window has correct length
        if (window[-1] - window[0]).seconds / (self.timestep * 60) != (
            self.num_frames - 1
        ):
            logging.info(f"Window {window[0]} - {window[-1]} wrong!")

        fn = self.get_filename(common_time)
        try:
            images = self.importer(fn)
        except FileNotFoundError:
            return None

        for i, date in enumerate(window):
            im = images[date]

            if self.use_bbox:
                im = self.apply_bbox(im)

            if (
                im.shape[0] != self.input_image_size[0]
                or im.shape[1] != self.input_image_size[1]
            ):
                im = self.downsample_data(im)

            data[i, ...] = im

        inputs, outputs, first_field = self.postprocessing(data)

        if self.apply_differencing and self.predicting:
            # We want to return the first original field to allow transformation back to full fields
            return inputs, outputs, idx, first_field

        return inputs, outputs, idx

    def apply_bbox(self, im):
        return im[..., self.bbox_x_slice, self.bbox_y_slice]

    def downsample_data(self, im):
        # Upsample image
        # Calculate window size
        block_x = int(im.shape[0] / self.input_image_size[0])
        block_y = int(im.shape[1] / self.input_image_size[1])
        if self.upsampling_method == "average":
            # Upsample by averaging
            im = block_reduce(
                im, func=np.nanmean, cval=0, block_size=(block_x, block_y)
            )
        else:
            raise NotImplementedError(
                f"Upsampling {self.upsampling_method} not yet implemented!"
            )
        return im

    def get_filename(self, common_time):
        return Path(
            self.path.format(
                year=common_time.year,
                month=common_time.month,
                day=common_time.day,
                hour=common_time.hour,
                minute=common_time.minute,
                second=common_time.second,
            )
        ) / Path(
            self.filename.format(
                commontime=common_time,
            )
        )

    def get_window(self, index):
        """Utility function to get window."""
        if isinstance(index, int):
            return self.windows.iloc[index].dt.to_pydatetime()
        elif index.numel() == 1:
            return self.windows.iloc[index.item()].dt.to_pydatetime()
        else:
            return np.stack(
                [
                    self.windows.iloc[index].iloc[i].dt.to_pydatetime()
                    for i in range(len(index))
                ]
            )

    def get_common_time(self, index):
        window = self.get_window(index)
        return window[self.common_time_index]

    def scaler(self, data: torch.Tensor):
        if self.normalization == "log_unit_diff":
            data[data > self.log_unit_diff_cutoff] = self.log_unit_diff_cutoff
            data[data < -self.log_unit_diff_cutoff] = -self.log_unit_diff_cutoff
            return (data / self.log_unit_diff_cutoff + 1) / 2
        if self.normalization == "log_unit":
            return (torch.log(data + 0.01) + 5) / 10
        if self.normalization == "log":
            return torch.log(data + 0.01)
        if self.normalization == "none":
            return data

    def invScaler(self, data: torch.Tensor):
        if self.normalization == "log_unit_diff":
            return (data * 2 - 1) * self.log_unit_diff_cutoff
        if self.normalization == "log_unit":
            return torch.exp((data * 10) - 5) - 0.01
        if self.normalization == "log":
            return torch.exp(data) - 0.01
        if self.normalization == "none":
            return data

    def postprocessing(self, data_in: np.ndarray):
        data = torch.Tensor(data_in)
        if self.transform_to_grayscale:
            # data of shape (window_size, im.shape[0], im.shape[1])
            # dbZ to mm/h
            data = 10 ** (data * 0.1)
            data = (data / 223) ** (1 / 1.53)  # fixed

        first_field = None
        if self.apply_differencing:
            # Difference data
            first_field = data[0, ...].clone()
            data = torch.diff(data, dim=0)

        if self.transform_to_grayscale:
            # mm / h to log-transformed
            data = self.scaler(data)
            first_field = self.scaler(first_field)

        # Divide to input & output
        # Use output frame number, since that is constant whether we apply differencing or not
        inputs = data[: -self.num_frames_output, ...].permute(0, 1, 2).contiguous()
        outputs = data[-self.num_frames_output :, ...].permute(0, 1, 2).contiguous()

        return inputs, outputs, first_field

    def from_transformed(self, data, scaled=True):
        if scaled:
            data = self.invScaler(data)  # to mm/h
        data = 223 * data ** (1.53)  # to z
        data = 10 * torch.log10(data)  # to dBZ

        return data

    def make_windows(self):
        # Get windows
        num_blocks = int(len(self.date_list) / self.len_date_block)
        blocks = np.array_split(self.date_list.index.to_pydatetime(), num_blocks)
        windows = pd.DataFrame(
            np.concatenate(
                [
                    np.lib.stride_tricks.sliding_window_view(b, self.num_frames)
                    for b in blocks
                ]
            )
        )
        return windows


def read_pgm_composite(filename, no_data_value=-32):
    """Read uint8 PGM composite, convert to dBZ."""
    data = imread(gzip.open(filename, "r"))
    mask = data == 255
    data = data.astype(np.float64)
    data = (data - 64.0) / 2.0
    data[mask] = no_data_value

    return data


def read_hdf5_db(filename, db):
    """Read composite reflectivity (dBZ) from the hdf5 database"""
    data = np.array(db[filename])
    return data


def read_lagrangian_h5(filename):
    with h5py.File(filename, "r") as f:
        data = {}
        for name, dset in f.items():
            if len(name) == 12:
                d, arr = unpack_h5_dataset(name, dset)
                data[d] = arr
    return data


def unpack_h5_dataset(name, dset):
    scale = dset["what"].attrs["gain"]
    offset = dset["what"].attrs["offset"]
    undetect = dset["what"].attrs["undetect"]
    nodata = dset["what"].attrs["nodata"]

    data = scale * dset["data"][:].astype(float) + offset
    data[dset["data"][:] == undetect] = 0
    data[dset["data"][:] == nodata] = 0
    day = datetime.strptime(name, "%Y%m%d%H%M")
    return day, data
