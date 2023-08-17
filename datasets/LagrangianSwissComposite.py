"""Pytorch Lightning DataModule for Netcdf datasets."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
from skimage.measure import block_reduce
import xarray as xr

# from .transformation import NowcastingTransformation


def normalize_mean_std(arr, mean, std, nanfill=0.0):
    arr = (arr - mean) / std
    arr[arr.isnan()] = nanfill
    return arr


def denormalize_mean_std(arr, mean, std):
    return arr * std + mean


class LagrangianSwissCompositeDataset(Dataset):
    """Dataset for netcdf-stored data."""

    def __init__(
        self,
        split="train",
        datapath: str = None,
        filename_format: str = None,
        split_list_path: str = None,
        image_size: "tuple[int]" = [64, 64],
        input_image_size: "tuple[int]" = [640, 710],
        bbox: "tuple[int]" = [0, 64, 0, 64],
        bbox_image_size: "tuple[int]" = [64, 64],
        input_block_length: int = None,
        prediction_block_length: int = None,
        len_date_block: int = None,
        common_time_index: int = None,
        normalization: dict = None,
        predicting=False,
        input_features: list = None,
        output_features: list = None,
        apply_differencing: bool = True,
        timestep: int = 5,
        advection_file_path: str = None,
        upsampling_method: str = "average",
        **kwargs,
    ):
        self.split = split
        self.predicting = predicting
        self.datapath = Path(datapath)
        self.advection_file_path = advection_file_path

        # Read variable names
        with open(self.datapath.parents[0] / "feature_names.txt", "r") as f:
            feature_names = f.readlines()
            feature_names = [x.strip() for x in feature_names]
        self.feature_names = feature_names
        self.drop_features = []
        self.num_features = len(feature_names)

        self.filename_format = filename_format

        if input_features is None or input_features == "all":
            self.select_features = np.ones(self.num_features, dtype=bool)
        else:
            self.select_features = np.zeros(self.num_features, dtype=bool)
            for feature in input_features:
                self.select_features[feature_names.index(feature)] = True
            self.num_features = len(input_features)
            self.feature_names = [n for n in input_features if n in feature_names]
            self.drop_features = [n for n in feature_names if n not in input_features]

        if output_features is None or output_features == "all":
            self.select_output_features = np.ones(self.num_features, dtype=bool)
        else:
            self.select_output_features = np.zeros(self.num_features, dtype=bool)
            for feature in output_features:
                self.select_output_features[self.feature_names.index(feature)] = True
            self.num_output_features = len(output_features)
            self.output_feature_names = [
                n for n in feature_names if n in output_features
            ]

        # Read normalization parameters
        if normalization is not None:
            self.normalize = True
            self.normalization_methods = {}
            self.denormalization_methods = {}
            for i, feature in enumerate(self.feature_names):
                self.normalization_methods[feature] = partial(
                    normalize_mean_std, **normalization[feature]
                )
                self.denormalization_methods[feature] = partial(
                    denormalize_mean_std, **normalization[feature]
                )
            # Get indices of features to normalize after selecting features
            # indices = np.cumsum(self.select_features) - 1
            self.normalization_methods_by_dataindex = {
                i: name for i, name in enumerate(self.feature_names)
            }

        else:
            self.normalize = False

        self.image_size = image_size
        self.input_image_size = input_image_size
        self.bbox_image_size = bbox_image_size
        self.upsampling_method = upsampling_method

        if bbox is None or self.image_size == self.bbox_image_size:
            self.use_bbox = False
        else:
            self.use_bbox = True
            self.bbox_x_slice = slice(bbox[0], bbox[1])
            self.bbox_y_slice = slice(bbox[2], bbox[3])

        # Block lengths
        self.input_block_length = input_block_length
        self.prediction_block_length = prediction_block_length
        self.len_date_block = len_date_block
        self.common_time_index_original = common_time_index
        self.common_time_index = input_block_length - 1
        self.num_frames = self.input_block_length + self.prediction_block_length

        # Get data times
        self.date_list = pd.read_csv(
            self.datapath.parents[0] / "filelist.csv",
            header=0,
            parse_dates=[0],
            names=["date", "path"],
        )
        self.date_list.set_index(self.date_list.date, inplace=True)
        self.split_list = pd.read_csv(
            split_list_path.format(split=split), header=None, parse_dates=[0]
        )
        self.split_list.set_index(self.split_list.columns[0], inplace=True)
        # self.bad_times = pd.read_csv(
        #     self.datapath.parents[0] / "bad_timesteps.txt",
        #     header=None,
        #     usecols=[0],
        #     parse_dates=[0],
        #     names=["date"],
        # )

        self.make_windows()

        # Settings for differencing
        self.apply_differencing = apply_differencing
        self.difference_indices = np.array(
            [
                True if "RATE" in var else False
                for i, var in enumerate(self.feature_names)
            ]
        )
        self.pick_num_inputs = self.input_block_length
        if self.apply_differencing:
            self.pick_num_inputs += 1

        self.timestep = timestep
        self.dataset = None

    def __len__(self):
        return len(self.date_index)

    def apply_bbox(self, im):
        """Apply bounding box to image."""
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

    def apply_normalization(self, data, feature_axis=-3, target=False):
        """Apply normalization to data."""
        if self.normalize:
            if target:
                # Normalize target features
                for i, name in enumerate(self.output_feature_names):
                    func = self.normalization_methods[name]
                    if feature_axis == -3:
                        data[..., i, :, :] = func(data[..., i, :, :])
                    elif feature_axis == -4:
                        data[..., i, :, :, :] = func(data[..., i, :, :, :])
                    else:
                        raise ValueError("Feature axis must be -3 or -4.")
            else:
                # Normalize input features
                for data_index, name in self.normalization_methods_by_dataindex.items():
                    func = self.normalization_methods[name]
                    if feature_axis == -3:
                        data[..., data_index, :, :] = func(data[..., data_index, :, :])
                    elif feature_axis == -4:
                        data[..., data_index, :, :, :] = func(
                            data[..., data_index, :, :, :]
                        )
                    else:
                        raise ValueError("Feature axis must be -3 or -4.")
        return data

    def apply_denormalization(self, data, feature_axis=-3, target=False):
        """De-normalize data."""
        if self.normalize:
            if target:
                # Normalize target features
                for i, name in enumerate(self.output_feature_names):
                    func = self.normalization_methods[name]
                    if feature_axis == -3:
                        data[..., i, :, :] = func(data[..., i, :, :])
                    elif feature_axis == -4:
                        data[..., i, :, :, :] = func(data[..., i, :, :, :])
                    else:
                        raise ValueError("Feature axis must be -3 or -4.")
            else:
                # Normalize input features
                for data_index, name in self.normalization_methods_by_dataindex.items():
                    func = self.denormalization_methods[name]
                    if feature_axis == -3:
                        data[..., data_index, :, :] = func(data[..., data_index, :, :])
                    elif feature_axis == -4:
                        data[..., data_index, :, :, :] = func(
                            data[..., data_index, :, :, :]
                        )
                    else:
                        raise ValueError("Feature axis must be -3 or -4.")
        return data

    def make_windows(self):
        # Get windows
        num_blocks = int(len(self.split_list) / self.len_date_block)
        blocks = np.array_split(self.split_list.index.to_pydatetime(), num_blocks)
        self.windows = pd.DataFrame(
            np.concatenate(
                [
                    np.lib.stride_tricks.sliding_window_view(b, self.num_frames)
                    for b in blocks
                ]
            )
        )
        # If the number of input/output frames is different than
        # when creating the dataset,
        # we can have windows that do not exist in the dataset
        # So we remove those
        self.windows = self.windows.loc[
            self.windows[self.common_time_index].isin(self.date_list.index)
        ]
        self.date_index = self.date_list.loc[
            self.windows[self.common_time_index].values
        ]

    def get_window(self, idx, in_datetime=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        window = self.windows.loc[
            self.windows[self.common_time_index].values
            == self.date_index.iloc[idx].date
        ]
        if in_datetime:
            return pd.to_datetime(window.values.squeeze()).to_pydatetime().tolist()
        return window

    def get_common_time(self, index, in_datetime=False):
        window = self.get_window(index)
        common_time = window[self.common_time_index]
        if in_datetime:
            return common_time.item().to_pydatetime()
        return common_time

    def get_index_by_date(self, date):
        try:
            idx = np.argwhere(
                self.date_index.date.dt.to_pydatetime() == date
            ).flatten()[-1]
        except IndexError:
            idx = None
        return idx

    def postprocessing(self, data_in: np.ndarray):
        data = torch.from_numpy(data_in.astype(np.float32))

        first_field = torch.empty((1, *self.image_size))
        if self.apply_differencing:
            # Difference data
            data_out = torch.zeros((self.num_frames, *data.shape[1:]))
            first_field = data[0, self.difference_indices, ...].clone()
            data_out[:, self.difference_indices, ...] = torch.diff(
                data[:, self.difference_indices, ...], dim=0
            )
            data_out[:, ~self.difference_indices, ...] = data[
                1:, ~self.difference_indices, ...
            ]
        else:
            data_out = data

        if self.normalize:
            # Normalize data
            data_out = self.apply_normalization(data_out)

        # if self.transform_to_grayscale:
        #     # physical units (normalized) to log-transformed
        #     data = self.scaler(data)
        #     first_field = self.scaler(first_field)

        # Divide to input & output
        # Use output frame number, since that is constant whether we apply differencing or not
        # Change from (time, channels, x, y) to (channels, time, x, y)
        inputs = (
            data_out[: -self.prediction_block_length, ...]
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        outputs = (
            data_out[-self.prediction_block_length :, self.select_output_features, ...]
            .permute(1, 0, 2, 3)
            .contiguous()
        )

        return inputs, outputs, first_field

    def get_filename(self, idx):
        """Get filename for a given index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.date_index.iloc[idx].path

    def __getitem__(self, idx):
        """Mandatory property for fetching data."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.date_index.iloc[idx].path
        ds = xr.open_dataset(
            file,
            cache=True,
            drop_variables=self.drop_features,
        )

        # Pick slice surrounding common time index
        try:
            data_ = (
                ds[self.feature_names]
                .to_array()
                .transpose("time", "variable", "y", "x")[
                    (
                        self.common_time_index_original + 1 - (self.pick_num_inputs)
                    ) : self.common_time_index_original
                    + 1
                    + self.prediction_block_length,
                    :,
                    ...,
                ]
                .values
            )
        except KeyError as e:
            print(f"Error reading file {file}")
            return None

        # Apply bounding box
        if self.use_bbox:
            data_ = self.apply_bbox(data_)

        # Normalize data
        inputs, outputs, first_field = self.postprocessing(data_)

        # import ipdb; ipdb.set_trace()

        if self.predicting:
            return inputs.squeeze(), outputs.squeeze(), idx, first_field.squeeze()
        else:
            return inputs.squeeze(), outputs.squeeze(), idx

        # return {
        #     "inputs": inputs,
        #     "targets": outputs,
        #     "idx": idx,
        #     "first_field": first_field,
        # }


class LagrangianSwissCompositeDataModule(pl.LightningDataModule):
    """Lightning DataModule for zarr-stored data."""

    def __init__(
        self, train_dataconfig, valid_dataconfig, module_params, predict_list="predict"
    ):
        """Initialize the data module."""
        super().__init__()
        self.train_dsconfig = train_dataconfig
        self.valid_dsconfig = valid_dataconfig
        self.module_params = module_params
        self.predict_list = predict_list

        # if self.module_params.transformations is None:
        #     self.train_transform = None
        # else:
        #     self.train_transform = NowcastingTransformation(
        #         self.module_params.transformations
        #     )

    def prepare_data(self):
        """Prepare data (dummy)."""
        # called only on 1 GPU
        pass

    def setup(self, stage):
        """Set up the dataset."""
        # called on every GPU
        # if stage == "train":
        self.train_dataset = LagrangianSwissCompositeDataset(split="train", **self.train_dsconfig)
         # if stage == "val":
        self.valid_dataset = LagrangianSwissCompositeDataset(split="valid", **self.valid_dsconfig)
        # if stage == "test":
        self.test_dataset = LagrangianSwissCompositeDataset(split="test", **self.valid_dsconfig)
        # if stage == "predict":
        # Predicting
        self.predict_dataset = LagrangianSwissCompositeDataset(
            split=self.predict_list, predicting=True, **self.valid_dsconfig
            )

    def train_dataloader(self):
        """Return the training dataloader."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.module_params.train_batch_size,
            num_workers=self.module_params.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        """Return the validation dataloader."""
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.module_params.valid_batch_size,
            num_workers=self.module_params.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        """Return the test dataloader."""
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.module_params.test_batch_size,
            num_workers=self.module_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return test_loader

    def predict_dataloader(self):
        """Return the prediction dataloader."""
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.module_params.predict_batch_size,
            num_workers=self.module_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     "Apply training data transformation"
    #     if self.trainer and self.trainer.state != "train":
    #         return batch
    #     elif self.train_transform is not None:
    #         return self.train_transform(batch)
    #     else:
    #         return batch

    # def state_dict(self):
    #     """Get state dict for checkpointing."""
    #     # track whatever you want here
    #     state = {"current_train_batch_index": self.current_train_batch_index}
    #     return state

    # def load_state_dict(self, state_dict):
    #     """Load state dict from checkpoint."""
    #     # restore the state based on what you tracked in (def state_dict)
    #     self.current_train_batch_index = state_dict["current_train_batch_index"]


def get_random_crop_slices(image_size, n_crops, crop_height, crop_width):
    max_x = image_size[1] - crop_width
    max_y = image_size[0] - crop_height

    crops = []
    for i in range(n_crops):
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crops.append((slice(y, y + crop_height), slice(x, x + crop_width)))

    return crops


def _collate_fn(batch):
    """Function to remove empty batches."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)