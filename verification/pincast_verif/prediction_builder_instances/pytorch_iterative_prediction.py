"""
    Class for computing nowcasts based on iterative CNNs trained with pytorch.
    The model must be accessible from the models module, that must be in your path.
"""
from attrdict import AttrDict
import warnings

import numpy as np
import h5py
try:
    import torch
except ImportError: 
    warnings.warn("No pytorch installation found", category=ImportWarning)
try:
    from models import RainNet
except ImportError:
    warnings.warn("No models module found.", category=ImportWarning)

from pysteps.utils import conversion, dimension

from .. import io_tools
from ..prediction_builder import PredictionBuilder




class PytorchIterativePrediction(PredictionBuilder):

    def __init__(self, config: AttrDict):
        super().__init__(config)
        model_path = config.model_path
        model_device = torch.device(config.model_device)
        modelconf = config.modelconf
        self.model = RainNet.load_from_checkpoint(model_path, config=modelconf)
        self.model.to(model_device)
        self.model.eval()

    def read_input(self, timestamp: str, num_next_files: int = 0):
        return super().read_input(timestamp, num_next_files)

    def save(self, nowcast: np.ndarray, group: h5py.Group, save_parameters: AttrDict):
        return super().save(nowcast, group, save_parameters)
    
    def run(self, timestamp: str):
        return super().run(timestamp)

    def preprocessing(self,
        data : np.ndarray,
        metadata : dict,
        params : AttrDict = None):
        '''
        All the processing of data before nowcasting
        in : data, metadata, params
        out: data, metadata
        '''

        if params is None:
            params = AttrDict({
                "bbox" : [125,637,604,1116],
                "nan_to_zero" : True,
                "downscaling" : 1.0,
                "final_im_size" : [512,512]
            })

        bbox = params.bbox
        bbox = (
            bbox[0] * metadata["xpixelsize"],
            bbox[1] *  metadata["xpixelsize"],
            bbox[2] *metadata["ypixelsize"],
            bbox[3] * metadata["ypixelsize"]
            )

        metadata["yorigin"] = "lower"
        data, metadata = dimension.clip_domain(
            R = data,
            metadata = metadata,
            extent = bbox
            )
        data, metadata = conversion.to_rainrate(data,metadata)

        if params.nan_to_zero:
            data[~np.isfinite(data)] = metadata["zerovalue"]
            
        if params.downscaling != 1.0 : 
             metadata["xpixelsize"] = metadata["ypixelsize"]
             data, metadata = dimension.aggregate_fields_space(
                data,
                metadata,
                metadata["xpixelsize"] * params.downscaling
                )
            

        return data, metadata


    def nowcast(self,
        data : np.ndarray,
        params : AttrDict = None):
        "Pytorch RainNet Nowcast"
        
        def scaler(data: torch.Tensor):
            # with ssim and ms-ssim, we scale NN inputs to (0,1)
            if "ssim" in params.nowcast_method:
                return (torch.log(data+0.01) + 5) / 10
            # logcosh uses simple log scaling ~ (-5,+5)
            else:
                return torch.log(data+0.01)

        def invScaler(data: torch.Tensor):
            if "ssim" in params.nowcast_method:
                return (torch.exp((data*10)-5) - 0.01)
            else:
                return torch.exp(data) - 0.01

        if params is None:
            params = AttrDict({
                "n_leadtimes" : 24,
                "im_size" : [512,512]
            })
        
        with torch.no_grad():
            out = torch.empty((params.n_leadtimes, *params.im_size))
            in_data = scaler(torch.Tensor(data))
            in_data = in_data[None, ...]
            for i in range(params.n_leadtimes):
                pred = self.model(in_data)
                out[i,:,:] = invScaler(pred.squeeze())
                in_data = torch.roll(in_data, -1, dims=1)
                in_data[:,3,:,:] = pred

        return out.numpy()


    def postprocessing(self, nowcast, params):

        if params is None:
            params = AttrDict({
                "zerovalue" : 0,
                "nan_to_zero" : True
            })

        if params.nan_to_zero:
            nowcast[~np.isfinite(nowcast)] = params.zerovalue

        nowcast = io_tools.rainrate_to_dBZ(nowcast)

        return nowcast
        
