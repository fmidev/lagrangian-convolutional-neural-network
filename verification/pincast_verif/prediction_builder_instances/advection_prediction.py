"""
    Class for computing PYSTEPS nowcasts based on advection extrapolation
"""

from attrdict import AttrDict
import numpy as np
import h5py
from pysteps import motion, nowcasts
from pysteps.utils import conversion, transformation, dimension

from .. import io_tools
from ..prediction_builder import PredictionBuilder


class AdvectionPrediction(PredictionBuilder):

    def __init__(self, config: AttrDict):
        super().__init__(config)

    def read_input(self, timestamp: str, num_next_files: int = 0):
        return super().read_input(timestamp, num_next_files)

    def save(self, nowcast: np.ndarray, group: h5py.Group,
             save_parameters: AttrDict):
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
                "threshold" : 0.1,
                "zerovalue" : 15.0,
                "bbox" : [125,637,604,1116],
                "nan_to_zero" : True, 
                "downscaling" : 1.0,
                "db_transform" : False,
                "convert" : True,
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
        if params.convert:
            data, metadata = conversion.to_rainrate(data,metadata)
            if params.db_transform : 
                data, metadata = transformation.dB_transform(
                    R = data, 
                    metadata = metadata,
                    threshold=params.threshold,
                    zerovalue=params.zerovalue
                    )
            else:
                data[data < params.threshold] = metadata["zerovalue"]
        else:
            data[data < params.threshold] = metadata["zerovalue"]
                

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
        "Advection extrapolation, S-PROG, LINDA feasible"

        if params is None:
            params = AttrDict({
                "nowcast_method" : "advection",
                "sample_slice" : [None, -1, None],
                "oflow_slice" : [0, -1, 1],
                "n_leadtimes" : 36,
                "oflow_params" : {
                    "oflow_method" : "lucaskanade"
                },
                "nowcast_params" : {}
            })


        oflow_name = params.oflow_method
        oflow_fun = motion.get_method(oflow_name)
        nowcast_name = params.nowcast_method
        nowcast_fun = nowcasts.get_method(nowcast_name)
        sample_slice = slice(*params.sample_slice)
        oflow = oflow_fun(data, **params.oflow_params)

        nowcast = nowcast_fun(
            data[sample_slice,...].squeeze(),
            oflow, 
            params.n_leadtimes,
            **params.nowcast_params
        )
        return nowcast


    def postprocessing(self, nowcast, params):

        if params is None:
            params = AttrDict({
                "threshold" : -10,
                "zerovalue" : 0,
                "nan_to_zero" : True,
                "db_transform" : False,
                "convert" : True
            })
            
        if params.convert:
            if params.db_transform:
                    nowcast, _ = transformation.dB_transform(
                        nowcast,
                        threshold = params.threshold,
                        zerovalue = params.zerovalue,
                        inverse=True)
            else:
                nowcast[nowcast < params.threshold] = params.zerovalue
            nowcast = io_tools.rainrate_to_dBZ(nowcast)
        else:
            nowcast[nowcast < params.threshold] = params.zerovalue

        if params.nan_to_zero:
            nowcast[~np.isfinite(nowcast)] = params.zerovalue
        
        if nowcast.ndim == 4: # S,T,W,H case
            nowcast = nowcast.transpose(1,0,2,3)
            # putting T axis first for saving 1 lt S,W,H preds together
            
        return nowcast

