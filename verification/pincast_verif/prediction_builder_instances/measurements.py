"""
    Class for computing nowcasts based on iterative CNNs trained with pytorch.
    The model must be accessible from the models module, that must be in your path.
"""
from attrdict import AttrDict
from skimage.measure import block_reduce

import numpy as np
import h5py
from pysteps.utils import conversion, dimension

from .. import io_tools
from ..prediction_builder import PredictionBuilder
try:
    from models import RainNet
except ImportError:
    print("Please import this class from somewhere\
         with your model definition in your path, here for RainNet")



class MeasurementBuilder(PredictionBuilder):

    def __init__(self, config: AttrDict):
        super().__init__(config)

    def read_input(self, timestamp: str, num_next_files: int = 0):
        return super().read_input(timestamp, num_next_files)

    def save(self, nowcast : np.ndarray,
             group : h5py.Group, save_parameters : AttrDict):
        """Save the nowcast into the hdf5 file, 
        in the group "group".

        Args:
            nowcast (np.ndarray): (n_timesteps,h,w) shaped predictions
            group (h5py.Group): parent group (eg. 
            timestamp/method) that will contain the saved nowcast
            save_parameters (AttrDict): parameters regarding
            saving nowcasts to the hdf5 file.
        """
        what_attrs = save_parameters.what_attrs

        nowcast_uint8 = io_tools.arr_compress_uint8(nowcast)
        io_tools.write_image(
            group = group,
            ds_name = "data",
            data = nowcast_uint8,
            what_attrs = what_attrs)
    
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
                "downsampling" : False,
                "threshold" : None
            })

        bbox = params.bbox
        bbox = (
            bbox[0] * metadata["xpixelsize"],
            bbox[1] *  metadata["xpixelsize"],
            bbox[2] *metadata["ypixelsize"],
            bbox[3] * metadata["ypixelsize"]
            )

        data = data.squeeze()
        assert data.ndim == 2  
        
        metadata["yorigin"] = "lower"
        data, metadata = dimension.clip_domain(
            R = data,
            metadata = metadata,
            extent = bbox
            )
        
        data, metadata = conversion.to_rainrate(data,metadata)

        if params.downsampling:
            # Upsample by averaging
            data = block_reduce(
                data, func=np.nanmean, cval=0, block_size=(2, 2)
            )
            
        if params.threshold is not None: 
            data[data < float(params.threshold)] = 0
            
        data, metadata = conversion.to_reflectivity(data,metadata)
        
        if params.nan_to_zero:
            data[~np.isfinite(data)] = -32
        data[data < -32] = -32
    

        return data, metadata


    def nowcast(self,
        data : np.ndarray,
        params : AttrDict = None):
        return data


    def postprocessing(self, nowcast, params: AttrDict = None):
        return nowcast
