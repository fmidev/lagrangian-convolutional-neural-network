"""
Abstract class serving as a blueprint for making components of 
prediction pipelines for different nowcasting algorithms. 

Classes are instanciated using a configuration AttrDict.

Depends on Attrdict, numpy, h5py, pysteps

Methods : 
read_input, preprocessing*, nowcast*, postprocessing*, run (chain previous methods), save
* are abstract methods: they have to be implemented explicitely for child classes. 
Other methods can also be overwritten if needs be. 

"""
from abc import ABC, abstractmethod
from datetime import datetime
from attrdict import AttrDict

import numpy as np
import h5py
from pysteps import io, rcparams

from . import io_tools


class PredictionBuilder(ABC):
    """
    Blueprint for making nowcasting prediction building
        pipelines using PYSTEPS.

        Abstract methods that REQUIRE to be implemented include
        * preprocessing
        * nowcast
        * postprocessing
    """

    def __init__(self, config : AttrDict):
        """
        Initializes a builder pipeline with 
        1) parameters from your Pysteps installation
        2) An attribute dictionnary containing all required configs

        Args:
            config (AttrDict): Attribute dictionary defined in
            a YAML file, containing the whole config necessary for
            the pipeline
        """
        # PYSTEPS PARAMETERS

        # the user has to set this
        data_source_name = config.data_source_name

        self.data_source = rcparams.data_sources[data_source_name]
        self.root_path = self.data_source["root_path"]
        self.path_fmt = self.data_source["path_fmt"]
        self.fn_pattern = self.data_source["fn_pattern"]
        self.fn_ext = self.data_source["fn_ext"]
        importer_name = self.data_source["importer"]
        self.importer_kwargs = self.data_source["importer_kwargs"]
        self.timestep = self.data_source["timestep"]
        self.importer = io.get_method(importer_name, "importer")

        # Componentwise parameters
        self.input_params = config.input
        self.preprocessing_params = config.preprocessing
        self.nowcast_params = config.nowcast
        self.postprocessing_params = config.postprocessing
        self.save_params = config.save

        self.date_path = config.datelist_path
        self.hdf5_path = config.hdf5_path

    def run(self, timestamp : str): 
        data, metadata = self.read_input(timestamp = timestamp,
                                      **self.input_params)
        data, metadata = self.preprocessing(data, metadata,
                                            self.preprocessing_params)
        nowcast = self.nowcast(data, self.nowcast_params)
        nowcast = self.postprocessing(nowcast = nowcast, 
                                      params = self.postprocessing_params)
        return nowcast

    def read_input(
        self,
        timestamp: str,
        num_next_files : int = 0
        ):
        """
        Reads in the required input radar 
        products for nowcasting

        Args:
            timestamp (str): The first time used for nowcasts
            num_next_files (int, optional): Number of subsequent
        radar product read in, dependent on the method

        Returns:
            Z (np.ndarray) of size
            (num_next_files + 1, x_size, y_size): Radar products
            metadata (dict) : metadata of those radar products
        """
        
        timestamp = datetime.strptime(
            timestamp,
            '%Y-%m-%d %H:%M:%S'
            )
        fns = io.archive.find_by_date(
            timestamp,
            self.root_path,
            self.path_fmt,
            self.fn_pattern,
            self.fn_ext,
            self.timestep,
            num_next_files = num_next_files
            )
        Z, _, metadata = io.read_timeseries(
            fns, 
            self.importer, 
            **self.importer_kwargs
            )

        return Z, metadata

    @abstractmethod
    def preprocessing(self, data, metadata, params):
        '''
        All the processing of data before nowcasting
        in : data, metadata, params
        out: data, metadata
        '''
        pass

    @abstractmethod
    def nowcast(self, data, params):
        '''
        IN : data, metadata, nowcast_parameters
        OUT : nowcast
        '''
        pass

    @abstractmethod
    def postprocessing(self, data, params):
        '''
        IN : data, postprocessing_params 
        out : data
        '''
        pass

    def save(self, nowcast : np.ndarray,
             group : h5py.Group, save_parameters : AttrDict,
             mask_group : h5py.Group = None):
        """Save the nowcast into the hdf5 file, 
        in the group "group".

        Args:
            nowcast (np.ndarray): (n_timesteps,h,w) shaped predictions
            group (h5py.Group): parent group (eg. 
            timestamp/method) that will contain the saved nowcast
            save_parameters (AttrDict): parameters regarding
            saving nowcasts to the hdf5 file.
        """
        save_indexes = save_parameters.save_indexes
        what_attrs = save_parameters.what_attrs
        use_advection_mask = save_parameters.get("advection_mask",False)

        if use_advection_mask:
            assert mask_group is not None ,"Please provide \
                a mask group if using an advection mask."

        for i in save_indexes:
            leadtime_group = group.require_group(str(i+1))
            if use_advection_mask:
                mask = mask_group[f"{i+1}/data"][:]
                nowcast[i,mask] = np.nan
            nowcast_uint8 = io_tools.arr_compress_uint8(nowcast[i,:,:])
            io_tools.write_image(
                group = leadtime_group,
                ds_name = "data",
                data = nowcast_uint8,
                what_attrs = what_attrs)

    
