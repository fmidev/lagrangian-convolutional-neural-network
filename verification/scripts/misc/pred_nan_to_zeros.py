"""
Small script to change all 
nan values in prediction hdf5 files
to zeros. Hardcoded configuration.

Bent Harnist, FMI - 23.02.2022
"""
import os
import h5py
from tqdm import tqdm
import numpy as np

def nan_to_zero(data) :
    data = data[data == 255] = 0
    return data.astype(np.uint8)

# HARD CODED CONFIGURATION
dir = "/home/users/harnist"
fname = "p1_pred_10p_logcosh_lt5.hdf5"
f_path = os.path.join(dir,fname)
method_name = "t1_puhti_10p_logcosh_lt5_30epoch"
path_template = "{time}/{method}/{lt}/data"


with h5py.File(f_path, 'a') as f : 
    for t in tqdm(f.keys()):
        for method in f[t].keys():
            for lt in f[t][method].keys():
                path = path_template.format(
                    time=t,
                    method=method,
                    lt=lt
                )                
                data = f[path][:]
                data[...] = nan_to_zero(data)

print(f"\n Success!\n ")
                                        
