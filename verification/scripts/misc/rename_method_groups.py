"""
Small script to rename the method group in prediction
hdf5 file. Hardcoded configuration.

Bent Harnist, FMI - 23.02.2022
"""

import h5py
from tqdm import tqdm

# HARD CODED CONFIGURATION
fname = "prediction_db_lk_default.hdf5"
path_template = "{time}/{method}"
old_name = "extrapolation"
new_name = "extrapolation_lk_default"

with h5py.File(fname, 'a') as f : 
    for t in f.keys():
        old_path = path_template.format(time = t,
                                        method = old_name)
        new_path = path_template.format(time = t,
                                        method = new_name)
        f.move(source = old_path, dest = new_path)
        
print(f"\n Success!\n Method group in {fname} renamed from {old_name} to {new_name}.")
                                        
