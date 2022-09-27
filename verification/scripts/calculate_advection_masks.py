import sys
sys.path.append("..")

import h5py
from pincast_verif import io_tools
from tqdm import tqdm
import numpy as np

adv_db_path = "/run/user/1000/gvfs/sftp:host=athras.fmi.fi/data/PINCAST/manuscript_1/nowcasts/pysteps/220405_extrapolation.hdf5"
dates_path = "../datelists/fmi_rainy_days_bbox_test.txt"
adv_db = h5py.File(adv_db_path, 'r')
dates = io_tools.read_file(dates_path)
mask_db_path = "../advection_mask_db.hdf5"
mask_db = h5py.File(mask_db_path, mode='w')

for timestamp in tqdm(adv_db):
    pred_grp = f"{timestamp}/extrapolation"
    mask_db.require_group(timestamp)
    for lt in adv_db[pred_grp]:
        mask_db.require_group(lt)
        pred_ds_name = f"{pred_grp}/{lt}/data"
        mask_ds_name = f"{timestamp}/{lt}/data"
        mask = adv_db[pred_ds_name][:]  == 255
        mask_db.create_dataset(mask_ds_name, data=mask, dtype=bool, compression='gzip')

adv_db.close()
mask_db.close()