import os
import pdb

import numpy as np
import xarray as xr

from _paths import path_radiation_obs
from tools.data_tools import get_files_daterange_filepattern


def read_bsrn(
    path_data="",
    date0=None,
    date1=None,
    daterange=None,
    file_pattern="sups_rao_rad03_l1_any_v00___DATE_STRING__.nc"):
    
    if path_data == '': path_data = path_radiation_obs
    
    files = get_files_daterange_filepattern(path_data=path_data,
                                            date0=date0,
                                            date1=date1,
                                            daterange=daterange,
                                            default_daterange=np.array([np.datetime64('2025-10-01')]),
                                            file_pattern=file_pattern)
    
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
    ds = ds.rename({'DATETIME': 'time'})
    ds = ds.assign_coords({'time': ds.time})
    ds = ds.sortby('time')
    
    return ds