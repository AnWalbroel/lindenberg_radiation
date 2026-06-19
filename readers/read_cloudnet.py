import os
import pdb

import xarray as xr
import numpy as np

import _paths
from tools.data_tools import get_files_daterange_filepattern

def read_cloudnet_categorize_model_data(
    path_data="",
    date0=None,
    date1=None,
    daterange=None,
    file_pattern="__DATE_STRING___lindenberg_categorize.nc"):
    
    def post_process_model_data(ds: xr.Dataset):
        
        keep_vars = ['temperature', 'pressure', 'q', 'uwind', 'vwind']
        data_vars = [*ds.data_vars]
        remove_vars = [var for var in data_vars if var not in keep_vars]
        ds = ds.drop_vars(remove_vars)
        ds = ds.drop_dims(['time', 'height'])
        ds = ds.rename({'model_time': 'time', 'model_height': 'height'})
        
        return ds
    
    if path_data == '':
        path_data = os.path.join(os.environ['PATH_DATA_BASE'],
                                 "cloudnet/")
    
    
    files = get_files_daterange_filepattern(path_data=path_data,
                                            date0=date0,
                                            date1=date1,
                                            daterange=daterange,
                                            default_daterange=np.array([np.datetime64('2025-10-01')]),
                                            file_pattern=file_pattern)
    
    ds_all = list()
    for file in files:
        ds = xr.open_dataset(file)
        ds = post_process_model_data(ds)
        
        ds_all.append(ds)
    
    ds = xr.concat(ds_all, dim='time').sortby('time')
    
    return ds