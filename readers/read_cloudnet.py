import os
import pdb

import xarray as xr
import numpy as np

import _paths
from tools.data_tools import get_files_daterange_filepattern

translate_dict = {'temperature': 'temp', 
                  'pressure': 'pres',
                  'uwind': 'u',
                  'vwind': 'v',
                  'q': 'q'}

def read_cloudnet_categorize_model_data(
    path_data="",
    date0=None,
    date1=None,
    daterange=None,
    file_pattern="__DATE_STRING___lindenberg_categorize.nc"):
    
    def preprocess_model_data(ds: xr.Dataset):
        
        keep_vars = ['temperature', 'pressure', 'q', 'uwind', 'vwind']
        data_vars = [*ds.data_vars]
        remove_vars = [var for var in data_vars if var not in keep_vars]
        ds = ds.drop_vars(remove_vars)
        ds = ds.rename_vars(translate_dict)
        
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
    ds = read_files(files, process_fct=preprocess_model_data)
    
    return ds


def read_cloudnet_microphysics_retrievals_data(
    path_data="",
    date0=None,
    date1=None,
    daterange=None,):
    
    """
    Reads droplet and ice crystal effective radii, as well as liquid and ice water content data.
    """
    
    if path_data == '':
        path_data = os.path.join(os.environ['PATH_DATA_BASE'],
                                 "cloudnet/")
    
    file_patterns = {'der': "__DATE_STRING___lindenberg_der.nc",
                     'ier': "__DATE_STRING___lindenberg_ier.nc",
                     'iwc': "__DATE_STRING___lindenberg_iwc-Z-T-method.nc",
                     'lwc': "__DATE_STRING___lindenberg_lwc-scaled-adiabatic.nc"}
    ds_dict = dict()
    for key, file_pattern in file_patterns.items():
        files = get_files_daterange_filepattern(path_data=path_data,
                                                date0=date0,
                                                date1=date1,
                                                daterange=daterange,
                                                default_daterange=np.array([np.datetime64('2025-10-01')]),
                                                file_pattern=file_pattern)
        
        ds_dict[key] = read_files(files)
    
    ds = xr.merge(ds_dict.values())
    
    return ds


def read_files(files: list, concat_dim='time', process_fct=None):
    
    ds = xr.open_mfdataset(files, concat_dim=concat_dim, combine='nested', 
                           preprocess=process_fct)
    ds = ds.sortby(concat_dim)
    
    return ds