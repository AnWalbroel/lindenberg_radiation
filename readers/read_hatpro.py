import os
import pdb

import numpy as np
import xarray as xr

from _paths import path_hatpro_data
from tools.data_tools import get_files_daterange_filepattern, running_mean_pdtime
from readers.read_weather_station import rename_data_vars_incl_substrings, set_bad_quality_to_nan

hatpro_translate_dict = {
    'iwv': 'iwv',
    'lwp': 'lwp',
    'temperature': 'temp',
    'absolute_humidity': 'rho_v',
    'relative_humidity': 'relhum',
    'potential_temperature': 'pot_temp',
    'equivalent_potential_temperature': 'theta_e',
    }

def read_hatpro_derived_data(
    path_data="",
    vars_in=['iwv','lwp','temperature','absolute_humidity'],
    date0=None,
    date1=None,
    daterange=None,
    file_pattern="__DATE_STRING___lindenberg_hatpro-single_442ec2ea.nc",
    remove_bad_quality=False,
    apply_running_mean_sec=0):
    
    if path_data == '': path_data = path_hatpro_data
    
    files = get_files_daterange_filepattern(path_data=path_data,
                                            date0=date0,
                                            date1=date1,
                                            daterange=daterange,
                                            default_daterange=np.array([np.datetime64('2025-10-01')]),
                                            file_pattern=file_pattern)
    
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
    ds = ds.sortby('time')
    
    del_vars = [var for var in ds.data_vars if ((var not in vars_in) and (np.all(np.char.find(var, vars_in) == -1)))]
    ds = ds.drop_vars(del_vars)
    
    ds = rename_data_vars_incl_substrings(ds, hatpro_translate_dict)
    
    if remove_bad_quality:
        ds = set_bad_quality_to_nan(ds, good_quality_condition=lambda x: x == 0)
    
    if apply_running_mean_sec > 0:
        vars_for_running_mean = [hatpro_translate_dict[var_in] for var_in in vars_in]
        for var in vars_for_running_mean:
            if ds[var].ndim > 1: raise NotImplementedError
            ds[var][...] = running_mean_pdtime(ds[var].values, 
                                               apply_running_mean_sec, 
                                               ds[var].time.values)
    
    return ds.load()
