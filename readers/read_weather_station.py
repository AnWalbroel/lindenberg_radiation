import os
import pdb

import numpy as np
import xarray as xr

from _paths import path_meteo_obs
from tools.data_tools import get_files_daterange_filepattern

meteo_var_translate_dict = {
    'air_pressure': 'pres',
    'air_temperature': 'temp',
    'relative_humidity': 'relhum',
    'wind_direction': 'wdir',
    'wind_speed': 'wspeed',
    'wind_speed_gust': 'wspeed_gust',
    'dew_point_temperature': 'dew_point',
    
    }

def read_weather_station_data(
    path_data="",
    date0=None,
    date1=None,
    daterange=None,
    file_pattern="__DATE_STRING___lindenberg_weather-station_ffb25f43.nc",
    remove_bad_quality=False):
    
    if path_data == '': path_data = path_meteo_obs
    
    files = get_files_daterange_filepattern(path_data=path_data,
                                            date0=date0,
                                            date1=date1,
                                            daterange=daterange,
                                            default_daterange=np.array([np.datetime64('2025-10-01')]),
                                            file_pattern=file_pattern)
    
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
    ds = ds.sortby('time')
    
    ds = rename_data_vars_incl_substrings(ds, meteo_var_translate_dict)
    
    if remove_bad_quality:
        ds = set_bad_quality_to_nan(ds, good_quality_condition=lambda x: x >= 4)
    
    return ds.load()


def rename_data_vars_incl_substrings(ds: xr.Dataset, translate_dict: dict):
    
    old_vars = np.asarray([*translate_dict.keys()])
    for dv in ds.data_vars:
        old_varname_in_dv = (np.char.find(dv, old_vars) != -1)
        if np.any(old_varname_in_dv):
            new_var_idx = np.where(old_varname_in_dv)[0][0]
            ds = ds.rename({dv: dv.replace(old_vars[new_var_idx], translate_dict[old_vars[new_var_idx]])})
    
    return ds


def set_bad_quality_to_nan(ds: xr.Dataset, quality_flag_name='quality_flag', good_quality_condition=lambda x: x == 0):
    
    data_vars = ds.data_vars
    for var in data_vars:
        var_qf = "_".join([var, "quality_flag"])
        if var_qf in data_vars:
            ds[var] = ds[var].where(good_quality_condition(ds[var_qf]), other=np.nan)

    return ds