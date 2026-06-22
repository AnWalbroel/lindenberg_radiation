import os
import pdb

import xarray as xr
import numpy as np

from _paths import path_radiation_sim
from tools.data_tools import get_files_daterange_filepattern


def read_pyRRTMG_simulation(
    path_data="",
    date="2025-10-01",
    hour=np.arange(24)):
    
    """
    Reads shortwave and longwave broadband radiation simulated with pyRRTMG for a given date
    and selected hours.
    
    Parameters:
    -----------
    path_data : str
        Full path where simulated radiation data is located. Can be empty, in which case the 
        default path is assumed.
    date : str
        Date in YYYY-MM-DD.
    hour : int or np.ndarray of int
        Hour(s) of the simulation.
    """
    
    if path_data == '': path_data = path_radiation_sim
    
    file_pattern = "lindenberg_radiation_sim___DATE_STRING__T__HOUR__.nc"
    files = identify_hourly_files(path_data, date, hour, file_pattern=file_pattern)
    
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
    ds = ds.sortby('time')
    
    return ds.load()


def identify_hourly_files(path_data: str, date: str, hour, file_pattern: str):
    
    if isinstance(hour, int):
        file_pattern = file_pattern.replace("__HOUR__", f"{hour:02}")
    elif isinstance(hour, np.ndarray):
        file_pattern = file_pattern.replace("__HOUR__", "*")
    files = get_files_daterange_filepattern(path_data,
                                            date0=date,
                                            file_pattern=file_pattern)
    
    if isinstance(hour, np.ndarray):
        str_hour_array = np.char.add("T", np.char.zfill(hour.astype('str'), 2))
        file_pattern = file_pattern.replace("__DATE_STRING__", "*")
        
        files = np.array(files)
        files = [file for file in files if np.any(np.char.find(file, str_hour_array) != -1)]
    
    return files