import os
import glob
import pdb

import numpy as np
import xarray as xr


def running_mean_pdtime(x, N, t):
    
    """
    Running mean of a 1D array x with a window width of N seconds.

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width in seconds.
    t : array of floats
        1D time vector (in numpy datetime64[ns]) required to
        compute the actual running mean window width.
    """

    # first, create xarray DataArray and convert it to pandas DataFrame:
    x_DA = xr.DataArray(x, dims=['time'], coords={'time': (['time'], t)})
    x_DF = x_DA.to_dataframe(name='x')

    # compute running mean (rolling mean): center=True is recommended to have a 5-min running
    # at 2020-01-01T14:00:00 from 2020-01-01T13:57:30 until 2020-01-01T14:02:30.
    x_rm = x_DF.rolling(f"{int(N)}s", center=True).mean().to_xarray().x

    return x_rm.values


def encode_time(
    DS: xr.Dataset, 
    time_var='time', 
    time_dim='time', 
    reference_period=np.datetime64("1970-01-01T00:00:00"),
    calendar="proleptic_gregorian"):
    
    """
    Encode the time dimension of a Dataset with respect to a given reference period.
    
    Parameters:
    -----------
    DS : xr.Dataset
        Dataset whose time dimension should be encoded.
    time_var : str
        Name of the time variable to be encoded
    time_dim : str
        Name of the time dimension of the variable to be encoded.
    reference_period : np.datetime64
        Reference period as np.datetime64 object, given in YYYY-MM-DDThh:mm:ss (ISO 8601).
    calendar : str
        String describing the calendar used. Numpy's datetime64 uses 'proleptic_gregorian'.
        See also https://cfconventions.org/cf-conventions/cf-conventions.html#calendar .
    """
    
    reference_period_str = str(reference_period).replace("T", " ")
    
    time_values = (DS[time_var].values - reference_period).astype('timedelta64[s]').astype(np.float64)
    if time_var == time_dim:
        DS[time_var] = time_values
    else:
        DS[time_var] = xr.DataArray(time_values, dims=time_dim)
    DS[time_var].attrs['units'] = f"seconds since {reference_period_str}"
    DS[time_var].encoding['units'] = f'seconds since {reference_period_str}'
    DS[time_var].encoding['dtype'] = 'double'
    
    if time_var == 'time': DS[time_var].attrs['standard_name'] = 'time'
    DS[time_var].attrs['calendar'] = calendar
    
    return DS


def write_basic_attributes(DS: xr.Dataset):
    
    DS.attrs['institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
    DS.attrs['contact'] = "Andreas Walbroel (a.walbroel@uni-koeln.de, https://orcid.org/0000-0003-2603-2724)"
    DS.attrs['author'] = "Andreas Walbroel"
    DS.attrs['licence'] = "CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/"
    
    return DS


def update_netCDF_file_history(
    DS: xr.Dataset, 
    script_name: str, 
    summary_str="", 
    history_attr='history',
    history_attr_exists=True):

    """
    Updates the history of an xarray Dataset that shall be saved to a netCDF file.
    
    Parameters:
    -----------
    DS : xr.Dataset
        Dataset to be saved and where the attribute is added.
    script_name : str
        Name of the script that was mainly used to update/modify DS.
    summary_str : str
        String that concisely describes the changes made to DS.
    history_attr : str
        Name of the attribute where the history of DS is described.
    history_attr_exists : bool
        Boolean indicating whether the hisotry attribute already exists. If False,
        it's created.
    """
    
    if not history_attr_exists: DS.attrs[history_attr] = ""

    attr_add = ""
    if history_attr_exists and (";" not in DS.attrs[history_attr][-2:]):
        attr_add = "; "
    DS.attrs[history_attr] += (f"{attr_add}{str(np.datetime64('now')).replace('T', ' ')}" +
                               f", {summary_str} with {script_name}; ")
    
    return DS


def convert_units(data: np.ndarray, unit_conv_list: list):
    
    """
    Convert some units: first (second) element of list: must be added to the data (the data 
    must be multiplied by) to get to the desired unit. The multiplication is performed after 
    adding the unit_conv_list[0] value.
    """
    
    return (data + unit_conv_list[0])*unit_conv_list[1]


def convert_units_back(data: np.ndarray, unit_conv_list: list):
    
    """
    Inverse of 'convert_units'. Undo conversion changes.
    """
    
    return (data / unit_conv_list[1]) - unit_conv_list[0]


def identify_files_daterange(path: str, daterange: np.ndarray, file_pattern: str, yyyymmdd_delim=""):
    
    """    
    Parameters:
    -----------
    path : str
        Full path where files containing the data are located.
    daterange : np.ndarray
        Array of np.datetime64 indicating the date range.
    file_pattern : str
        String indicating the file pattern of the data.
    yyyymmdd_delim : str
        Delimiter used between year, month and days in the date strings of the data files.
    """
    
    daterange = daterange.astype('datetime64[D]')

    files = list()
    for date in daterange:
        date_str = str(date).replace("-", yyyymmdd_delim)
        file = glob.glob(path + file_pattern.replace("__DATE_STRING__", date_str))
        if len(file) >= 1:
            files.extend(file)
    
    return files


def handle_daterange_or_date_start_end(
    daterange=None, 
    date0=None, 
    date1=None,
    default_daterange=np.array([np.datetime64("2000-01-01")])):
    
    if (daterange is None) and (date0 is None) and (date1 is None):
        daterange = default_daterange
    elif (daterange is None) and ((date0 is not None) and (date1 is not None)):
        daterange = np.arange(np.datetime64(date0), np.datetime64(date1) + np.timedelta64(1, "D"),
                              np.timedelta64(1, "D"))
    elif (daterange is None) and (date0 is not None) and (date1 is None):
        daterange = np.array([np.datetime64(date0)])
        
    return daterange


def get_files_daterange_filepattern(
    path_data: str,
    date0=None,
    date1=None,
    daterange=None,
    default_daterange=np.array([np.datetime64("2001-01-01")]),
    file_pattern="*__DATE_STRING__*.nc",
    yyyymmdd_delim=""):
    
    daterange = handle_daterange_or_date_start_end(daterange, 
                                                   date0,
                                                   date1,
                                                   default_daterange=default_daterange)
    
    files = identify_files_daterange(path=path_data,
                                     daterange=daterange,
                                     file_pattern=file_pattern,
                                     yyyymmdd_delim=yyyymmdd_delim)
    
    return files