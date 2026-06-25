import pdb

import numpy as np
import csv
import pandas as pd
import xarray as xr

import tools.constants as constants
from tools.met_tools import q_to_h2ovmr
from tools.data_tools import convert_units


def import_anderson_std_atm(file: str):
    
    """
    Imports greenhouse gas concentrations from standard atmospheres given by
    Anderson, G P, Clough, S A, Kneizys, F X, Chetwynd, J H, and Shettle, E P. AFGL (Air Force 
    Geophysical Laboratory) atmospheric constituent profiles (0. 120km). Environmental research 
    papers. United States: N. p., 1986. Web.
    Converts to volumne mixing ratios.
    
    Parameters:
    -----------
    file : str
        Full path to additional T-CARS data.
    """
    
    df = pd.read_table(file, sep='\s+',
                       names=('height','PPP','TTT','air','h2o','co2','o3','n2o','co','ch4','o2'),
                       dtype={'height': np.float32,'PPP': np.float32,'TTT': np.float32,'air': np.float32,
                              'h2o': np.float32,'co2': np.float32,'o3': np.float32,'n2o': np.float32,
                              'co': np.float32,'ch4':np.float32,'o2':np.float32}, skiprows=1)
    df.index = df['height']

    ds = xr.Dataset(coords={'height': (['height'], df.index.values*1000.)})
    for var in df.columns:
        if var in ds.coords: 
            continue
        elif var in ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2']:
            df[var] = constants.sfac['ppm'] * df[var].values
        ds[var] = xr.DataArray(df[var].values, dims=['height'])
    
    return ds


def import_std_atm(file: str):
    
    """
    Imports height, pressure, temperature, water vapour, CO2, O3, N2O, CO, CH4 and O2 data from
    University of Oxford, National Centre for Earth Observation - Natural Environmental Research 
    Council: RFM Atmospheric Profiles - FASCODE/ICRCCM Model Atmospheres, 
    https://eodg.atm.ox.ac.uk/RFM/atm/
    This data also seems to be the same as in
    Anderson, G P, Clough, S A, Kneizys, F X, Chetwynd, J H, and Shettle, E P. AFGL (Air Force 
    Geophysical Laboratory) atmospheric constituent profiles (0. 120km). Environmental research 
    papers. United States: N. p., 1986. Web.
    This reader is also based on the read_atm.py provided in https://eodg.atm.ox.ac.uk/RFM/atm/.
    Converts gas data from ppmv to volume mixing ratios, and uses SI units elsewhere.
    
    Parameters:
    -----------
    file : str
        Full path and file name to standard atmosphere data.
    """
    
    vmr_data = ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2']
    rename_dict = {'pre': 'pres',
                   'tem': 'temp',}
    unit_conv_dict = {'pres': [0., 100.]}
    for vmr_ in vmr_data:
        rename_dict[vmr_] = vmr_
        unit_conv_dict[vmr_] = [0., constants.sfac['ppm']]
    
    with open(file) as f:
        rec = '!'
        while rec[0] == '!': rec = f.readline()  # skip initial comments
        flds = rec.split()
        nlev = int(flds[0])
        atm = { 'nlev':nlev } 
        rec = f.readline()
        while rec[0:4].lower() != '*end':  # repeat until end marker record
            if rec[0] == '!': continue 
            flds = rec.split()
            key = flds[0][1:].lower() # remove '*' and change to lower case
            prf = np.fromfile(f,sep=", ",count=nlev)
            atm[key] = prf
            rec = f.readline()
            if rec == '': break   # also exit if end-of-file without *END
    
    ds = xr.Dataset(coords={'height': (['height'], atm['hgt']*1000.)})
    for key, val in rename_dict.items():
        ds[val] = xr.DataArray(atm[key], dims=['height'])
    
    for var in unit_conv_dict:
        ds[var] = convert_units(ds[var], unit_conv_dict[var])
    
    return ds


def import_trace_gas_csv(file):

    """
    Imports the NOAA Trace Gas data from "https://gml.noaa.gov/aggi/aggi.html" and puts it
    into an xarray data set.

    Parameters:
    -----------
    file : str
        Full path and filename of trace gas concentration data in a .csv table.
    """
    
    def get_species_units(lines: np.ndarray):
        
        varnames = lines[2]
        tg_species = [species.lower().replace('-', '') for species in varnames if species != ""]
        units = [unit for unit in lines[3] if unit != ""]
        assert len(tg_species) == len(units)
        
        return tg_species, units
    
    def to_vmr_units(DS: xr.Dataset, tg_species: list, units: list):
        
        for unit, var in zip(units, tg_species):
            DS[var] *= constants.sfac[unit]
        
        return DS
    
    with open(file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        list_of_lines = [row for row in csv_reader]
    
    lines = np.array(list_of_lines)
    n_lines = lines.shape[0]
    first_footer_line_no = np.where(np.all(lines == '', axis=1))[0][0]
    
    tg_species, units = get_species_units(lines)

    header = 3
    DF = pd.read_csv(file, sep=',', header=header, index_col=0, 
                     names=tg_species, na_values='nd',
                     nrows=n_lines - (n_lines-first_footer_line_no)-header-1)
    DF.index.name = 'time'
    DS = DF.to_xarray()
    
    time = np.floor(DS.time.values).astype('int64')
    time = np.asarray([np.datetime64(f"{t}-07-01T00:00:00") for t in time])
    DS = DS.assign_coords({'time': (['time'], time.astype('datetime64[ns]'))})
    
    DS = to_vmr_units(DS, tg_species, units)

    return DS