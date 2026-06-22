import pdb

import numpy as np
import csv
import pandas as pd
import xarray as xr

import tools.constants as constants
from tools.met_tools import q_to_h2ovmr
from tools.data_tools import convert_units


def read_default_profiles(scaling=None, offset=None):
    
    
    def apply_offset_and_scaling(data_dict: dict, scaling: dict, offset: dict):
        
        for var in data_dict.keys():
            scaling_ = 1.
            offset_ = 0.
            if var in scaling.keys():
                scaling_ = scaling[var]
            if var in offset.keys():
                offset_ = offset[var]
            
            data_dict[var] = (data_dict[var] + offset_) * scaling_
            
        return data_dict
    
    
    data_dict = dict(
        pres=np.array([955.8902, 850.5316, 754.5992, 667.7425, 589.8407, 519.4208, 455.4800,
                       398.0854, 347.1714, 301.7350, 261.3102, 225.3597, 193.4192, 165.4902, 141.0319,
                       120.1249, 102.6889, 87.8294, 75.1226, 64.3059, 55.0863, 47.2091, 40.5354, 34.7954,
                       29.8654, 22.9835]),     # pressure at full levels, in hPa
    
        pres_h=np.array([1013.000, 902.000, 802.000, 710.000, 628.000, 554.000, 487.000, 426.000, 
                         372.000, 324.000, 281.000, 243.000, 209.000, 179.000, 153.000, 130.000, 111.000, 
                         95.000, 81.200, 69.500, 59.500, 51.000, 43.700, 37.600, 32.200, 27.700, 19.070]),
                         # pressure at half levels, in hPa
        
        temp_h=np.array([294.200, 289.700, 285.200, 279.200, 273.200, 267.200, 261.200, 254.700, 
                         248.200, 241.700, 235.300, 228.800, 222.300, 215.800, 215.700, 215.700, 215.700,
                         215.700, 216.800, 217.900, 219.200, 220.400, 221.600, 222.800, 
                         223.900, 225.100, 228.450]), # temperature at half levels, in K
        
        height_h=np.array([0., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 
                           8000., 9000.,10000., 11000., 12000., 13000., 14000., 15000., 
                           16000., 17000., 18000., 19000., 20000., 21000., 22000., 
                           23000., 24000., 25000., 27500.]),
                           # auxiliary half level heights, in m

        temp_sfc=294.2,        # temperature at surface, in K
        lwp=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0]),    # in-cloud lwp, in g m-2
        iwp=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0]),    # in-cloud iwp, g m-2
        clc=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0]),    # cloud cover
        re_liq=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0]), # effecive radius, in um
        re_ice=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0]), # effective radius ice, in um
        h2o_vmr=np.array([0.0101199, 0.0072961, 0.0048715, 0.0030469, 0.0018778, 
                          0.0011616, 0.0007868, 0.0005183, 0.0003294, 0.0002053, 0.0001066, 
                          0.0000389, 0.0000116, 0.0000040, 0.0000026, 0.0000021, 0.0000020, 
                          0.0000020, 0.0000020, 0.0000020, 0.0000021, 0.0000022, 0.0000023, 
                          0.0000024, 0.0000026, 0.0000027]),   
                          # is actually specific humidity in kg kg-1 --> is converted to vmr below

        co2_vmr=280*constants.sfac['ppm'],
        o3_vmr=0.,
        n2o_vmr=0.32*constants.sfac['ppm'],
        ch4_vmr=1.60*constants.sfac['ppm'],
        o2_vmr=209480*constants.sfac['ppm'],
        
        julian_day=251,
        cos_zenith=0.7,        # cosine of zenith angle
        alb_dir_uv=0.2,        # direct albedo UV spectrum
        alb_dif_uv=0.2,        # diffuse albedo UV spectrum
        alb_dir_nir=0.2,       # direct albedo near IR spectrum
        alb_dif_nir=0.2,       # diffuse albedo near IR spectrum
        
        # tau_aero_sw=0.,      # aerosol optical thickness UV spectrum
        # ssa_aero_sw=0.,      # single scattering albedo UV spectrum
        # asm_aero_sw=0.,      # asymmetry parameter UV spectrum
        # tau_aero_lw=0.,      # aerosol optical thickness IR spectrum
        # emissivity=40*0.996,
        )
    
    for var in ['temp', 'height']:
        data_dict[var] = 0.5*(data_dict[var+"_h"][...,:-1] + data_dict[var+"_h"][...,1:])
    
    data_dict = apply_offset_and_scaling(data_dict, scaling, offset)
    
    data_dict['h2o_vmr'] = q_to_h2ovmr(data_dict['h2o_vmr'])    # kg kg-1 to volume mixing ratio
    time = pd.to_datetime(data_dict['julian_day'], unit="D")
    data_dict['time'] = np.datetime64(pd.Timestamp(f"2020-{time.month}-{time.day}T00:00:00"))
    
    DS = xr.Dataset(coords={'time': (['time'], np.array([data_dict['time']]).astype('datetime64[ns]')),
                            'height': (['height'], data_dict['height']),
                            'height_h': (['height_h'], data_dict['height_h'])})
    n_time, n_hgt, n_hgt_h = len(DS.time), len(DS.height), len(DS.height_h)
    
    gas_list = ['h2o_vmr', 'co2_vmr', 'o3_vmr', 'n2o_vmr', 'ch4_vmr', 'o2_vmr']
    for var in data_dict.keys():
        if var in DS.coords: continue
        
        if isinstance(data_dict[var], (float, int)) and (var in gas_list):
            DS[var] = xr.DataArray(np.broadcast_to(np.array([data_dict[var]]), (n_time, n_hgt)), 
                                   dims=['time', 'height'])
            
        elif isinstance(data_dict[var], (float, int)) and (var not in gas_list):
            DS[var] = xr.DataArray(np.array([data_dict[var]]), dims=['time'])
            
        else:
            var_shape = data_dict[var].shape
            if (n_hgt in var_shape) & (n_time not in var_shape):
                DS[var] = xr.DataArray(np.reshape(data_dict[var], (n_time, n_hgt)), dims=['time', 'height'])
            elif (n_hgt_h in var_shape) & (n_time not in var_shape):
                DS[var] = xr.DataArray(np.reshape(data_dict[var], (n_time, n_hgt_h)), dims=['time', 'height_h'])
            
    return DS
    
    
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