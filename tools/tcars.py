import os
import sys
import pdb
from collections import OrderedDict as odict

import numpy as np
import xarray as xr
import pyrrtmg as rrtmg

import tools.constants as constants
from tools.data_tools import encode_time, write_basic_attributes, update_netCDF_file_history
from readers.readers_tcars import import_std_atm, import_trace_gas_csv
from tools.met_tools import rho_air, h2ovmr_to_q, convert_spechum_to_abshum, compute_heating_rate


class tcars:
    
    """
    Loads all necessary constants for radiative transfer simulations using T-CARS (python interface of RRTMG)*.
    
    *: Deneke, H. (2024) hdeneke/pyRRTMG: Release with correct versioning scheme . Zenodo [code]. https://doi.org/10.5281/zenodo.11147087
    
    Init:
    path_tcars_data : str
        Full path where additional data for T-CARS is stored.
    ds : xr.dataset
        Dataset containing atmospheric data. Must have "time" and "height" dimensions.
    """
    
    def __init__(self, path_tcars_data: str, ds: xr.Dataset):
        
        self.ds = ds
        self.path_tcars_data = path_tcars_data
        
        # for more detailed descriptions of the flags below, see 
        # https://github.com/hdeneke/pyRRTMG/blob/master/src_f/rrtmg_sw/column_model/doc/rrtmg_sw_instructions.txt
        self.icld = 1       # icld = 1 if clouds should be included; icld = 0 if clouds are omitted; 
                            # flag values 2 or 3 change the "overlap assumption"
        self.iaer = 0       # iaer = 10: one or more layers contain aerosols; = 0 to omit aerosols
        self.permuteseed_sw = 0     # permuteseed_sw
        self.permuteseed_lw = 0     # permuteseed_lw
        self.irng = 0               # irng
        self.idrv = 0               # idrv https://github.com/hdeneke/pyRRTMG/tree/master/src_f/rrtmg_lw -> 4th paragraph
                                    # --> whether to also calculate the derivative of flux with respect to surface temp
        
        self.adjes = 1.0        # adjustment of total solar irradiance (solar constant)
        self.solar_constant = constants.solar_constant      # solar constant; total solar irradiance at TOA
        self.inflgsw = 2            # 0: direct specif. of cloud opt depth, cloud fraction, single scat albedo, phase function
                                    # 2: ice and liq cloud opt depths are calculated from lwp, iwp, effective radii
        self.inflglw = 2            # similar to inflgsw; see also INFLAG in pyRRTMG doc noted above
        self.iceflgsw = 2           # see ICEFLAG in pyRRTMG doc
        self.iceflglw = 2
        self.liqflgsw = 1           # see LIQFLAG in pyRRTMG doc
        self.liqflglw = 1
        
        # flags for the use of gas volume mixing ratios: 
        # -1: turn off, _vmr is set to 0
        # 0: use reference given in standard atmosphere (Anderson 1986)
        # 1: use values given in DS
        self.iflag_h2o_vmr = 1
        self.iflag_co2_vmr = 0      # additional option: 2: use NOAA measurements
        self.iflag_o3_vmr = 0
        self.iflag_n2o_vmr = 0      # additional option: 2: use NOAA measurements
        self.iflag_ch4_vmr = 0      # additional option: 2: use NOAA measurements
        self.iflag_o2_vmr = 0
        
        self.iflag_emissivity = 0   # -1: use emissivity of 1, 0: use emissivity of 0.996, 
                                    # 1: user-defined emissivity in set_emissivity
                
        self.set_translation_dict()
        self.init_cloud_properties()
        self.init_aerosol_properties()
        self.set_emissivity()
        
        
    def load_background_vmrs(self, which_std_atm='subarctic_summer'):
        
        """
        Loads trace gas data from modeled standard atmosphere and observations.
        """
        
        def std_atm_data_to_tcars_time_height_grid(ds: xr.Dataset):
            
            ds = ds.expand_dims(dim={'time': self.ds.time.values}, axis=0)
            ds = ds.interp(coords={'height': self.ds.height.values}, kwargs={"fill_value": 'extrapolate'})
            
            return ds
        
        def trace_gas_obs_to_tcars_time_height_grid(ds: xr.Dataset):
            
            if ((ds.time.values[-1] < self.ds.time.values[0]) or 
                (ds.time.values[0] > self.ds.time.values[-1])):
                ds = ds.sel(time=self.ds.time.values, method='nearest')
                ds = ds.assign_coords({'time': (['time'], self.ds.time.values)})
            else:
                ds = ds.interp(time=self.ds.time.values).expand_dims(dim={'height': self.ds.height.values}, axis=1)
            ds = ds.expand_dims(dim={'height': self.ds.height.values}, axis=1)
            
            return ds
        
        std_atm_filename_dict = {
            'subarctic_summer': "sas.atm",
            'subarctic_winter': "saw.atm",
            'midlat_summer': "mls.atm",
            'midlat_winter': "mlw.atm",
            'std': "std.atm",
            'tropical': "tro.atm",
            }
        
        filename = os.path.join(self.path_tcars_data, 
                                "std_atm/",
                                std_atm_filename_dict[which_std_atm])
        ds = import_std_atm(filename)
        ds = std_atm_data_to_tcars_time_height_grid(ds)
        
        n_time, n_hgt = len(self.ds.time), len(self.ds.height)
        dims_list = ['time', 'height']
        for var in ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2']:
            var_vmr = var + "_vmr"
            
            if ((f'iflag_{var_vmr}' in self.__dict__.keys()) and (self.__dict__[f'iflag_{var_vmr}'] == 0) or 
                (f'iflag_{var_vmr}' not in self.__dict__.keys())):
                self.ds[var_vmr] = xr.DataArray(ds[var].values, dims=dims_list)
        

        add_ghgs = {'cfc11': 232.0, 
                    'cfc12': 516.0, 
                    'cfc22': 233.0, 
                    'ccl4': 82.0}       # all in ppt, Source: NOAA
        for var in add_ghgs:
            self.ds[var+"_vmr"] = xr.DataArray(np.full((n_time, n_hgt), add_ghgs[var]*constants.sfac['ppt']), 
                                               dims=dims_list)
        
        ds = import_trace_gas_csv(self.path_tcars_data + "/trace_gas_data/NOAA_Annual_Mean_MoleFractions.csv")
        ds = trace_gas_obs_to_tcars_time_height_grid(ds)

        for var in ['co2', 'n2o', 'ch4', 'cfc11', 'cfc12', 'ccl4']:
            var_vmr = var + "_vmr"
            if (((f'iflag_{var_vmr}' in self.__dict__.keys()) and (self.__dict__[f'iflag_{var_vmr}'] == 2)) or 
                (f'iflag_{var_vmr}' not in self.__dict__.keys())):
                self.ds[var_vmr] = xr.DataArray(ds[var].values, dims=dims_list)

    
    def set_rrtmg_input(self, which_std_atm: str):
        
        """
        Forward data loaded into Dataset to dictionaries used by T-CARS (=RRTMG).
        
        Parameters:
        -----------
        which_std_atm : str
            Defines which standard atmosphere to load. The following options are supported:
            - "subarctic_summer"
            - "subarctic_winter"
            - "midlat_summer"
            - "midlat_winter"
            - "std"
            - "tropical"
        """

        self.load_background_vmrs(which_std_atm=which_std_atm)
        self.update_DS()
        self.update_cloud_properties()
        self.update_aerosol_properties()

        atm_args = ['Play', 'Plev',                                  # pressure of layer and level in hPa
                    'Tlay', 'Tlev', 'Tsfc',                          # temperature of layer and level (layer boundaries) in K
                    'h2ovmr', 'o3vmr', 'co2vmr', 'ch4vmr', 'n2ovmr', # vol mix ratio of gases
                    'o2vmr', 'cfc11vmr', 'cfc12vmr', 'ccl4vmr', 'cfc22vmr']
        
        atm = dict()
        for ds_var, t_var in self.trl_.items():
            if t_var in atm_args:
                try:
                    atm[t_var] = np.asfortranarray(self.ds[ds_var], dtype=np.float64)
                except KeyError:
                    print(f"{ds_var} is needed but was not found in the dataset provided to {os.path.basename(__file__)}" + 
                          f" while executing {sys.argv[0]}.")
                
                
        for var in [
                    'alb_dir_uv',           # UV/VIS surface albedo, direct (0.2-0.625 um)
                    'alb_dif_uv',           # UV/VIS surface albedo, diffuse (0.2-0.625 um)
                    'alb_dir_nir',          # near-IR surface albedo, direct (0.625-12.20 um)
                    'alb_dif_nir',          # near-IR surface albedo, diffuse (0.625-12.20 um)
                    'julian_day',           # "Day of the year" or Julian Day number
                    'cos_zenith',           # cosine of the solar zenith angle
                    ]:
            
            if var in self.ds.data_vars:
                self.__setattr__(var, self.ds[var].values)
            
            # Fill values if data not available in DS:
            elif var == 'julian_day':
                self.julian_day = 1
            elif var in ["alb_dir_uv", "alb_dif_uv", "alb_dir_nir", "alb_dif_nir"]:
                self.__setattr__(var, 0.1)
            elif var == 'cos_zenith':
                self.cos_zenith = 0.0
        
        
        self.rrtmg_input = [self.icld,
                            self.iaer,
                            self.permuteseed_sw,
                            self.permuteseed_lw,
                            self.irng,
                            self.idrv]
        
        self.rrtmg_input += [atm[k] for k in atm_args]
        
        self.rrtmg_input += [self.alb_dif_nir, 
                             self.alb_dir_nir, 
                             self.alb_dif_uv, 
                             self.alb_dir_uv,
                             self.emis,
                             self.cos_zenith,
                             self.adjes,
                             self.julian_day,
                             self.solar_constant,
                             self.inflgsw,
                             self.inflglw,
                             self.iceflgsw,
                             self.iceflglw,
                             self.liqflgsw,
                             self.liqflglw]
        
        for var in self.cloud_props.keys():
            self.rrtmg_input += [self.cloud_props[var]]
        for var in self.aerosol_props.keys():
            self.rrtmg_input += [self.aerosol_props[var]]
        
        self.rrtmg_input = odict(zip(rrtmg.input_vars, self.rrtmg_input))
        
        
        for var in ['h2o', 'co2', 'o3', 'n2o', 'ch4', 'o2']:
            if self.__dict__[f'iflag_{var}_vmr'] == -1:
                self.rrtmg_input[var+"vmr"][...] = 0
    
    
    def set_translation_dict(self):
        
        """
        Translation dict to rename variables of self.ds (keys) to the names used by T-CARS 
        (values).
        """
        
        self.trl_ = {'pres': 'Play',
                     'temp': 'Tlay',
                     'temp_sfc': 'Tsfc',
                     'pres_h': 'Plev',
                     'temp_h': 'Tlev',
                     'clwp': 'lwp',
                     'ciwp': 'iwp',
                     'clc': 'cldfrac',
                     're_liq': 're_liq',
                     're_ice': 're_ice',
                     'tauc_sw': 'tauc_sw',
                     'tauc_lw': 'tauc_lw',
                     'ssac_sw': 'ssac_sw',
                     'asmc_sw': 'asmc_sw',
                     'fsfc_sw': 'fsfc_sw',
                     }
        
        for var in ['h2o_vmr', 'co2_vmr', 'o3_vmr', 'n2o_vmr', 'ch4_vmr', 'o2_vmr',
                    'cfc11_vmr', 'cfc12_vmr', 'cfc22_vmr', 'ccl4_vmr']:
            self.trl_[var] = var.replace('_vmr', 'vmr')
    
    
    def init_cloud_properties(self):
        
        self.cloud_props = dict()
        for var in [
                    'tauc_sw',      # in cloud optical depth, short wave
                    'tauc_lw',      # in cloud optical depth, long wave
                    'cldfrac',      # cloud fraction
                    'ssac_sw',      # in cloud single scattering albedo
                    'asmc_sw',      # in cloud assymetry parameter
                    'fsfc_sw',      # in cloud forward scattering fraction, 
                                    # delta function pointing forward; forward peaked scattering
                    'iwp',          # cloud ice water path, in g m-2
                    'lwp',          # cloud liquid water path, in g m-2
                    're_ice',       # effective radius ice, in um
                    're_liq',       # effective radius liquid, in um
                    ]:
            
            shape_ = (len(self.ds.time), len(self.ds.height))
            fill_val = 0.0
            if "_sw" in var:
                shape_ = (rrtmg.nbnd_sw,) + shape_
            elif "_lw" in var:
                shape_ = (rrtmg.nbnd_lw,) + shape_
                
            self.cloud_props[var] = np.full(shape_, fill_val, dtype=np.float64, order='F')
    
    
    def init_aerosol_properties(self):
        
        shape_base = (len(self.ds.time), len(self.ds.height))
        
        kw = {'dtype': np.float64, 'order': "F"}
        self.aerosol_props = {
            "tauaer_sw": np.full(shape_base + (rrtmg.nbnd_sw,), 0.0, **kw),    
                         # aerosol opt depth (iaer=10 only); (ncol,nlay,nbndsw)
                         # iaer=10 means one or more layers contain aerosols
            "ssaaer_sw": np.full(shape_base + (rrtmg.nbnd_sw,), 1.0, **kw),    
                         # aerosol single scat albedo (iaer=10 only)
            "asmaer_sw": np.full(shape_base + (rrtmg.nbnd_sw,), 0.0, **kw),
                         # aerosol assymetry parameter (iaer=10 only)
            "ecaer_sw": np.full(shape_base + (6,), 0.0, **kw),
                        # aerosol opt depth at 0.55 um (iaer=6 only)
            "tauaer_lw": np.full(shape_base + (rrtmg.nbnd_lw,), 0.0, **kw),
                         # aerosol opt depth at mid-point of LW spectral bands
            }
    
    
    def update_DS(self):
        
        """
        Update temperature data on height layers and use temperature of lowest level as surface 
        temperature. This ensures consistency when editing temperature levels.
        """
        
        self.ds['temp'][...] = 0.5*(self.ds['temp_h'].values[...,:-1] + self.ds['temp_h'].values[...,1:])
    
    
    def update_cloud_properties(self):
        
        self.cloud_props = self.update_properties(self.cloud_props)
            
            
    def update_aerosol_properties(self):
        
        self.aerosol_props = self.update_properties(self.aerosol_props)
    
    
    def update_properties(self, props: dict):
        
        prop_args = [*props.keys()]
        for var in self.ds.data_vars:
            
            if var in prop_args:
                var_update = var
            elif (var in self.trl_.keys()) and (self.trl_[var] in prop_args):
                var_update = self.trl_[var]
            else:
                continue
            props[var_update] = self.ds[var].values
        
        return props
    
    
    def set_emissivity(self, custom_emissivity=0.9837):
        
        """
        Emissivity: 16 bands in longwave, see 
        https://github.com/hdeneke/pyRRTMG/blob/master/src_f/rrtmg_lw/column_model/doc/rrtmg_lw_instructions.txt
        -> 3-1000 um
        
        Use 0.999 for sea ice fraction >= 0.5.
        """
        
        array_kwargs = dict(dtype=np.float64, order='F')
        emis_shape = (len(self.ds.time), 16)
        if self.iflag_emissivity == -1:
            emis_val = 1.0
        elif self.iflag_emissivity == 0:
            emis_val = 0.996            # as in Thielke et al. 2022, https://doi.org/10.1038/s41597-022-01461-9
        elif self.iflag_emissivity == 1:
            # define your own emissivity array of emis_shape or use the following:
            emis_val = custom_emissivity
        self.emis = np.ones(emis_shape, **array_kwargs) * emis_val
    
    
    def run_tcars(self):
        
        self.run_sanity_test()
        self.flxhr = rrtmg.calc_flxhr(**self.rrtmg_input)
        self.organise_output()
        
        
    def run_sanity_test(self):
        
        for var in ['play', 'plev', 'tlay', 'tlev', 'clwp', 'ciwp', 'h2ovmr',
                    'o3vmr', 'co2vmr', 'ch4vmr', 'n2ovmr', 'o2vmr', 'cfc11vmr',
                    'cfc12vmr', 'cfc22vmr', 'ccl4vmr']:
            assert np.all(self.rrtmg_input[var] >= 0.), f"{var} is insane! Consider debugging..."
        
        relq_msg = ("Effective radius of liq. droplets seems out of valid bounds [2.5, 60] um " +
                    "(or 0 um for cloudfree height layers).")
        assert np.all((self.rrtmg_input['relq'] >= 0.) & (self.rrtmg_input['relq'] <= 60.)), relq_msg
        
        reic_msg = ("Effective radius of ice particles seems out of valid bounds [13, 130] um " +
                    "(or 0 um for cloudfree height layers).")
        assert np.all((self.rrtmg_input['reic'] >= 0.) & (self.rrtmg_input['reic'] <= 130.)), reic_msg
    
    
    def organise_output(self):
        
        self.out_ds = xr.Dataset(coords=self.ds.coords)
        
        n_time, n_hgt, n_hgt_h = len(self.ds.time), len(self.ds.height), len(self.ds.height_h)
        shape_lay = (n_time, n_hgt)
        shape_lev = (n_time, n_hgt_h)
        for var, flx in self.flxhr.items():
            
            dims_list = ['time']
            if flx.shape == shape_lay:
                dims_list.append('height')
            elif flx.shape == shape_lev:
                dims_list.append('height_h')
            self.out_ds[var] = xr.DataArray(flx, dims=dims_list)
            
            
        # pyRRTMG seems to return incorrect heating rates for the uppermost height layer. 
        # Compute it manually instead:
        spec_hum = h2ovmr_to_q(self.ds.h2o_vmr)
        rho_v = convert_spechum_to_abshum(self.ds.temp, self.ds.pres*100., spec_hum)
        rho = rho_air(self.ds.pres*100., self.ds.temp, rho_v)
        for sky_cond in ['', 'c']:
            for band in ['sw', 'lw']:
                HR_ = compute_heating_rate(self.out_ds[f'{band}uflx{sky_cond}'], 
                                           self.out_ds[f'{band}dflx{sky_cond}'], 
                                           rho, self.ds.height_h, convert_to_K_day=True)
                self.out_ds[f'{band}hr{sky_cond}'] = HR_
        
        
        # information about the variable names and their meanings: 
        # https://github.com/hdeneke/pyRRTMG/blob/master/src_f/_rrtmg.f90 , paragraph "! Output"
        self.out_ds['height'].attrs = {'long_name': "Height of the full levels (centre of layer)", 
                                       'units': "m"}
        self.out_ds['height_h'].attrs = {'long_name': "Height of the half levels (layer boundaries)", 
                                         'units': "m"}
        self.out_ds['swuflx'].attrs = {'long_name': "Upward shortwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['swdflx'].attrs = {'long_name': "Downward shortwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['swdirflx'].attrs = {'long_name': "Direct shortwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['swhr'].attrs = {'long_name': "Shortwave radiative heating rates for layers",
                                       'units': "K day-1"}
        self.out_ds['swuflxc'].attrs = {'long_name': "Clear sky upward shortwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['swdflxc'].attrs = {'long_name': "Clear sky downward shortwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['swdirflxc'].attrs = {'long_name': "Clear sky direct shortwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['swhrc'].attrs = {'long_name': "Clear sky shortwave radiative heating rates for layers",
                                       'units': "K day-1"}
        self.out_ds['lwuflx'].attrs = {'long_name': "Upward longwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['lwdflx'].attrs = {'long_name': "Downward longwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['lwhr'].attrs = {'long_name': "Longwave radiative heating rates for layers",
                                       'units': "K day-1"}
        self.out_ds['lwuflxc'].attrs = {'long_name': "Clear sky upward longwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['lwdflxc'].attrs = {'long_name': "Clear sky downward longwave flux at half level",
                                       'units': "W m-2"}
        self.out_ds['lwhrc'].attrs = {'long_name': "Clear sky longwave radiative heating rates for layers",
                                       'units': "K day-1"}
        
        self.out_ds.attrs['title'] = "RRTMG outputs from T-CARS environment"
    
    
    def save_output(self, path: str, filename: str):
        
        os.makedirs(path, exist_ok=True)
        
        self.out_ds = write_basic_attributes(self.out_ds)
        self.out_ds = update_netCDF_file_history(DS=self.out_ds,
                                                 script_name=os.path.basename(__file__),
                                                 summary_str="create simulations",
                                                 history_attr_exists=False)
        
        self.out_ds = encode_time(self.out_ds)
        
        outfile = os.path.join(path, filename)
        self.out_ds.to_netcdf(outfile, mode='w', format='NETCDF4')
        print(f"Saved {outfile}....")