import os
import pdb

import numpy as np
import xarray as xr
from skyfield import api
from skyfield.api import Loader

from _paths import path_radiation_sim, path_tcars_data
from readers.read_cloudnet import (read_cloudnet_categorize_model_data,
                                   read_cloudnet_microphysics_retrievals_data)
from tools.plot_tools import get_cm_cmap, change_colormap_len, create_colourbar
from tools.met_tools import q_to_h2ovmr
from tools.tcars import tcars


def main():
    
    """
    Radiation simulations using pyRRTMG (T-CARS ver).
    """
    
    path_output = path_radiation_sim
    path_plots = os.path.join(os.environ['PATH_PLOTS_BASE'],
                              "data_overview/")
    
    date_str = "2025-10-01"
    station_ele_amsl = 104.                      # station elevation above mean sea level in m (manually extracted from cloudnet data)
    height_grid = np.arange(station_ele_amsl, 25000., 10.)      # lower boundary
    date = np.datetime64(date_str)
    data_quicklooks = False
    set_dict = {'save_figures': False,
                'date_str': date_str}
    
    cn_model_ds = read_cloudnet_categorize_model_data(date0=date)
    cn_mp_ds = read_cloudnet_microphysics_retrievals_data(date0=date)
    if data_quicklooks: data_overview_quicklooks(path_plots, cn_model_ds, cn_mp_ds, **set_dict)
    
    for hour in range(24):
        timeline = set_hourly_simulation_timeline(date=date, hour=hour, time=cn_mp_ds.time)
        ds = prepare_tcars_ds(timeline, cn_model_ds, cn_mp_ds, height_grid)
        tcars_client = tcars(path_tcars_data=path_tcars_data, ds=ds)
        tcars_client = prepare_tcars_client_for_sim(tcars_client)
        
        tcars_client.set_rrtmg_input(which_std_atm="midlat_summer")
        tcars_client.run_tcars()
        
        filename_tcars_sim = f"lindenberg_radiation_sim_{date_str.replace('-','')}T{hour:02}.nc"
        tcars_client.save_output(path_output, filename_tcars_sim)
    
    # TODO: use measured 2 m air temperature
    # TODO: derive surface skin temperature from measured LW up with certain emissivity (which one, 0.998?)
    # TODO: use HATPRO IWV to scale model specific humidity
    # TODO: other surface albedo
    # make test without "radius_or_water_path_is_zero" test in sanity enforcer; or setting re_liq or re_ice to val[0] instead of 0 in first check
    # compare self-computed HR and original HR


def prepare_tcars_client_for_sim(tcars_client):
    
    tcars_client = customise_gas_flags(tcars_client)
    tcars_client = set_emissivity(tcars_client, 0.998)      # TODO ADAPT
    
    return tcars_client


def customise_gas_flags(tcars_client):
    
    tcars_client.iflag_co2_vmr = 2
    tcars_client.iflag_o3_vmr = 0
    tcars_client.iflag_n2o_vmr = 2
    tcars_client.iflag_ch4_vmr = 2
    tcars_client.iflag_o2_vmr = 0
    
    return tcars_client


def set_emissivity(tcars_client, emissivity: float):
    
    tcars_client.iflag_emissivity = 1
    tcars_client.set_emissivity(custom_emissivity=emissivity)
    
    return tcars_client


def prepare_tcars_ds(
    timeline: np.ndarray,
    meteo_ds: xr.Dataset, 
    mp_ds: xr.Dataset, 
    height_grid: np.ndarray):
    
    """
    For certain timeline over which the simulation is supposed to run, extract pressure, 
    temperature and humidity data from meteo data set (meteo_ds) and extract microphysical
    data (droplet effective radii, ice crystal effective radii, ice and liquid water contents) 
    from mp_ds. The data will be put onto levels ('half levels' or 'layer boundaries') and 
    layers ('full levels').
    """
    
    mp_ds = mp_ds.interp({'time': timeline}, method='linear')
    meteo_ds = meteo_ds.interp({'time': timeline}, method='linear')
    
    ds = init_tcars_ds(time=timeline,
                       height_h=height_grid,
                       height=0.5*(height_grid[:-1] + height_grid[1:]))
    
    ds = meteo_to_tcars_ds(ds, meteo_ds)
    ds = add_solar_cos_zen_lindenberg(ds)
    ds = set_albedo(ds)
    ds = add_cloud_data(ds, mp_ds)
    
    return ds


def add_cloud_data(ds: xr.Dataset, mp_ds: xr.Dataset):
    
    def convert_to_rrtmg_units(ds: xr.Dataset):
        
        """
        From kg m-2 to g m-2. And from m to um.
        """
        
        for var in ['clwp', 'ciwp']: ds[var] *= 1000.
        for var in ['re_liq', 're_ice']: ds[var] *= 1e+06
        
        return ds
    
    def cloud_sanity_enforcer(ds: xr.Dataset):
        
        bounds = {'re_liq': np.array([2.5, 60.]),
                  're_ice': np.array([13., 130.])}
        corresponding_water_path = {'re_liq': 'clwp',
                                    're_ice': 'ciwp'}
        for k, val in bounds.items():
            wp = corresponding_water_path[k]
            re_between_0_and_lower_bound = ((ds[k] > 0.) & (ds[k] < val[0]))
            ds[k] = ds[k].where(~re_between_0_and_lower_bound, other=0.)
            ds[wp] = ds[wp].where(~re_between_0_and_lower_bound, other=0.)
            
            radius_or_water_path_is_zero = (ds[k] == 0.) | (ds[wp] == 0.)
            ds[k] = ds[k].where(~radius_or_water_path_is_zero, other=0.)
            ds[wp] = ds[wp].where(~radius_or_water_path_is_zero, other=0.)
        
        return ds
    
    
    mp_ds = prepare_micophysics_dataset(mp_ds, ds.height.values, ds.height_h.values)
    cloud_vars = ['re_liq', 're_ice', 'ciwp', 'clwp']
    for var in cloud_vars:
        if var in ds.data_vars:
            ds[var][...] = mp_ds[var].values
    
    # if (ds['clwp'] > 0).any():
    #     ds['clwp'] = scale_microphysics_lwp_ret_by_hatpro_obs(ds['clwp'], hat_ds['lwp'],
    #                                                          fill_mask=cloud_vars.fill_mask)
    
    ds = convert_to_rrtmg_units(ds)
    ds = cloud_sanity_enforcer(ds)
    
    ds['clc'] = ds['clc'].where((ds.re_liq == 0.) & 
                                (ds.re_ice == 0.) &
                                (ds.ciwp == 0.) &
                                (ds.clwp == 0.), other=1.)
    
    return ds


def prepare_micophysics_dataset(mp_ds: xr.Dataset, height_lay: np.ndarray, height_lev: np.ndarray):
    
    mp_ds = mp_ds.interp({'height': height_lay,
                          'height_lev': height_lev},
                         method='linear',
                         kwargs={'fill_value': 0.})
    for var in ['re_liq', 're_liq_scaled', 're_ice', 'iwc', 'lwc', 'iwc_lev', 'lwc_lev']:
        mp_ds[var] = mp_ds[var].where(~mp_ds[var].isnull(), other=0.)
    mp_ds = get_layer_water_paths(mp_ds)
    
    return mp_ds


def get_layer_water_paths(ds: xr.Dataset):
    
    """
    Compute layer-integrated cloud LWP and IWP (not the total vertical integral of LWC and IWC!).
    ciwp and clwp are height-resolved variables, where e.g.
    ciwp[k] = iwc[k] * (height_lev[k+1] - height_lev[k])
    """
    
    for wc, wp in zip(['iwc', 'lwc'], ['ciwp', 'clwp']):
        wp_values = ds[wc].values*ds.height_lev.diff('height_lev').values
        ds[wp] = xr.DataArray(wp_values, dims=['time', 'height'])
    
    return ds


def set_albedo(DS: xr.Dataset):
    
    """
    Set surface albedo for Lindenberg research site.
    """
    
    for var in ['alb_dir_uv', 'alb_dif_uv']:
        DS[var][...] = 0.2
    for var in ['alb_dir_nir', 'alb_dif_nir']:
        DS[var][...] = 0.2
    
    return DS


def add_solar_cos_zen_lindenberg(ds: xr.Dataset):
    
    lat, lon = 52.208, 14.118
    sun_ele = get_sun_elevation_time_loc(time=ds.time.values, lat=lat, lon=lon)
    cos_zen = np.cos(np.deg2rad(90. - sun_ele))
    ds['cos_zenith'] = xr.DataArray(cos_zen, dims=['time'])
    
    return ds


def get_sun_elevation_time_loc(time: np.ndarray, lat: float, lon: float):
    
    time_sky = load_skyfield_time(time)
    load = custom_skyfield_loader()
    
    sun, earth = create_observable_object(loader=load)
    loc = earth + api.wgs84.latlon(lat, lon, elevation_m=11)
    sun_ele = compute_sun_ele_for_loc(loc, time_sky, sun)
    
    return sun_ele


def load_skyfield_time(time: np.ndarray):
    
    ts = api.load.timescale()
    time_xr = xr.DataArray(time.astype('datetime64[ns]'), dims=['time'],
                           coords={'time': (['time'], time.astype('datetime64[ns]'))})
    time_sky = ts.utc(time_xr.dt.year.values, time_xr.dt.month.values, time_xr.dt.day.values,
                      time_xr.dt.hour.values, time_xr.dt.minute.values)
    
    return time_sky


def custom_skyfield_loader():
    
    return Loader(os.path.join(os.environ['PATH_DATA_BASE'], "skyfield/"))


def create_observable_object(loader=None):
    
    if loader is None:
        eph = api.load("de421.bsp")
    else:
        eph = loader("de421.bsp")
    
    sun = eph['sun']
    earth = eph['earth']
    
    return sun, earth


def compute_sun_ele_for_loc(loc, time, sun):
    
    """
    Observe sun from given location (loc) for a given time. Get azimuth and altitude (elevation)
    angle.
    """
    
    sun_astro = loc.at(time).observe(sun)
    sun_ele_azi = sun_astro.apparent().altaz()
    sun_ele = sun_ele_azi[0].degrees
    
    return sun_ele


def meteo_to_tcars_ds(ds: xr.Dataset, meteo_ds: xr.Dataset):
    
    def meteo_quality_control(ds: xr.Dataset):
        
        assert ((ds.temp > 170.) & (ds.temp < 330.)).all()
        assert ((ds.pres < 1200.)).all()
        assert ((ds.h2o_vmr >= 0.)).all()
        assert (~(ds.temp + ds.pres + ds.h2o_vmr).isnull()).all()
        assert (~(ds.temp_h + ds.pres_h).isnull()).all()
        
        return ds
        
    meteo_ds['h2o_vmr'] = q_to_h2ovmr(meteo_ds.q)
    meteo_ds['pres'] *= 0.01        # Pa to hPa
    meteo_ds = meteo_ds.interp(height=ds.height_h.values)
    
    for var in ['pres', 'temp', 'h2o_vmr']:
        ds[var+"_h"] = xr.DataArray(meteo_ds[var].values,
                                    dims=['time', 'height_h'])
        ds[var] = xr.DataArray(0.5*(meteo_ds[var].isel(height=slice(1,None)).values +
                                    meteo_ds[var].isel(height=slice(None,-1)).values),
                               dims=['time', 'height'])
    
    ds['temp_sfc'][...] = meteo_ds['temp'].isel(height=0).values
    
    ds = ds.drop_vars('h2o_vmr_h')
    ds = meteo_quality_control(ds)
    
    return ds


def init_tcars_ds(
    time: np.ndarray, 
    height_h: np.ndarray,
    height=None):
    
    if height is None:
        height = 0.5*(height_h[...,1:] + height_h[...,:-1])
    
    DS = xr.Dataset(coords={'time': (['time'], time),
                            'height_h': (['height_h'], height_h),
                            'height': (['height'], height)})
    
    n_time = len(time)
    n_hgt_h = len(height_h)
    n_hgt = len(height)
    
    cloud_vars = ['clwp',        # in-cloud liquid water path in g m-2
                  'ciwp',        # in-cloud ice water path in g m-2
                  'clc',        # cloud cover
                  're_liq',     # effective radius liq in um
                  're_ice',     # effective radius ice in um
                  ]
    atmos_vars = ['pres_h',     # pressure at half levels in hPa
                  'pres',       # pressure at full levels in hPa
                  'temp_h',     # temperature at half levels in K
                  'temp',       # temperature at full levels in K
                  'h2o_vmr',    # water vapour volume mixing ratio at full levels
                  ]
    gas_vars = ['co2_vmr',
                'o3_vmr',
                'n2o_vmr',
                'ch4_vmr',
                'o2_vmr']
    sfc_vars = ['alb_dir_uv',   # direct albedo UV spectrum 
                'alb_dif_uv',   # diffuse albedo UV spectrum
                'alb_dir_nir',  # direct albedo near IR spectrum
                'alb_dif_nir',  # diffuse albedo near IR spectrum
                'temp_sfc',     # surface temperature in K
                ]     
    vars = cloud_vars + atmos_vars + gas_vars + sfc_vars
    
    for var in vars:
        if var in ['pres_h', 'temp_h']:
            shape = (n_time,n_hgt_h)
            dims = ('time', 'height_h')
        elif var in sfc_vars:
            shape = (n_time,)
            dims = ('time')
        else:
            shape = (n_time,n_hgt)
            dims = ('time', 'height')
        
        if var in (cloud_vars + gas_vars + sfc_vars):
            fill_val = 0.
        elif var in atmos_vars:
            fill_val = np.nan
        
        DS[var] = xr.DataArray(np.full(shape, fill_val), dims=dims)
    
    DS['julian_day'] = DS.time.dt.dayofyear
    
    return DS


def set_hourly_simulation_timeline(date: np.datetime64, hour: int, time: xr.DataArray):
    
    timeline = time.sel(time=date.astype('datetime64[D]').astype('str')).sel(time=time.dt.hour == hour).values
    
    return timeline


def data_overview_quicklooks(
    path_plots: str,
    cn_model_ds: xr.Dataset, 
    cn_mp_ds: xr.Dataset,
    save_figures=False,
    date_str="2025-10-01"):
    
    import matplotlib as mpl
    mpl.use("WebAgg")
    import matplotlib.pyplot as plt
    
    def plot_data(ds: xr.Dataset, 
                  vars: list, 
                  cmaps: list, 
                  vmins: list, 
                  vmaxs: list,
                  ylims=None,
                  filename=f"lindenberg_overview_{date_str}"):
        
        f1, axs = plt.subplots(4, 1, sharex=True, figsize=(12,10))
    
        for ax, var, cmap_str, vmin, vmax in zip(axs, 
                                                 vars, 
                                                 cmaps, 
                                                 vmins,
                                                 vmaxs):
            cmap = get_cm_cmap(cmap_str, to_listed_cmap=True)
            img = ax.pcolormesh(ds.time, ds.height, ds[var].T,
                                cmap=cmap, vmin=vmin, vmax=vmax)
            
            f1, ax = create_colourbar(f1, ax, img, cb_label=f"{var} ({ds[var].units})")
            ax.set_ylabel("Height (m)")
            ax.set_xlabel(", ".join(["Time", date_str]))
            if ylims is not None: ax.set_ylim(ylims)
            ax.label_outer()
        
        if save_figures:
            os.makedirs(path_plots, exist_ok=True)
            
            plotfile = os.path.join(path_plots, filename + ".png")
            f1.savefig(plotfile, dpi=200)

            print(f"Saved {plotfile} ....")
            
        else:
            plt.show()
            pdb.set_trace()
            
        plt.close()

    date_str_short = date_str.replace("-", "")
    
    vars_model = ['temp', 'q', 'u', 'v']
    cmaps_model = ['batlow', 'oslo_r', 'vik', 'vik']
    vmins_model = [200, 0.0, -40, -40]
    vmaxs_model = [290, 0.006, 40, 40]
    vars_mp = ['re_liq', 're_ice', 'iwc', 'lwc']
    cmaps_mp = ['batlow', 'batlow', 'batlow', 'batlow']
    vmins_mp = [None, None, None, None]
    vmaxs_mp = [None, None, None, None]
    vars_mp_err = ['re_liq_error', 're_ice_error', 'iwc_error', 'lwc_error']
    cmaps_mp_err = ['batlow', 'batlow', 'batlow', 'batlow']
    vmins_mp_err = [None, None, None, None]
    vmaxs_mp_err = [None, None, None, None]
    vars_mp_ret = ['re_liq_retrieval_status', 're_ice_retrieval_status', 'iwc_retrieval_status', 'lwc_retrieval_status']
    cmaps_mp_ret = ['batlow', 'batlow', 'batlow', 'batlow']
    vmins_mp_ret = [None, None, None, None]
    vmaxs_mp_ret = [None, None, None, None]
    plot_data(cn_model_ds, vars_model, cmaps_model, vmins_model, vmaxs_model,
              ylims=[0., 25000.],
              filename=f"lindenberg_cloudnet_model_overview_{date_str_short}")
    plot_data(cn_mp_ds, vars_mp, cmaps_mp, vmins_mp, vmaxs_mp,
              ylims=[0,4000],
              filename=f"lindenberg_cloudnet_microphysics_overview_{date_str_short}")
    plot_data(cn_mp_ds, vars_mp_err, cmaps_mp_err, vmins_mp_err, vmaxs_mp_err,
              ylims=[0,4000],
              filename=f"lindenberg_cloudnet_microphysics_err_overview_{date_str_short}")
    plot_data(cn_mp_ds, vars_mp_ret, cmaps_mp_ret, vmins_mp_ret, vmaxs_mp_ret,
              ylims=[0,4000],
              filename=f"lindenberg_cloudnet_microphysics_ret_overview_{date_str_short}")


if __name__ == "__main__":
    main()