import os
import sys
wdir = os.getcwd()
if wdir not in sys.path: sys.path.append(wdir)
import pdb

import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

import _paths
from readers.read_radiation_sim import read_pyrrtmg_simulation
from readers.read_radiation_obs import read_bsrn

def main():
    
    path_plots = os.path.join(os.environ['PATH_PLOTS_BASE'],
                              "compare_radiation_sim_obs/")
    
    date_str = "2025-10-01"
    date_str_yyyymmdd = date_str.replace('-','')
    date = np.datetime64(date_str)
    set_dict = {'save_figures': False}
    
    
    ds_obs = read_bsrn(date0=date_str)
    ds_sim = read_pyrrtmg_simulation(date=date_str)
    
    ds_sim = ds_sim.isel(height_h=0)
    ds_sim['swdifflx'] = compute_diffuse_from_direct_and_global_radiation(ds_sim['swdirflx'], ds_sim['swdflx'])
    
    
    # plot_obs_overview(ds_obs, path_plots,
    #                   plot_name=f"lindenberg_radiation_obs_overview_{date_str_yyyymmdd}",
    #                   **set_dict)
    plot_obs_overview_with_sim_overlying(ds_obs,
                                         ds_sim, 
                                         path_plots, 
                                         plot_name=f"lindenberg_radiation_sim_obs_{date_str_yyyymmdd}", 
                                         **set_dict)


def plot_obs_overview(
    ds_obs: xr.Dataset,
    path_plots: str,
    plot_name: str,
    save_figures=False):
    
    f1, axs = plt.subplots(4,1, sharex=True, figsize=(10,10))
    
    vars_obs = ["rsds", "rsdifds", "rsdirds", "rlds"]
    vars_sim = ["swdflx", "swdifflx", "swdirflx", "lwdflx"]
    for ax, var_obs, var_sim in zip(axs, vars_obs, vars_sim):
        ax.plot(ds_obs.time, ds_obs[var_obs], label='direct')
        ax.fill_between(ds_obs.time, ds_obs[var_obs+"_min"], ds_obs[var_obs+"_max"], 
                        color=(0.6,0.6,0.6,0.5),
                        label='min-max')
        
        tax = ax.twinx()
        tax.plot(ds_obs.time, ds_obs[var_obs+"_flag"], color=(0.3,0.3,0.3), label='flag')
        
        lh, ll = ax.get_legend_handles_labels()
        ax.legend(lh, ll, loc='upper right', frameon=False)
        ax.set_ylabel(" ".join([var_sim, "(W$\,$m$^{-2}$)"]))
    
    if save_figures:
        os.makedirs(path_plots, exist_ok=True)
        
        plotfile = os.path.join(path_plots, plot_name + ".png")
        f1.savefig(plotfile, dpi=200)
        print(f"Saved {plotfile} ....")
        
    else:
        plt.show()
        pdb.set_trace()
        
    plt.close()


def plot_obs_overview_with_sim_overlying(
    ds_obs: xr.Dataset, 
    ds_sim: xr.Dataset, 
    path_plots: str, 
    plot_name: str,
    save_figures=False):
    
    f1, axs = plt.subplots(4,1, sharex=True, figsize=(10,10))
    
    vars_obs = ["rsds", "rsdifds", "rsdirds", "rlds"]
    vars_sim = ["swdflx", "swdifflx", "swdirflx", "lwdflx"]
    for ax, var_obs, var_sim in zip(axs, vars_obs, vars_sim):
        ax.plot(ds_obs.time, ds_obs[var_obs], label='obs')
        ax.plot(ds_sim.time, ds_sim[var_sim], label='sim')
        
        lh, ll = ax.get_legend_handles_labels()
        ax.legend(lh, ll, loc='upper right', frameon=False)
        ax.set_ylabel(" ".join([var_sim, "(W$\,$m$^{-2}$)"]))
    
    if save_figures:
        os.makedirs(path_plots, exist_ok=True)
        
        plotfile = os.path.join(path_plots, plot_name + ".png")
        f1.savefig(plotfile, dpi=200)
        print(f"Saved {plotfile} ....")
        
    else:
        plt.show()
        pdb.set_trace()
        
    plt.close()


def compute_diffuse_from_direct_and_global_radiation(sw_direct: xr.DataArray, sw_global: xr.DataArray):
    
    return sw_global - sw_direct


if __name__ == "__main__":
    main()