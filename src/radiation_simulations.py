import os
import pdb

import numpy as np
import xarray as xr

import _paths
from readers.read_cloudnet import (read_cloudnet_categorize_model_data,
                                   read_cloudnet_microphysics_retrievals_data)
from tools.plot_tools import get_cm_cmap, change_colormap_len, create_colourbar

def main():
    
    """
    Radiation simulations using pyRRTMG (T-CARS ver).
    """
    
    path_output = os.path.join(os.environ['PATH_DATA_BASE'],
                               "radiation_simulations/")
    path_plots = os.path.join(os.environ['PATH_PLOTS_BASE'],
                              "data_overview/")
    
    date_str = "2025-10-01"
    height_grid = np.arange(0., 25000., 10.)
    date = np.datetime64(date_str)
    data_quicklooks = False
    set_dict = {'save_figures': False,
                'date_str': date_str}
    
    cn_model_ds = read_cloudnet_categorize_model_data(date0=date)
    cn_mp_ds = read_cloudnet_microphysics_retrievals_data(date0=date)
    
    if data_quicklooks: data_overview_quicklooks(path_plots, cn_model_ds, cn_mp_ds, **set_dict)
    pdb.set_trace()
    
    # plot overview of model and mp data (incl quality bits and status flags)
    
    # load cloudnet data, which also contains model output of thermodyn profiles
    # set up t-cars
    # cloud sanity enforcer, check logic in loop
    # simplify load_background_vmrs if elif
    # Könntest du da die 2m Lufttemperatur einbeziehen? Als surface (skin) temperature könnte man einfach aus LW up mit gewählter Emissivitität (0.998? oder was da standardmäßig angenommen wird) berechnen. Könntest du den IWV vom HATPRO nehmen um das Modelprofil damit zu skalieren?


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
    
    cn_model_ds = cn_model_ds.load()
    cn_mp_ds = cn_mp_ds.load()
    
    date_str_short = date_str.replace("-", "")
    
    vars_model = ['temp', 'q', 'u', 'v']
    cmaps_model = ['batlow', 'oslo_r', 'vik', 'vik']
    vmins_model = [200, 0.0, -40, -40]
    vmaxs_model = [290, 0.006, 40, 40]
    vars_mp = ['der', 'ier', 'iwc', 'lwc']
    cmaps_mp = ['batlow', 'batlow', 'batlow', 'batlow']
    vmins_mp = [None, None, None, None]
    vmaxs_mp = [None, None, None, None]
    vars_mp_err = ['der_error', 'ier_error', 'iwc_error', 'lwc_error']
    cmaps_mp_err = ['batlow', 'batlow', 'batlow', 'batlow']
    vmins_mp_err = [None, None, None, None]
    vmaxs_mp_err = [None, None, None, None]
    vars_mp_ret = ['der_retrieval_status', 'ier_retrieval_status', 'iwc_retrieval_status', 'lwc_retrieval_status']
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