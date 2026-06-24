import os

# os.environ['DRIVE'] = "/mnt/d/"
# os.environ['PATH_DATA_BASE'] = os.path.join(os.environ['DRIVE'],
#                                             "heavy_data/lindenberg_radiation/")
# os.environ['PATH_PLOTS_BASE'] = os.path.join(os.environ['DRIVE'],
#                                              "Studium_NIM/work/Plots/lindenberg_radiation/")
os.environ['DRIVE'] = "/net/vuthan/awalbroe/"
os.environ['PATH_DATA_BASE'] = os.path.join(os.environ['DRIVE'],
                                            "data/lindenberg_radiation/")
os.environ['PATH_PLOTS_BASE'] = os.path.join(os.environ['DRIVE'],
                                             "plots/lindenberg_radiation/")

path_radiation_sim = os.path.join(os.environ['PATH_DATA_BASE'],
                                  "radiation_simulations/")
path_tcars_data = os.path.join(os.environ['PATH_DATA_BASE'],
                               "tcars_data/")
path_cloudnet_data = os.path.join(os.environ['PATH_DATA_BASE'],
                                 "cloudnet/")
path_radiation_obs = os.path.join(os.environ['PATH_DATA_BASE'],
                                  "radiation/")
path_meteo_obs = os.path.join(os.environ['PATH_DATA_BASE'],
                              "meteo/")
path_hatpro_data = os.path.join(os.environ['PATH_DATA_BASE'],
                                "hatpro/")