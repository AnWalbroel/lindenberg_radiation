import os

os.environ['DRIVE'] = "/mnt/f/"
os.environ['PATH_DATA_BASE'] = os.path.join(os.environ['DRIVE'],
                                            "heavy_data/lindenberg_radiation/")
os.environ['PATH_PLOTS_BASE'] = os.path.join(os.environ['DRIVE'],
                                             "Studium_NIM/work/Plots/lindenberg_radiation/")

path_radiation_sim = os.path.join(os.environ['PATH_DATA_BASE'],
                                  "radiation_simulations/")
path_tcars_data = os.path.join(os.environ['PATH_DATA_BASE'],
                               "tcars_data/")
path_cloudnet_data = os.path.join(os.environ['PATH_DATA_BASE'],
                                 "cloudnet/")