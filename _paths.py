import os

os.environ['DRIVE'] = "/mnt/f/"
os.environ['PATH_DATA_BASE'] = os.path.join(os.environ['DRIVE'],
                                            "heavy_data/lindenberg_radiation/")
os.environ['PATH_PLOTS_BASE'] = os.path.join(os.environ['DRIVE'],
                                             "Studium_NIM/work/Plots/lindenberg_radiation/")