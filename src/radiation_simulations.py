import os

import numpy as np

import _paths

def main():
    
    """
    Radiation simulations using pyRRTMG (T-CARS ver).
    """
    
    path_output = os.path.join(os.environ['PATH_DATA_BASE'],
                               "radiation_simulations/")
    
    date_str = "2025-10-01"
    date = np.datetime64(date_str)
    
    # load cloudnet data, which also contains model output of thermodyn profiles
    # set up t-cars


if __name__ == "__main__":
    main()