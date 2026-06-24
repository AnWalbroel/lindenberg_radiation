import tools.constants as constants
import numpy as np


def compute_IWV_q(
    q,
    press,
    nan_threshold=0.0,
    scheme='balanced'):

    """
    Compute Integrated Water Vapour (also known as precipitable water content)
    out of specific humidity (in kg kg^-1), gravitational constant and air pressure (in Pa).
    The moisture data may contain certain number gaps (up to nan_threshold*n_levels) but
    the height variable must be free of gaps.

    Parameters:
    -----------
    q : array of floats
        One dimensional array of specific humidity in kg kg^-1.
    press : array of floats
        One dimensional array of pressure in Pa.
    nan_threshold : float, optional
        Threshold describing the fraction of nan values of the total height level
        number that is still permitted for computation.
    scheme : str, optional
        Chose the scheme 'balanced' or 'top_weighted'. They differ in the way the altitude
        levels are used to compute IWV. Recommendation and default: 'balanced'
    """

    IWV = np.nan

    # Check if the Pressure axis is sorted in descending order:
    if np.any(np.diff(press) > 0):
        print("Warning! Height axis must be in ascending order (pressure in descending) to compute the integrated" +
            " water vapour.")

        # if the pressure data is okay until 300 hPa, compute IWV nonetheless and truncate the
        # profile beyond:
        where_broken = np.where(np.diff(press) > 0)[0]      # when where_broken == 152, then press[153] - press[152] is broken
        if press[where_broken[0]] > 30000.0:    # then, sufficient altitude doesn't have valid data valid data, return IWV=nan
            return IWV

    # truncate data to non nan height or pressure levels:
    non_nan_idx = np.where(~np.isnan(press))[0]
    q = q[non_nan_idx[0]:non_nan_idx[-1]+1]
    press = press[non_nan_idx[0]:non_nan_idx[-1]+1]

    # check if height axis is free of gaps:
    if np.any(np.isnan(np.diff(press))): 
        print("Height axis contains gaps. Aborted IWV computation.")
        return IWV


    n_height = len(press)
    # Check if q has got any gaps:
    n_nans = np.count_nonzero(np.isnan(q))


    # If no nans exist, the computation is simpler. If some nans exist IWV will still be
    # computed but needs to look for the next non-nan value. If too many nans exist IWV
    # won't be computed.
    if scheme == 'balanced':
        if (n_nans == 0):

            IWV = 0.0
            for k in range(n_height):
                if k == 0:      # bottom of grid
                    dp = 0.5*(press[k+1] - press[k])        # just a half of a level difference

                elif k == n_height-1:   # top of grid
                    dp = 0.5*(press[k] - press[k-1])        # the other half level difference

                else:           # mid of grid
                    dp = 0.5*(press[k+1] - press[k-1])

                IWV = IWV - q[k]*dp

        elif n_nans / n_height < nan_threshold:

            # Loop through height grid:
            IWV = 0.0
            k = 0
            prev_nonnan_idx = -1
            while k < n_height:

                # check if hum on current level is nan:
                # if so search for the next non-nan level:
                if np.isnan(q[k]):
                    next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

                    if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):   # mid or near top of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of rho_v
                        IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k-1])
                    
                    elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):  # bottom of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of q

                        # fixing height grid variable in case only the lowest measurement doesn't exist:
                        if np.isnan(press[0]) and not (np.isnan(press[1]+press[2])):
                            IWV -= 0.5*q[next_nonnan_idx]*(press[2] - press[1])
                        else:
                            IWV -= 0.5*q[next_nonnan_idx]*(press[k+1] - press[k])

                    else: # reached top of grid
                        IWV += 0.0

                else:
                    prev_nonnan_idx = k

                    if k == 0:          # bottom of grid
                        IWV -= 0.5*q[k]*(press[k+1] - press[k])
                    elif k == 1 and np.isnan(press[k-1]):       # next to bottom of grid
                        IWV -= 0.5*q[k]*(press[k+1] - press[k])
                    elif (k > 0) and (k < n_height-1):  # mid of grid
                        IWV -= 0.5*q[k]*(press[k+1] - press[k-1])
                    else:               # top of grid
                        IWV -= 0.5*q[k]*(press[-1] - press[-2])

                k += 1

        else:
            IWV = np.nan


    elif scheme == 'top_weighted':
        if (n_nans == 0):

            IWV = 0.0
            for k in range(n_height):
                if k < n_height-2:      # bottom or mid of grid
                    dp = press[k+1] - press[k]

                else:   # top and next to top of grid
                    dp = 0.5*(press[-1] - press[-2])        # half the height for top two levels

                IWV = IWV - q[k]*dp

        elif n_nans / n_height < nan_threshold:

            # Loop through height grid:
            IWV = 0.0
            k = 0
            prev_nonnan_idx = -1
            while k < n_height:
                
                # check if hum on current level is nan:
                # if so search for the next non-nan level:
                if np.isnan(q[k]):
                    next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

                    if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):   # mid of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of q

                        if k+1 != n_height-1:
                            IWV -= 0.5*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
                        else:   # near top of grid
                            IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
                    
                    elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):  # bottom of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of q

                        # fixing height grid variable in case only the lowest measurement doesn't exist:
                        if np.isnan(press[0]) and not (np.isnan(press[1]+press[2])):
                            IWV -= q[next_nonnan_idx]*(press[2] - press[1])
                        else:
                            IWV -= q[next_nonnan_idx]*(press[k+1] - press[k])
                        

                    else: # reached top of grid
                        IWV += 0.0

                else:
                    prev_nonnan_idx = k

                    if k < n_height-2:  # bottom or mid of grid
                        IWV -= q[k]*(press[k+1] - press[k])
                    else:               # top of grid
                        IWV -= 0.5*q[k]*(press[-1] - press[-2])

                k += 1

        else:
            IWV = np.nan


    IWV = IWV / constants.g       # yet had to be divided by gravitational acceleration

    return IWV


def e_sat(
    temp,
    which_algo='hyland_and_wexler'):

    """
    Calculates the saturation pressure over water after Goff and Gratch (1946)
    or Hyland and Wexler (1983).
    Source: Smithsonian Tables 1984, after Goff and Gratch 1946
    http://cires.colorado.edu/~voemel/vp.html
    http://hurri.kean.edu/~yoh/calculations/satvap/satvap.html

    e_sat_gg_water in Pa.

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    which_algo : str
        Specify which algorithm is chosen to compute e_sat (in Pa). Options:
        'hyland_and_wexler' (default), 'goff_and_gratch'
    """

    if which_algo == 'hyland_and_wexler':
        e_sat_gg_water = temp**(0.65459673e+01) * np.exp(-0.58002206e+04 / temp + 0.13914993e+01 - 0.48640239e-01*temp + 
                                0.41764768e-04*(temp**2) - 0.14452093e-07*(temp**3))

    elif which_algo == 'goff_and_gratch':
        e_sat_gg_water = 100 * 1013.246 * 10**(-7.90298*(373.16/temp-1) + 5.02808*np.log10(
                373.16/temp) - 1.3816e-7*(10**(11.344*(1-temp/373.16))-1) + 8.1328e-3 * (10**(-3.49149*(373.16/temp-1))-1))

    return e_sat_gg_water


def convert_rh_to_abshum(
    temp,
    relhum):

    """
    Convert array of relative humidity (between 0 and 1) to absolute humidity
    in kg m^-3. 

    Saturation water vapour pressure computation is based on: see e_sat(temp).

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    relhum : array of floats
        Array of relative humidity (between 0 and 1).
    """

    e_sat_water = e_sat(temp)

    rho_v = relhum * e_sat_water / (constants.R_v * temp)

    return rho_v


def convert_rh_to_spechum(
    temp,
    pres,
    relhum):

    """
    Convert array of relative humidity (between 0 and 1) to specific humidity
    in kg kg^-1.

    Saturation water vapour pressure computation is based on: see e_sat(temp).

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    pres : array of floats
        Array of pressure (in Pa).
    relhum : array of floats
        Array of relative humidity (between 0 and 1).
    """

    e_sat_water = e_sat(temp)

    e = e_sat_water * relhum
    q = constants.M_dv * e / (e*(constants.M_dv - 1) + pres)

    return q
    
    
def convert_abshum_to_spechum(
    temp,
    pres,
    abshum):

    """
    Convert array of absolute humidity (kg m^-3) to specific humidity
    in kg kg^-1.

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    pres : array of floats
        Array of pressure (in Pa).
    abshum : array of floats
        Array of absolute humidity (in kg m^-3).
    """

    q = abshum / (abshum*(1 - 1/constants.M_dv) + (pres/(constants.R_d*temp)))

    return q


def q_to_h2ovmr(q: np.ndarray):
    
    """
    Converts specific humidity q (in kg kg-1) to volume mixing ratio (unitless).
    
    Parameters:
    -----------
    q : np.ndarray or xr.DataArray
        Specific humidity in kg kg-1.
    """
    
    return constants.m_mol_air*q / (constants.mw_h2o*(1.0 - q))


def h2ovmr_to_q(h2ovmr: np.ndarray):
    
    """
    Converts water vapour volume mixing ratio (unitless) to specific humidity q (in kg kg-1).
    
    Parameters:
    -----------
    h2ovmr : np.ndarray or xr.DataArray
        Water vapour volume mixing ratio.
    """
    
    return h2ovmr / ((constants.m_mol_air/constants.mw_h2o) + h2ovmr)


def rho_air(
    pres,
    temp,
    abshum):

    """
    Compute the density of air (in kg m-3) with a certain moisture load.

    Parameters:
    -----------
    pres : array of floats
        Array of pressure (in Pa).
    temp : array of floats
        Array of temperature (in K).
    abshum : array of floats
        Array of absolute humidity (in kg m^-3).
    """

    rho = (pres - abshum*constants.R_v*temp) / (constants.R_d*temp) + abshum

    return rho


def convert_spechum_to_abshum(
    temp,
    pres,
    q):

    """
    Convert array of specific humidity (kg kg^-1) to absolute humidity
    in kg m^-3.

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    pres : array of floats
        Array of pressure (in Pa).
    q : array of floats
        Array of specific humidity (in kg kg^-1).
    """

    abshum = pres / (constants.R_d*temp*(1/q + 1/constants.M_dv - 1))

    return abshum


def compute_heating_rate(
    upward_flux: np.ndarray,
    downward_flux: np.ndarray,
    rho: np.ndarray,
    height_lev: np.ndarray,
    convert_to_K_day=False):
    
    """
    Compute heating rates according to Petty (2006) chapter 10.4.1, equation 10.54 from
    upward and downward radiation fluxes (shortwave and longwave possible). Note that
    the radiation fluxes must be given on height levels while air density must be provided 
    on a height layer grid (whose boundaries are the height levels) because the heating rates
    will be put onto the height layer grid.
    
    Parameters:
    -----------
    upward_flux : np.ndarray or xr.DataArray
        Upward shortwave or longwave radiation flux in W m-2.
    downward_flux : np.ndarray or xr.DataArray
        Downward shortwave or longwave radiation flux in W m-2.
    rho : np.ndarray or xr.DataArray
        Air density (dry air + absolute humidity) in kg m-3.
    height_lev : np.ndarray or xr.DataArray
        Height levels (boundaries of height layers) in m.
    convert_to_K_day : bool
        If True, heating rates will be given in K day-1. Otherwise, in K s-1.
    """
    
    F_net = upward_flux - downward_flux
    dF_net_dz = np.diff(F_net, axis=-1) / np.diff(height_lev, axis=-1)
    HR = - dF_net_dz / (constants.c_pd * rho)       # heating rate in K s-1
    if convert_to_K_day: HR *= 86400.               # heating rate in K day-1
    
    return HR