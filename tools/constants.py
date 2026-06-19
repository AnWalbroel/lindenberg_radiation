import numpy as np

R_d = 287.0597  # gas constant of dry air, in J kg-1 K-1
R_v = 461.5     # gas constant of water vapour, in J kg-1 K-1
R_ = 8.314462618    # universal gas constant, in J mol-1 K-1, https://physics.nist.gov/cgi-bin/cuu/Value?r
M_dv = R_d / R_v # molar mass ratio , in ()

m_mol_air = 0.0289647       # molar mass of dry air, in kg mol-1, https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
mw_h2o = 0.01802  # h2o molar mass in kg mol-1
mw_o3 = 0.0479982 # ozone molar mass in kg mol-1
mw_co2 = 0.0440   # co2 molar mass in kg mol-1
mw_n2o = 0.01802  # n2o molar mass in kg mol-1
mw_co = 0.02801   # co molar mass in kg mol-1
mw_ch4 = 0.01604  # ch4 molar mass in kg mol-1
mw_o2 = 0.015999  # o2 molar mass in kg mol-1

e_0 = 611       # saturation water vapour pressure at freezing point (273.15 K), in Pa
T0 = 273.15     # freezing temperature, in K
c_pd = 1005.7   # specific heat capacity of dry air at constant pressure, in J kg-1 K-1
c_vd = 719.0    # specific heat capacity of dry air at constant volume, in J kg-1 K-1
c_h2o = 4187.0  # specific heat capacity of water at 15 deg C; in J kg-1 K-1
L_v = 2.501e+06 # latent heat of vaporization, in J kg-1
sigma_sb = 5.670374419e-08  # Stefan-Boltzmann constant W m-2 K-4, 
                            # https://physics.nist.gov/cgi-bin/cuu/Value?sigma|search_for=stefan

g = 9.80665     # gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )
omega_earth = 2*np.pi / 86164.09    # earth's angular velocity: World Book Encyclopedia Vol 6. Illinois: World Book Inc.: 1984: 12.
R_e = 6371000   # volumetric radius of earth in m, https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

sfac = {'ppm': 1e-06, 'ppb': 1e-09, 'ppt': 1e-12}   # acronyms, scaling factor for ppm, ppb, ...

solar_constant = 1368.      # Solar constant, in W m-2 ; 
                            # some sources suggest 1367 W m-2 (https://doi.org/10.1016/B978-0-08-087872-0.00302-4)