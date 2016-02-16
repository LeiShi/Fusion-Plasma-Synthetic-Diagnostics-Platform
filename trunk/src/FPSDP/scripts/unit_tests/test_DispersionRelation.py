# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:01:14 2016

@author: lei

test FPSDP.Models.Waves.DispersionRelation
"""

import numpy as np

import FPSDP.Models.Waves.DispersionRelation as dr
import FPSDP.Plasma.DielectricTensor as dt
import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp
from FPSDP.GeneralSettings.UnitSystem import cgs

c = cgs['c']
m_e = cgs['m_e']
e = cgs['e']
keV = cgs['keV']
pi = np.pi

tp.set_parameter1D(ne_shape='uniform', ne_0=6.0e13, Te_shape='uniform', 
                   Te_0=15*keV, B_0=3.0e4)
                   
p1d = tp.create_profile1D()
p1d.setup_interps()
c_dielect = dt.ColdElectronColdIon(p1d)
r_dielect_1 = dt.RelElectronColdIon(p1d, max_harmonic=1, max_power=5)
r_dielect_2 = dt.RelElectronColdIon(p1d, max_harmonic=2, max_power=5)
r_dielect_3 = dt.RelElectronColdIon(p1d, max_harmonic=3, max_power=5)
r_dielect_4 = dt.RelElectronColdIon(p1d, max_harmonic=4, max_power=5)
r_dielect_5 = dt.RelElectronColdIon(p1d, max_harmonic=5, max_power=5)
r_dielect_1_1 = dt.RelElectronColdIon(p1d, max_harmonic=1, max_power=1)


omega = 124e9 * 2*pi
k_para = 0
n_perp_r = np.linspace(-0.1, 1, 100)
n_perp_i = np.linspace(-0.3, 0.3, 50)
k_perp_r = omega/c*n_perp_r
k_perp_i = omega/c*n_perp_i

k_perp = k_perp_r[np.newaxis, :] + 1j*k_perp_i[:, np.newaxis]

coordinates = [tp.Parameter1D['R_0']]

c_Lambd = dr.Lambda(omega, k_para, k_perp, c_dielect, coordinates)
r_Lambd1 = dr.Lambda(omega, k_para, k_perp, r_dielect_1, coordinates)
r_Lambd2 = dr.Lambda(omega, k_para, k_perp, r_dielect_2, coordinates)
r_Lambd3 = dr.Lambda(omega, k_para, k_perp, r_dielect_3, coordinates)
r_Lambd4 = dr.Lambda(omega, k_para, k_perp, r_dielect_4, coordinates)
r_Lambd5 = dr.Lambda(omega, k_para, k_perp, r_dielect_5, coordinates)
r_Lambd11 = dr.Lambda(omega, k_para, k_perp, r_dielect_1_1, coordinates)
