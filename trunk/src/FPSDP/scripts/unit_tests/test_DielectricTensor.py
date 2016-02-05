# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:21:07 2016

@author: lei

test for DielectricTensor
"""

import numpy as np

import FPSDP.Plasma.DielectricTensor as dt
import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp

#p2d = tp.create_profile2D(True)
tp.set_parameter1D(Te_0=10*tp.cgs['keV'])

p1d = tp.create_profile1D()
p1d.setup_interps()

omega = 8e11
c = 3e10
pi = np.pi

k_para = omega/(c*12)
k_perp = omega/c


X = np.linspace(250,150,201)

#chi_e_cold = dt.SusceptCold(p1d,'e')
#chie_c = chi_e_cold([X], omega, k_para, k_perp)

#chi_e_warm = dt.SusceptWarm(p1d,'e')
#chie_w = chi_e_warm([X], omega, k_para, k_perp)


#chi_e_hot = dt.SusceptNonrelativistic(p1d, 'e', max_harmonic=3)
#chie_h = chi_e_hot([X], omega, k_para, k_perp)

# chi_e_rel = dt.SusceptRelativistic(p1d, 'e', max_harmonic=2, max_power=2)
# chie_r = chi_e_rel([X], omega, k_para, k_perp)



