# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:21:07 2016

@author: lei

test for DielectricTensor
"""

import numpy as np

import FPSDP.Plasma.DielectricTensor as dt
import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp

p = tp.create_profile2D(True)

chi_e = dt.SusceptCold(p,'e')

omega = np.array(400e9)
Z2D = p.grid.Z2D
R2D = p.grid.R2D
chie = chi_e(omega, [Z2D,R2D])

