"""Create test detectors and carry out the alpha and ECEI calculations
"""
import numpy as np
import FPSDP.ECEI.Alpha1 as a1
import FPSDP.Plasma.TestParameter as tp
import FPSDP.ECEI.Detector as dtc
import FPSDP.ECEI.Intensity as intensity
from FPSDP.Geometry.Grid import path
from FPSDP.GeneralSettings.UnitSystem import cgs

import IDLout 

#define consts
c = cgs['c']
m_e = cgs['m_e']
e = cgs['e']
keV = cgs['keV']

plasma = tp.create_profile()
Z_mid = plasma['Grid'].NZ/2
B_mid = plasma['B'][Z_mid,:]
T_mid = plasma['Te'][Z_mid,:]
f_c_mid = e*B_mid/(m_e*c)/(2*np.pi)

ndtc = 3
NR = plasma['Grid'].NR
R = plasma['Grid'].R2D[0,:]
Z = plasma['Grid'].Z2D[:,0]
pth = path(2,[R[0],R[-1]],[Z[Z_mid],Z[Z_mid]])

measure_locs = [-10,-20,-50,-80,-100]

T_real = np.zeros(len(measure_locs))
f_ctrs = np.zeros(len(measure_locs))
for i in range(len(measure_locs)):
    f_ctrs[i] = 2*f_c_mid[measure_locs[i]]
    T_real[i] = T_mid[measure_locs[i]]
f_flts = f_ctrs[:,np.newaxis]
dtcs = []
for i in range(len(measure_locs)):
    dtcs.append(dtc.detector(f_ctrs[i],1,f_flts[i],[1],pth))

profiles = dtc.create_spatial_frequency_grid(dtcs,plasma)

alphas = []
for i in range(len(profiles)):
    alphas.append( a1.get_alpha_table(profiles[i]) )

Int_array = intensity.get_intensity(dtcs,plasma) 
T_measured = np.array(Int_array[1])
T_diff = T_measured - T_real    

#T_m_2D = intensity.get_2D_intensity(plasma)

#IDLout.IDLoutput(profiles,alphas)