"""Create test detectors and carry out the alpha and ECEI calculations
"""
import numpy as np
import FPSDP.Diagnostic.ECEI.Alpha1 as a1
import FPSDP.Plasma.TestParameter as tp
import FPSDP.Diagnostic.ECEI.Detector as dtc
import FPSDP.Diagnositc.ECEI.Intensity as intensity
from FPSDP.Geometry.Grid import path
from FPSDP.GeneralSettings.UnitSystem import cgs

import IDLout 

# define consts
c = cgs['c']
m_e = cgs['m_e']
e = cgs['e']
keV = cgs['keV']

# create test plasma profiles and get useful quantities along the mid plane
plasma = tp.create_profile()
Z_mid = plasma['Grid'].NZ/2
B_mid = plasma['B'][Z_mid,:]
T_mid = plasma['Te'][Z_mid,:]
f_c_mid = e*B_mid/(m_e*c)/(2*np.pi)

# some coordinate data
NR = plasma['Grid'].NR
R = plasma['Grid'].R2D[0,:]
Z = plasma['Grid'].Z2D[:,0]

# predefined light path 
# the path(N,R[N],Z[N]) function takes 2 arrays specifying the (R,Z) coordinates of each point on the path
# in this case, N=2, which means the path is a straight line with specified start point and end point.  
pth = path(2,[R[0],R[-1]],[Z[Z_mid],Z[Z_mid]])

# set the targeted locations
measure_locs = [-10,-20,-50]

# get the local "real" temperatures and center detector frequencys
T_real = np.zeros(len(measure_locs))
f_ctrs = np.zeros(len(measure_locs))
for i in range(len(measure_locs)):
    f_ctrs[i] = 2*f_c_mid[measure_locs[i]] # targeted at 2nd harmonics
    T_real[i] = T_mid[measure_locs[i]]
# filter frequencies are set to be delta-like 
f_flts = f_ctrs[:,np.newaxis]
dtcs = []
for i in range(len(measure_locs)):
    # each detector is created based on a center frequency, a set of filter information, and a specified light path
    dtcs.append(dtc.detector(f_ctrs[i],1,f_flts[i],[1],pth))

# A specific grid is created based on the chosen detectors, on which alpha will be calculated and the intensity integration 
# will be carried out. 
profiles = dtc.create_spatial_frequency_grid(dtcs,plasma)

alphas = []
for i in range(len(profiles)):
    # alpha is calculated by calling the method "get_alpha_table"
    alphas.append( a1.get_alpha_table(profiles[i]) )

# Intensity measured by each detector can also be obtained by calling the method "get_intensity"
Int_array = intensity.get_intensity(dtcs,plasma) 
# The measured temperature is stored in the second place of the returned tuple of "get_intensity"
T_measured = np.array(Int_array[1])

T_diff = T_measured - T_real 
# For a full 2D comparison, use the get_2D_intensity method    
T_m_2D = intensity.get_2D_intensity(plasma)
# For benchmark with existing IDL code, write data out for IDL reading script
IDLout.IDLoutput(profiles,alphas)
