#!/usr/bin/env python
import numpy as np
import FPSDP.Plasma.GTS_Profile.GTS_Loader as gts
from FPSDP.Geometry.Grid import Cartesian2D, Cartesian3D
#import matplotlib.pyplot as plt
#set and show the parameters

#Note that in XYZ(R-Z-PHI) coordinates, Z is increasing in opposite direction to small toroidal coordinate phi (in r-theta-phi), which means -Z is in increasing direction of toroidal data file numbers.(Those in PHI.000XX) Prepare the whole 32 files if not sure about the range of Z.

#initialize the cartesian grids
grid2D = Cartesian2D(DownLeft = (-0.3,0.5), UpRight = (0.3,0.9), NR = 256, NZ = 256)

gts2D_Alcator = gts.GTS_Loader(grid2D, 1,10,28,fluc_file_path = '~/workdir2/GTS_ALCATOR_Case/L_Mode/2D_Fluctuations/',eq_fname = '/scratch3/scratchdirs/wangw/Nov2014_1/EQ_C88497_835', prof_fname = '/global/scratch2/sd/shilei/GTS_ALCATOR_Case/L_Mode/profile_run/GTS_backg_profiles.nc' , gts_file_path = '/scratch3/scratchdirs/wangw/Nov2014_1/')







