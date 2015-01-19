#!/usr/bin/env python
import numpy as np
import FPSDP.Plasma.GTS_Profile.GTS_Loader as gts
from FPSDP.Geometry.Grid import Cartesian2D, Cartesian3D
#import matplotlib.pyplot as plt
#set and show the parameters

#Note that in XYZ(R-Z-PHI) coordinates, Z is increasing in opposite direction to small toroidal coordinate phi (in r-theta-phi), which means -Z is in increasing direction of toroidal data file numbers.(Those in PHI.000XX) Prepare the whole 32 files if not sure about the range of Z.

#initialize the cartesian grids
grid2D = Cartesian2D(DownLeft = (-0.4,0.3), UpRight = (0.4,1.0), NR = 101, NZ = 101)

gts2D_Alcator = gts.GTS_Loader(grid2D, t0=1,dt=10,nt= 28,'~/workdir2/GTS_ALCATOR_Case/L_Mode/2D_Fluctuations/',eq_fname = '/scratch3/scratchdirs/wangw/Nov2014_1/EQ_C88497_835', prof_fname = '/global/scratch2/sd/shilei/GTS_ALCATOR_Case/L_Mode/profile_run/GTS_backg_profiles.nc' , gts_file_path = '/scratch3/scratchdirs/wangw/Nov2014_1/')







