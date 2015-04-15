#!/usr/bin/env python
import numpy as np
from FPSDP.Geometry.Grid import Cartesian2D, Cartesian3D
import FPSDP.Plasma.GTS_Profile.GTS_Loader as gts 
#import matplotlib.pyplot as plt
#set and show the parameters

#Note that in XYZ(R-Z-PHI) coordinates, Z is increasing in opposite direction to small toroidal coordinate phi (in r-theta-phi), which means -Z is in increasing direction of toroidal data file numbers.(Those in PHI.000XX) Prepare the whole 32 files if not sure about the range of Z.

#initialize the cartesian grids
grid2D = Cartesian2D(DownLeft = (-0.3,0.5), UpRight = (0.3,0.9), NR = 256, NZ = 256)
grid3D = Cartesian3D(Xmin = 0.5,Xmax = 0.9,Ymin = -0.3,Ymax = 0.3,Zmin = -0.2,Zmax = 0.2,NX = 128,NY = 128,NZ = 16)


fluc_path = '/global/scratch2/sd/shilei/GTS_ALCATOR_Case/L_Mode/2D_Fluctuations/'
eq_fname = '/global/scratch2/sd/shilei/GTS_ALCATOR_Case/L_Mode/RUN_DIR/EQ_C88497_835'
prof_fname = '/global/scratch2/sd/shilei/GTS_ALCATOR_Case/L_Mode/profile_run/GTS_backg_profiles.nc'
gts_file_path = '/global/scratch2/sd/shilei/GTS_ALCATOR_Case/L_Mode/RUN_DIR/'

gts3D_Alcator = gts.GTS_Loader(grid3D, t0=200,dt=10,nt= 1,fluc_file_path = fluc_path , eq_fname = eq_fname, prof_fname = prof_fname , gts_file_path = gts_file_path,n_cross_section = 1)

gts2D_Alcator = gts.GTS_Loader(grid2D, t0=1,dt=10,nt= 28,fluc_file_path = fluc_path , eq_fname = eq_fname, prof_fname = prof_fname , gts_file_path = gts_file_path,n_cross_section = 1)
