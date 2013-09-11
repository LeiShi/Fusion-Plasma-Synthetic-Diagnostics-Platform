#!/usr/bin/env python
import numpy as np
from Coord_Mapper import *

#set and show the parameters
set_para(NT=1,TStart=0,TStep=1)
show_para()

#initialize the cartesian grids
x1d = np.linspace(Xmin0,Xmax0,NX0)
y1d = np.linspace(Ymin0,Ymax0,NY0)
z1d = np.linspace(Zmin0,Zmax0,NZ0)
x3d = np.zeros((NZ,NY,NZ))
y3d = np.zeros((NZ,NY,NZ))
z3d = np.zeros((NZ,NY,NZ))
x3d += x1d[np.newaxis, np.newaxis, :]
y3d += y1d[np.newaxis, :, np.newaxis]
z3d += z1d[:, np.newaxis, np.newaxis]

#initialize arrays for returning data
ne = np.empty((NT0,NZ0,NY0,NX0))
Te = np.empty((NZ0,NY0,NX0))
Bm = np.empty((NZ0,NY0,NX0))

#get the fluctuations
get_fluctuations_from_GTS(x3d,y3d,z3d,ne,Te,Bm)


