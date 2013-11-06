#!/usr/bin/env python
import numpy as np
import Coord_Mapper as cm
import test_C as tm 
#set and show the parameters
cm.set_para(NT=1,TStart=0,TStep=1,Xmin=0.5,Xmax=2.5,NX=5,NY=5)

#initialize the cartesian grids
x1d = np.linspace(cm.Xmin0,cm.Xmax0,cm.NX0)
y1d = np.linspace(cm.Ymin0,cm.Ymax0,cm.NY0)
z1d = np.linspace(cm.Zmin0,cm.Zmax0,cm.NZ0)
x3d = np.zeros((cm.NZ0,cm.NY0,cm.NX0))
y3d = np.zeros((cm.NZ0,cm.NY0,cm.NX0))
z3d = np.zeros((cm.NZ0,cm.NY0,cm.NX0))
x3d += x1d[np.newaxis, np.newaxis, :]
y3d += y1d[np.newaxis, :, np.newaxis]
z3d += z1d[:, np.newaxis, np.newaxis]

#initialize arrays for returning data
ne = np.empty((cm.NT0,cm.NZ0,cm.NY0,cm.NX0))
Te = np.empty((cm.NZ0,cm.NY0,cm.NX0))
Bm = np.empty((cm.NZ0,cm.NY0,cm.NX0))

#show x3d,y3d,z3d
print(x3d)
print(y3d)
print(z3d)

#get the fluctuations
sum1=tm.arraysum(x3d[0,:,:])
print(sum1)

sum=cm.mmc.arraysum(x3d[0,:,:])
print(sum)
#cm.mmc.get_GTS_profiles_(x3d,y3d,z3d,ne,Te,Bm)
#cm.get_fluctuations_from_GTS(x3d,y3d,z3d,ne,Te,Bm)


