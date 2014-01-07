#!/usr/bin/env python
import numpy as np
import Coord_Mapper as cm
import test_C as tm 
#import matplotlib.pyplot as plt
#set and show the parameters
cm.set_para(NT=1,TStart=700,TStep=1,Xmin=2,Xmax=2.5,NX=101,NY=101)

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

cm.mmc.get_GTS_profiles_(x3d,y3d,z3d,ne,Te,Bm)

import matplotlib.pyplot as plt

print 'NY0 = '+str(cm.NY0)
print ne.shape

plt.plot(x1d, ne[0,0,cm.NY0/2,:])

#cm.get_fluctuations_from_GTS(x3d,y3d,z3d,ne,Te,Bm)


