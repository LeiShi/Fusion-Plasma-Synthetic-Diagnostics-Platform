#!/usr/bin/env python
import numpy as np
import FPSDP.Plasma.GTS_Profile.GTS_Loader as gts
from FPSDP.Geometry.Grid import Cartesian2D, Cartesian3D
#import matplotlib.pyplot as plt
#set and show the parameters

#Note that in XYZ(R-Z-PHI) coordinates, Z is increasing in opposite direction to small toroidal coordinate phi (in r-theta-phi), which means -Z is in increasing direction of toroidal data file numbers.(Those in PHI.000XX) Prepare the whole 32 files if not sure about the range of Z.



#initialize arrays for returning data
ne = np.empty((cm.NT0,cm.NZ0,cm.NY0,cm.NX0))
Te = np.empty((cm.NZ0,cm.NY0,cm.NX0))
Bm = np.empty((cm.NZ0,cm.NY0,cm.NX0))

#show x3d,y3d,z3d
print(mesh.X3D)
print(mesh.Y3D)
print(mesh.Z3D)

#get the fluctuations

cm.get_fluctuations_from_GTS(mesh.X3D,mesh.Y3D,mesh.Z3D,ne,Te,Bm)

import matplotlib.pyplot as plt

print 'NY0 = '+str(cm.NY0)
print ne.shape

plt.plot(mesh.X1D, ne[0,0,cm.NY0/2,:])

#plt.plot(x1d, ne[9,0,cm.NY0/2,:])

plt.figure(2)

plt.imshow(ne[0,0,:,:])

#cm.get_fluctuations_from_GTS(x3d,y3d,z3d,ne,Te,Bm)


