import scipy.interpolate as ip
import matplotlib.pyplot as mp
import numpy as np
from beam import *
import math

class GRID:
    def __init__(self,X=(1.3,1.6), Y=(-0.3,0.5), Z=(-0.5,0.5),NX=64,NY=64,NZ=32):
        self.Xmin = X[0]
        self.Ymin = Y[0]
        self.Zmin = Z[0]

        self.Xmax = X[1]
        self.Ymax = Y[1]
        self.Zmax = Z[1]

        self.X3D = np.linspace(X[0],X[1],NX)
        self.Y3D = np.linspace(Y[0],Y[1],NY)
        self.Z3D = np.linspace(Z[0],Z[1],NZ)
        


class XGC:
    def __init__(self,R=(1.3,1.6), Z=(-0.5,0.5), PHI=(0,2*math.pi),NR=64,NZ=64,NPHI=32):
        self.grid = GRID()
        x,y,z = np.meshgrid(self.grid.X3D,self.grid.Y3D,self.grid.Z3D)
        xav = (self.grid.Xmax + self.grid.Xmin)/2.0
        yav = (self.grid.Ymax + self.grid.Ymin)/2.0
        self.ne_on_grid = 1e19*np.exp(-(x-xav)**2/0.001 - (y-yav)**2/0.001 - z**2/0.01)
        self.ni_on_grid = 1e19*np.exp(-(x-xav)**2/0.002 - (y-yav)**2/0.003 - z**2/0.005)
        self.ti_on_grid = 1e4*np.exp(-(x-xav)**2/0.0001 - (y-yav)**2/0.0002 - z**2/0.01)
        self.te_on_grid = 1e4*np.exp(-(x-xav)**2/0.002 - (y-yav)**2/0.002 - z**2/0.004)

config_file = "beam.in"
xgc = XGC()

b1d = Beam1D(config_file,1,xgc)
dl = np.sqrt(np.sum((b1d.get_mesh()-b1d.get_origin())**2,axis = 1))

mp.plot(dl,b1d.density_beam[0,:],'kx')
mp.plot(dl,b1d.density_beam[1,:],'bx')
mp.plot(dl,b1d.density_beam[2,:],'rx')

mp.show()
"""
#!/usr/bin/env python
from FPSDP.Plasma.XGC_Profile.load_XGC_profile import *
from FPSDP.Geometry.Grid import *
import matplotlib as mp
mp.use('agg')
import matplotlib.pyplot as plt

xgc_path = '/global/project/projectdirs/m499/jlang/particle_pinch/'

grid2D = Cartesian2D(DownLeft = (-0.5,1.3),UpRight = (0.5,1.6),NR = 256, NZ = 512)

#grid3D = Grid.Cartesian3D(Xmin = 1.3,Xmax = 1.6,Ymin = -0.5, Ymax = 0.5, Zmin = -0.3, Zmax = 0.3, NX = 256,NY = 512,NZ = 80)

time_start = 1
time_end = 1
time_step = 1

def load(dimension,full_load,fluc_only):
    if dimension == 3:
        xgc = XGC_Loader(xgc_path,grid3D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 1, Full_Load = full_load, Fluc_Only = fluc_only)
    elif dimension == 2:
        xgc = XGC_Loader(xgc_path,grid2D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 1, Full_Load = full_load, Fluc_Only = fluc_only)

    return xgc    

xgc = load(2,True,False)
print xgc.ne_on_grid.shape

print max(xgc.ne_on_grid[0][0][:][:].max(axis=1))
plt.figure()
plt.contourf(xgc.ne_on_grid[0][0][:][:])
#plt.savefig('foo.pdf')
plt.show(True)
"""
