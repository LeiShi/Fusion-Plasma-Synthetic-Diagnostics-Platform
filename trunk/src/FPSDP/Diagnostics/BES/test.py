import scipy.interpolate as ip
import matplotlib.pyplot as mp
import numpy as np
import math


import FPSDP.Diagnostics.Beam.beam as be
import FPSDP.Diagnostics.BES.bes as bes
import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid
import scipy.io as sio

xgc_path = '/global/project/projectdirs/m499/jlang/particle_pinch/'

grid3D = Grid.Cartesian3D(Xmin = 1.4,Xmax = 2.1,Ymin = -0.2, Ymax = 0.2, Zmin = -0.5, Zmax = 0.5, NX = 64,NY =16,NZ = 64)
#grid3D = Grid.Cartesian3D(Xmin = 1.4,Xmax = 2.1,Ymin = -0.2, Ymax = 0.2, Zmin = -0.5, Zmax = 0.5, NX = 256,NY = 32,NZ = 256)


time_start = 180
time_end = 180
time_step = 20
time_ = [180]#, 120, 140, 160, 180]

bes_ = bes.BES('FPSDP/Diagnostics/BES/bes.in')
xgc_ = bes_.beam.data

#def load(full_load,fluc_only):
#    xgc_ = xgc.XGC_Loader(xgc_path,grid3D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 1, Equilibrium_Only = False,Full_Load = full_load, Fluc_Only = fluc_only, load_ions=True, equilibrium_mesh = '3D')

#    return xgc_

#xgc_ = load(True,False)


"""class GRID:
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
"""

b1d1 = bes_.beam
dl1 = np.sqrt(np.sum((b1d1.get_mesh()-b1d1.get_origin())**2,axis = 1))
print bes_.intensity([0],0)

mp.figure()
ms = b1d1.get_mesh()
foc = bes_.fib_pos[0,:] + bes_.focal*bes_.op_direc
mp.plot(ms[:,0],ms[:,1])
mp.plot(foc[0],foc[1],'o',markersize=4)
mp.show()

"""radius = 0.008
z = np.linspace(0,0.4,100)
r = np.linspace(-radius,radius,20)
direct =  np.array([-0.2,0.7,0.0])
ori = np.array([2.0,-0.49,0.0])
perp = np.array([1,-direct[0]/direct[1],0])
perp = perp/sum(perp**2)
R,Z = np.meshgrid(r,z)
R_ = np.reshape(R,-1)
Z_ = np.reshape(Z,-1)
pos = np.zeros((3,len(R_)))
pos[0,:] = ori[0] + direct[0]*Z_ + R_*perp[0]
pos[1,:] = ori[1] + direct[1]*Z_ + R_*perp[1]
#mp.contourf(xgc_.grid.X3D[:,0,:].T,xgc_.grid.Z3D[:,0,:].T,xgc_.ne_on_grid[0,0,:,0,:].T)
#mp.plot(pos[0,:],pos[1,:])
pos = pos.T

emis1 = b1d1.get_emis_lifetime(pos,[0, 1])
emis_test = b1d1.get_emis(pos,[0])
emis = emis1[0,:,:]-emis1[1,:,:]
diff = emis_test - emis1[0,:,:]


mp.figure()
mp.title('ne')
mp.plot(dl1,b1d1.dens[0,:])

mp.figure()
mp.title('Ti')
mp.plot(dl1,b1d1.ti[0,:])

mp.figure()
mp.title('ne*S')
mp.plot(dl1,b1d1.dens[0,:]*b1d1.S[0,:])

mp.figure()
mp.title('S_cr')
mp.plot(dl1,b1d1.S[0,:])

mp.figure()
mp.title('nb')
mp.plot(dl1,b1d1.density_beam[0,0,:])

mp.figure()
mp.title('epsilon')
test = b1d1.get_emis(b1d1.mesh,[0])
mp.plot(dl1[0:-2],test[0,0,0:-2])

mp.figure()
mp.title('epsilon lifetime')
mp.plot(dl1,b1d1.get_emis_lifetime(b1d1.mesh,[0])[0,0,:])
mp.plot(dl1,b1d1.get_emis_lifetime(b1d1.mesh,[0])[0,1,:])
mp.plot(dl1,b1d1.get_emis_lifetime(b1d1.mesh,[0])[0,2,:])

mp.show()

"""
