import scipy.interpolate as ip
import matplotlib.pyplot as mp
import numpy as np
from beam import *
from ADAS_file import *
import math

import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid
import scipy.io as sio

xgc_path = '/global/project/projectdirs/m499/jlang/particle_pinch/'

grid3D = Grid.Cartesian3D(Xmin = 1.4,Xmax = 2.1,Ymin = -0.2, Ymax = 0.2, Zmin = -0.5, Zmax = 0.5, NX = 128,NY = 32,NZ = 64)


time_start = 140
time_end = 180
time_step = 20
time_ = [140, 160, 180]#, 120, 140, 160, 180]

def load(full_load,fluc_only):
    xgc_ = xgc.XGC_Loader(xgc_path,grid3D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 1, Equilibrium_Only = False,Full_Load = full_load, Fluc_Only = fluc_only, load_ions=True, equilibrium_mesh = '3D')

    return xgc_

xgc_ = load(True,False)


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

config_file1 = "FPSDP/Diagnostics/BES/beam.in"
#config_file2 = "FPSDP/Diagnostics/BES/beam1.in"
#config_file3 = "FPSDP/Diagnostics/BES/beam2.in"
#config_file4 = "FPSDP/Diagnostics/BES/beam3.in"
#config_file5 = "FPSDP/Diagnostics/BES/beam4.in"

b1d1 = Beam1D(config_file1,range(len(time_)),xgc_)
#dl1 = np.sqrt(np.sum((b1d1.get_mesh()-b1d1.get_origin())**2,axis = 0))
#b1d2 = Beam1D(config_file2,range(len(time_)),xgc_)
#dl2 = np.sqrt(np.sum((b1d2.get_mesh()-b1d2.get_origin())**2,axis = 1))
#b1d3 = Beam1D(config_file3,range(len(time_)),xgc_)
#dl3 = np.sqrt(np.sum((b1d3.get_mesh()-b1d3.get_origin())**2,axis = 1))
#b1d4 = Beam1D(config_file4,range(len(time_)),xgc_)
#dl4 = np.sqrt(np.sum((b1d4.get_mesh()-b1d4.get_origin())**2,axis = 1))
#b1d5 = Beam1D(config_file5,range(len(time_)),xgc_)
#dl5 = np.sqrt(np.sum((b1d5.get_mesh()-b1d5.get_origin())**2,axis = 1))

#print b1d.density_beam[0,0,:]
mp.contourf(xgc_.grid.X3D[:,0,:].T,xgc_.grid.Z3D[:,0,:].T,xgc_.ne_on_grid[0,0,:,0,:].T)
#mp.plot(b1d1.get_mesh()[0,:],b1d1.get_mesh()[1,:],'kx')
#mp.plot(b1d2.get_mesh()[:,1],b1d2.get_mesh()[:,0],'mx')
#mp.plot(b1d3.get_mesh()[:,1],b1d3.get_mesh()[:,0],'rx')
#mp.plot(b1d4.get_mesh()[:,1],b1d4.get_mesh()[:,0],'bx')
#mp.plot(b1d5.get_mesh()[:,1],b1d5.get_mesh()[:,0],'gx')
#mp.colorbar()

#mp.figure()
#z = np.reshape(xgc_.grid.Z3D,-1)
#x = np.reshape(xgc_.grid.X3D,-1)
#y = np.reshape(xgc_.grid.Y3D,-1)
radius = 0.03
z = np.linspace(0.001,1.4,500)
r = np.linspace(-radius,radius,40)
direct =  np.array([-0.2,0.7,0.0])
ori = np.array([1.99,-0.49,0.0])
perp = np.array([1,-direct[0]/direct[1],0])
perp = perp/sum(perp**2)
R,Z = np.meshgrid(r,z)
R_ = np.reshape(R,-1)
Z_ = np.reshape(Z,-1)
pos = np.zeros((3,len(R_)))
pos[0,:] = ori[0] + direct[0]*Z_ + R_*perp[0]
pos[1,:] = ori[1] + direct[1]*Z_ + R_*perp[1]
mp.plot(pos[0,:],pos[1,:])

emis1 = b1d1.get_emis(pos,0)
emis2 = b1d1.get_emis(pos,1)
emis = emis1-emis2
#emis = np.reshape(emis[0,:],(len(R[0,:]),len(Z[0,0,:])))
mp.figure()
mp.tricontourf(pos[0,:],pos[1,:],emis[0,:])
#mp.tricontourf(pos[0,:]-0.1,pos[1,:],emis1[0,:])
mp.title('emission')
mp.colorbar()

emis1 = b1d1.get_emis_fluc(pos,0)
emis2 = b1d1.get_emis_fluc(pos,1)
emis = emis1-emis2
#emis = np.reshape(emis[0,:],(len(R[0,:]),len(Z[0,0,:])))
mp.figure()
#mp.tricontourf(pos[0,:],pos[1,:],emis[0,:])
mp.tricontourf(pos[0,:]-0.1,pos[1,:],emis1[0,:])
mp.title('emission fluc')
mp.colorbar()

emistot = b1d1.get_emis_ave(pos)
emis1 = emistot[0,:,:]
emis2 = emistot[1,:,:]
emis = emis1-emis2
mp.figure()
#mp.tricontourf(pos[0,:],pos[1,:],emis[0,:])
mp.tricontourf(pos[0,:]-0.1,pos[1,:],emis1[0,:])
mp.title('emission average')
mp.colorbar()

#mp.figure()
#mp.plot(dl1,b1d1.density_beam[0,0,:],'k-', label='pt1')
#mp.plot(dl2,b1d2.density_beam[0,0,:],'m-', label='pt2')
#mp.plot(dl3,b1d3.density_beam[0,0,:],'r-', label='pt3')
#mp.plot(dl4,b1d4.density_beam[0,0,:],'b-', label='pt4')
#mp.plot(dl5,b1d5.density_beam[0,0,:],'g-', label='pt5')
#mp.plot(dl,b1d.density_beam[0,1,:],'b-', label='E2')
#mp.plot(dl,b1d.density_beam[0,2,:],'r-', label='E3')
#mp.legend()

mp.show()

