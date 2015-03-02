#!/usr/bin/env python
from FPSDP.Plasma.XGC_Profile.load_XGC_profile import *
from FPSDP.Geometry.Grid import *
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

test = 0
xgc_path = '/global/project/projectdirs/m499/jlang/particle_pinch/'

grid2D = Cartesian2D(DownLeft = (-0.5,1.4),UpRight = (0.5,2),NR = 256, NZ = 512)


time_start = 180
time_end = 180
time_step = 1

def load(full_load,fluc_only):
    xgc = XGC_Loader(xgc_path,grid2D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 1, Full_Load = full_load, Fluc_Only = fluc_only, load_ions = True)

    return xgc    

xgc = load(True,False)

print max(xgc.ne[0][0][:])

if test == 0:
    plt.figure()
    plt.contourf(xgc.ni_on_grid[0][0][:][:])
    plt.colorbar()
    
    
    
    mat = sio.loadmat('save.mat')
    n_e = mat['n_e']
    R = mat['Rgrid'][0,:]
    Z = mat['Zgrid'][0,:]
    print R.shape
    
    err = (xgc.ne[0][0][:]-mat['n_e'][:,0])/abs(n_e[:,0])
    print sum(xgc.ne[0][0][:])/xgc.ne.shape[2]
    print sum(n_e[:,0])/n_e.shape[0]
    
    v = np.linspace(-1,1,15)
    plt.figure()
    plt.plot(err)
    plt.figure()
    plt.tricontourf(R,Z,err,v)
    plt.colorbar(ticks=v)

    plt.figure()
    plt.title('MATLAB')
    plt.tricontourf(R,Z,n_e[:,0])
    plt.colorbar()

    plt.figure()
    plt.title('PYTHON')
    plt.tricontourf(R,Z,xgc.ni[0][0][:])
    plt.colorbar()
    
    plt.show()
elif test == 1:
    plt.figure()
    mk = 0.8
    for i in np.linspace(0.1,0.24,5):
        idx = np.where(abs(xgc.psi-i) < 1e-3)
        plt.plot(xgc.mesh['R'][idx],xgc.mesh['Z'][idx],'.b',markersize=mk)
    idx = np.where(abs(xgc.psi-0.2661) < 6e-4)
    plt.plot(xgc.mesh['R'][idx],xgc.mesh['Z'][idx],'.b',markersize=mk)
    idx = idx[0]
    plt.plot(xgc.mesh['R'][idx[0:-1:5]],xgc.mesh['Z'][idx[0:-1:5]],'x')
    # 0.27 0.285 0.2925
    for i in [0.27, 0.2775, 0.285]:
        idx = np.where(abs(xgc.psi-i) < 1e-3)
        plt.plot(xgc.mesh['R'][idx],xgc.mesh['Z'][idx],'.b',markersize=mk)
    #idx = np.where(xgc.psi > 0.31)
    #plt.plot(xgc.mesh['R'][idx],xgc.mesh['Z'][idx],'.k')
    #plt.plot(xgc.mesh['R'],xgc.mesh['Z'],'x')
                
    plt.axis('equal')
    plt.show()
