"""local test of xgc data files
"""

import numpy as np
import h5py as h5
from scipy.interpolate import SmoothBivariateSpline
import matplotlib.pyplot as plt

data_path = '/p/gkp/lshi/FWR_XGC_Interface/XGC_data/'

mf = data_path + 'xgc.mesh.h5'
bf = data_path + 'xgc.bfield.h5'
phif = data_path + 'xgc.3d.00150.h5'


def parse_num(s):
    """ parse a string into either int or float.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)

def load_m(fname):
    f = open(fname,'r')
    result = {}
    for line in f:
        words = line.split('=')
        key = words[0].strip()
        value = words[1].strip(' ;\n')
        result[key]= parse_num(value)
    f.close()
    return result
    

def get_mesh():
    """Load the R-Z data

    return a dictionary with keywords: R, Z 
    """
    mesh = h5.File(mf,'r')
    RZ = mesh['coordinates']['values']
    Rpts =RZ[:,0]
    Zpts = RZ[:,1]
    mesh.close()
    return dict(R=Rpts,Z=Zpts)

def get_psi(Z,R):
    """Load psi data

    spline over Z,R
    Note that choose R as the 2nd variable in order to store it in the fastest dimension later
    """
    mesh = h5.File(mf,'r')
    psi = mesh['psi'][...]
    psi_sp = SmoothBivariateSpline(Z,R,psi)
    mesh.close()
    return psi_sp

def get_B(Z,R):
    """Load the B data

    Spline over Z,R plane
    """
    B_mesh = h5.File(bf,'r')
    B = B_mesh['node_data[0]']['values']
    B_total = np.sqrt(B[:,0]**2 + B[:,1]**2 + B[:,2]**2)
    B_sp = SmoothBivariateSpline(Z,R,B_total)
    B_mesh.close()
    return B_sp

def get_phi(Z,R,plane = 0):
    """Load phi data

    Spline over Z,R plane
    """
    phi_mesh = h5.File(phif,'r')
    phi = phi_mesh['eden']
    phi_sp = SmoothBivariateSpline(Z,R,phi[:,plane],s=10000)
    phi_mesh.close()
    return phi_sl



def plot_all(**C):
    """plot all the quantities: (R,Z),psi,B, and phi.
    """
    #plot R,Z dots
    if ('R' in C.keys() and 'Z' in C.keys() ):
        R=C['R']
        Z=C['Z']
        plt.figure(1)
        plt.plot(R,Z,'b+')
    else:
        raise Exception('no R or Z data passed in, plot failed.')

    #set R,Z ranges
    rmin = np.min(R)
    rmax = np.max(R)
    zmin = np.min(Z)
    zmax = np.max(Z)
    my_extent = (rmin,rmax,zmin,zmax)
    rr = np.linspace(rmin,rmax,200)
    zz = np.linspace(zmin,zmax,200)

    if ('psi' in C.keys()):
        
        #plot psi
        plt.figure(2)
        psi_sl = C['psi']
        psi_img = psi_sl(zz,rr)
        plt.imshow(psi_img,extent = my_extent)
        plt.colorbar()
    
    if ('B' in C.keys()):    
        #plot B
        plt.figure(3)
        B_sl = C['B']
        B_img = B_sl(zz,rr)
        plt.imshow(B_img,extent = my_extent)
        plt.colorbar()

    if ('phi' in C.keys()): 
        #plot phi
        plt.figure(4)
        phi_sl = C['phi']
        phi_img = phi_sl(zz,rr)
        plt.imshow(phi_img,extent = my_extent)
        plt.colorbar()

def close_plot():

    for i in range(4):
        plt.close()

