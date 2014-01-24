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
    """load the whole .m file and return a dictionary contains all the entries.
    """
    f = open(fname,'r')
    result = {}
    for line in f:
        words = line.split('=')
        key = words[0].strip()
        value = words[1].strip(' ;\n')
        result[key]= parse_num(value)
    f.close()
    return result

def find_nearest_3(Rwant,Zwant,R,Z,psi,psi_sp):
    """Find the nearest 3 points on the mesh for a given R,Z point, the result is used for 3 point interpolation.
    Argument:
    Rwant,Zwant: double, the R,Z coordinates for the desired point
    R,Z: double array, the array contains all the R,Z values on the mesh
    psi: double array, the array contains all the poloidal flux values
    psi_sp: spline interpolated psi value, used to get the psi value at the desired location. Also used for double check.

    return:
    length 3 list, contains the indices of the three nearest mesh points.  
    """

    #narrow down the search region by choosing only the points with psi values close to psi_want.
    psi_want = psi_sp(Zwant,Rwant).flatten()
    psi_max = np.max(psi)
    search_region = np.where(( np.absolute(psi-psi_want)<= psi_max/100 ))[0]
    
    distance = np.sqrt((Rwant-R[search_region])**2 + (Zwant-Z[search_region])**2)
    min1 = np.argmin(distance)
    if (len([min1]) >= 3):
        return search_region[min1[0,1,2]]
    elif(len([min1]) == 2):
        min3 = np.argmin(np.delete(distance,min1))
        return [search_region[min1[0]], search_region[min1[1]], np.delete(search_region,min1)[min3]]
    else:
        min2 = np.argmin(np.delete(distance,min1))
        if (len([min2]) >= 2):
            return [search_region[min1],np.delete(search_region,min1)[min2[0]], np.delete(search_region,min1)[min2[1]]]
        else:
            min3 = np.argmin(np.delete(np.delete(distance,min1),min2))
            print min1,min2,min3
            return [search_region[min1],np.delete(search_region,min1)[min2],np.delete(np.delete(search_region,min1),min2)[min3]]
    

    
    

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

