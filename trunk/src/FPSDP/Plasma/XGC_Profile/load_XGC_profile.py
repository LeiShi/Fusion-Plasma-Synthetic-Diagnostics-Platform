"""Load XGC output data, interpolate electron density perturbation onto desired Cartesian grid mesh. 
"""

from ...Maths.Interpolatation import linear_3d_3point

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
    


class XGC_loader():
    """Loader for a given set of XGC output files
    """

    def __init__(this,xgc_path,t_start,t_end,dt):
        this.xgc_path = xgc_path
        this.mesh_file = xgc_path + 'xgc.mesh.h5'
        this.bfield_file = xgc_path + 'xgc.bfield.h5'
        this.time_steps = np.arange(t_start,t_end+dt,dt)
        this.load_mesh()
        this.load_psi()
        this.load_B()
        this.load_phi()
    

    def load_mesh(this):
        """Load the R-Z data

         
        """
        mesh = h5.File(this.mesh_file,'r')
        RZ = mesh['coordinates']['values']
        Rpts =RZ[:,0]
        Zpts = RZ[:,1]
        mesh.close()
        this.mesh = dict(R=Rpts,Z=Zpts)
        return 0

    def load_psi(this):
        """Load psi data

        spline over Z,R
        Note that choose R as the 2nd variable in order to store it in the fastest dimension later
        """
        mesh = h5.File(this.mesh_file,'r')
        this.psi = mesh['psi'][...]
        this.psi_sp = SmoothBivariateSpline(this.mesh['Z'],this.mesh['R'],this.psi)
        mesh.close()
        return 0

    def load_B(this):
        """Load the B data

        Spline over Z,R plane
        """
        B_mesh = h5.File(this.bfield_file,'r')
        this.B = B_mesh['node_data[0]']['values']
        this.B_total = np.sqrt(this.B[:,0]**2 + this.B[:,1]**2 + this.B[:,2]**2)
        this.B_sp = SmoothBivariateSpline(this.mesh['Z'],this.mesh['R'],this.B_total)
        B_mesh.close()
        return 0

    def load_phi(this,planes = [0]):
        """Load phi data
        """
        this.phi = np.zeros( (len(this.time_steps),len(this.mesh['R']),len(planes)) )
        for i in range(len(this.time_steps)):
            phif = this.xgc_path + 'xgc.3d.'+str(this.time_steps[i]).zfill(5)+'.h5'
            phi_mesh = h5.File(phif,'r')
            
            this.phi[i] += phi_mesh['eden'][...][:,planes]
            
            phi_mesh.close()
        return 0



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

