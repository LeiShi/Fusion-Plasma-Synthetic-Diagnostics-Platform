"""Load XGC output data, interpolate electron density perturbation onto desired Cartesian grid mesh. 
"""

from ...Maths.Interpolation import linear_3d_3point
from ...GeneralSettings.UnitSystem import cgs

import numpy as np
import h5py as h5
from scipy.interpolate import griddata
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

data_path = '/p/gkp/lshi/FWR_XGC_Interface/new_XGC_data/'

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

def find_nearest_4(Rwant,Zwant,my_xgc):
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
    psi_sp = my_xgc.psi_sp
    psi = my_xgc.psi
    R = my_xgc.mesh['R']
    Z = my_xgc.mesh['Z']
    
    psi_want = psi_sp(Zwant,Rwant).flatten()
    psi_max = np.max(psi)
    inner_search_region = np.intersect1d( np.where(( np.absolute(psi-psi_want)<= psi_max/100))[0],np.where(( psi-psi_want <0))[0], assume_unique = True)
    outer_search_region = np.intersect1d( np.where((psi-psi_want)<=psi_max/100)[0],np.where( (psi-psi_want)>=0 )[0])
    
    inner_distance = np.sqrt((Rwant-R[inner_search_region])**2 + (Zwant-Z[inner_search_region])**2)
    outer_distance = np.sqrt((Rwant-R[outer_search_region])**2 + (Zwant-Z[outer_search_region])**2)
    
    nearest_4 = []
    min1 = np.argmin(inner_distance)
    min2 = np.argmin(np.delete(inner_distance,min1))
    nearest_4.append( inner_search_region[min1] )
    nearest_4.append( np.delete(inner_search_region,min1)[min2] )

    min3 = np.argmin(outer_distance)
    min4 = np.argmin(np.delete(outer_distance,min3))
    nearest_4.append(outer_search_region[min3])
    nearest_4.append(np.delete(outer_search_region,min3)[min4])

    return np.array(nearest_4)


class XGC_loader():
    """Loader for a given set of XGC output files
    """

    def __init__(this,xgc_path,grid,t_start,t_end,dt):
        this.xgc_path = xgc_path
        this.mesh_file = xgc_path + 'xgc.mesh.h5'
        this.bfield_file = xgc_path + 'xgc.bfield.h5'
        this.time_steps = np.arange(t_start,t_end+dt,dt)
        this.grid = grid
        this.load_mesh()
        this.load_psi()
        this.load_B()
        this.load_eq()
        this.load_fluctuations()
        this.calc_total_ne()
    
    def change_grid(this,grid):
        this.grid = grid

    
    def load_mesh(this):
        """Load the R-Z data

         
        """
        mesh = h5.File(this.mesh_file,'r')
        RZ = mesh['coordinates']['values']
        Rpts =RZ[:,0]
        Zpts = RZ[:,1]
        mesh.close()
        this.points = np.array([Zpts,Rpts]).transpose()
        this.mesh = {'R':Rpts, 'Z':Zpts}
        return 0

    def load_psi(this):
        """Load psi data

        spline over Z,R
        Note that choose R as the 2nd variable in order to store it in the fastest dimension later
        """
        mesh = h5.File(this.mesh_file,'r')
        this.psi = mesh['psi'][...]
        this.psi_on_grid = griddata(this.points, this.psi, (this.grid.Z2D,this.grid.R2D), method = 'cubic', fill_value=np.max(this.psi))
        mesh.close()
        return 0

    def load_B(this):
        """Load the B data

        Spline over Z,R plane
        """
        B_mesh = h5.File(this.bfield_file,'r')
        this.B = B_mesh['node_data[0]']['values']
        this.B_total = np.sqrt(this.B[:,0]**2 + this.B[:,1]**2 + this.B[:,2]**2)
        this.B_on_grid = griddata(this.points, this.B_total, (this.grid.Z2D,this.grid.R2D), method = 'cubic') 
        B_mesh.close()
        return 0

    def load_fluctuations(this,planes = [0]):
        """Load non-adiabatic electron density and electrical static potential fluctuations
        the mean value of these two quantities on each time step is also calculated.
        """
        this.nane = np.zeros( (len(this.time_steps),len(planes),len(this.mesh['R'])) )
        this.phi = np.zeros(this.nane.shape)
        this.nane_bar = np.zeros((len(this.time_steps)))
        this.phi_bar = np.zeros(this.nane_bar.shape)
        for i in range(len(this.time_steps)):
            flucf = this.xgc_path + 'xgc.3d.'+str(this.time_steps[i]).zfill(5)+'.h5'
            fluc_mesh = h5.File(flucf,'r')
            
            this.nane[i] += np.swapaxes(fluc_mesh['eden'][...][:,planes],0,1)
            this.phi[i] += np.swapaxes(fluc_mesh['dpot'][...][:,planes],0,1)

            this.nane_bar[i] += np.mean(fluc_mesh['eden'][...])
            this.phi_bar[i] += np.mean(fluc_mesh['dpot'][...])

            this.nane[i] -= this.nane_bar[i]
            this.phi[i] -= this.phi_bar[i]
            
            fluc_mesh.close()
        return 0
    
    def load_eq(this):
        """Load equilibrium profiles, including ne0, Te0
        """
        eqf = this.xgc_path + 'xgc.oneddiag.h5'
        eq_mesh = h5.File(eqf,'r')
        this.eq_psi = eq_mesh['psi_mks'][:]
        this.eq_te = eq_mesh['e_perp_temperature_1d'][0,:]
        this.eq_ne = eq_mesh['e_gc_density_1d'][0,:]
        eq_mesh.close()
        this.te0_sp = interp1d(this.eq_psi,this.eq_te,bounds_error = False,fill_value = 0)
        this.ne0_sp = interp1d(this.eq_psi,this.eq_ne,bounds_error = False,fill_value = 0)
        this.te0 = this.te0_sp(this.psi)
        this.ne0 = this.ne0_sp(this.psi)

    def calc_total_ne(this):
        """calculate the total electron density in raw XGC grid points
        """
        ne0 = this.ne0
        te0 = this.te0
        inner_idx = np.where(np.absolute(te0)>0)[0]
        dne_ad = np.zeros(this.phi.shape)
        dne_ad[:,:,inner_idx] += ne0[inner_idx] * this.phi[:,:,inner_idx] /this.te0[inner_idx]
        ad_valid_idx = np.where(np.absolute(dne_ad)<= np.max(np.absolute(ne0)))[0]
        na_valid_idx = np.where(np.absolute(this.nane)<= np.max(np.absolute(ne0)))[0]
        this.ne = np.zeros(dne_ad.shape)
        this.ne += ne0[np.newaxis,np.newaxis,:]
        this.ne[:,:,ad_valid_idx] += dne_ad[:,:,ad_valid_idx]
        this.ne[:,:,na_valid_idx] += this.nane[:,:,na_valid_idx]

    def interpolate_all_on_grid(this):
        """ create all interpolated quantities on given grid.
        """
        R2D = this.grid.R2D
        Z2D = this.grid.Z2D
        this.ne_on_grid = np.zeros((this.ne.shape[0],this.ne.shape[1],R2D.shape[0],R2D.shape[1]))
        this.phi_on_grid = np.zeros(this.ne_on_grid.shape)
        this.dne_ad_on_grid = np.zeros(this.ne_on_grid.shape)
        this.nane_on_grid = np.zeros(this.ne_on_grid.shape)
        for i in range(this.ne.shape[0]):
            for j in range(this.ne.shape[1]):
                this.ne_on_grid[i,j,...] += griddata(this.points,this.ne[i,j,:],(Z2D,R2D),method = 'cubic', fill_value = 0)
                this.phi_on_grid[i,j,...] += griddata(this.points,this.phi[i,j,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
                this.nane_on_grid[i,j,...] += griddata(this.points,this.nane[i,j,:],(Z2D,R2D),method = 'cubic',fill_value = 0)



