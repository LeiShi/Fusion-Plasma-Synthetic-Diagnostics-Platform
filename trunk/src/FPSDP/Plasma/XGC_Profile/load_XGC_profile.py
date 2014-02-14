"""Load XGC output data, interpolate electron density perturbation onto desired Cartesian grid mesh. 
"""
from ...GeneralSettings.UnitSystem import cgs
from ...Geometry.Grid import Cartesian2D,Cartesian3D

import numpy as np
import h5py as h5
from scipy.interpolate import griddata
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.io.netcdf as nc

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
    """Find the nearest 4 points on the mesh for a given R,Z point, the result is used for 4 point interpolation.

    Argument:
    Rwant,Zwant: double, the R,Z coordinates for the desired point
    my_xgc: XGC_Loader object, containing all the detailed information for the XGC output data
    
    return:
    length 4 list, contains the indices of the three nearest mesh points.  
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

        if isinstance(grid,Cartesian2D):
            this.load_mesh_2D()
            this.load_psi_2D()
            this.load_B_2D()
            this.load_eq_2D()
            this.load_fluctuations_2D()
            this.calc_total_ne_2D()
            this.interpolate_all_on_grid_2D()
        elif isinstance(grid, Cartesian3D):
            grid.ToCynlindrical()
            
            
    
    def change_grid(this,grid):
        """change the current grid to another grid,reload all quantities on new grid
        Argument:
        grid: Grid object, normally Cartesian2D or Cartesian3D
        """
        this.grid = grid
        

    
    def load_mesh_2D(this):
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

    def load_psi_2D(this):
        """Load psi data

        spline over Z,R
        Note that choose R as the 2nd variable in order to store it in the fastest dimension later
        """
        mesh = h5.File(this.mesh_file,'r')
        this.psi = mesh['psi'][...]
        this.psi_on_grid = griddata(this.points, this.psi, (this.grid.Z2D,this.grid.R2D), method = 'cubic', fill_value=np.max(this.psi))
        mesh.close()
        return 0

    def load_mesh_psi_3D(this):
        """load R-Z mesh and psi values, then create map between each psi value and the series of points on that surface, calculate the arc length table.
        """
        mesh = h5.File(this.mesh_file,'r')
        RZ = mesh['coordinates']['values']
        Rpts =RZ[:,0]
        Zpts = RZ[:,1]
        this.points = np.array([Zpts,Rpts]).transpose()
        this.mesh = {'R':Rpts, 'Z':Zpts}

        this.psi = mesh['psi'][...]
        this.psi_on_grid = griddata(this.points, this.psi, (this.grid.Z2D,this.grid.R2D), method = 'cubic', fill_value=np.max(this.psi))

        
        mesh.close()
        

    def load_B_2D(this):
        """Load the B data

        Spline over Z,R plane
        """
        B_mesh = h5.File(this.bfield_file,'r')
        this.B = B_mesh['node_data[0]']['values']
        this.B_total = np.sqrt(this.B[:,0]**2 + this.B[:,1]**2 + this.B[:,2]**2)
        this.B_on_grid = griddata(this.points, this.B_total, (this.grid.Z2D,this.grid.R2D), method = 'cubic') 
        B_mesh.close()
        return 0

    def load_fluctuations_2D(this):
        """Load non-adiabatic electron density and electrical static potential fluctuations
        the mean value of these two quantities on each time step is also calculated.
        """
        this.nane = np.zeros( (len(this.time_steps),len(this.mesh['R'])) )
        this.phi = np.zeros(this.nane.shape)
        this.nane_bar = np.zeros((len(this.time_steps)))
        this.phi_bar = np.zeros(this.nane_bar.shape)
        for i in range(len(this.time_steps)):
            flucf = this.xgc_path + 'xgc.3d.'+str(this.time_steps[i]).zfill(5)+'.h5'
            fluc_mesh = h5.File(flucf,'r')
            
            this.nane[i] += np.swapaxes(fluc_mesh['eden'][...][:,0],0,1)
            this.phi[i] += np.swapaxes(fluc_mesh['dpot'][...][:,0],0,1)

            this.nane_bar[i] += np.mean(fluc_mesh['eden'][...])
            this.phi_bar[i] += np.mean(fluc_mesh['dpot'][...])

            this.nane[i] -= this.nane_bar[i]
            this.phi[i] -= this.phi_bar[i]
            
            fluc_mesh.close()
        return 0
    
    def load_eq_2D(this):
        """Load equilibrium profiles, including ne0, Te0
        """
        eqf = this.xgc_path + 'xgc.oneddiag.h5'
        eq_mesh = h5.File(eqf,'r')
        this.eq_psi = eq_mesh['psi_mks'][:]
        this.eq_te = eq_mesh['e_perp_temperature_1d'][0,:]
        this.eq_ti = eq_mesh['i_perp_temperature_1d'][0,:]
        this.eq_ne = eq_mesh['e_gc_density_1d'][0,:]
        eq_mesh.close()
        this.te0_sp = interp1d(this.eq_psi,this.eq_te,bounds_error = False,fill_value = 0)
        this.ne0_sp = interp1d(this.eq_psi,this.eq_ne,bounds_error = False,fill_value = 0)
        this.ti0_sp = interp1d(this.eq_psi,this.eq_ti,bounds_error = False,fill_value = 0)
        this.te0 = this.te0_sp(this.psi)
        this.ne0 = this.ne0_sp(this.psi)

    def calc_total_ne_2D(this):
        """calculate the total electron density in raw XGC grid points
        """
        ne0 = this.ne0
        te0 = this.te0
        inner_idx = np.where(np.absolute(te0)>0)[0]
        this.dne_ad = np.zeros(this.phi.shape)
        this.dne_ad[:,inner_idx] += ne0[inner_idx] * this.phi[:,inner_idx] /this.te0[inner_idx]
        ad_valid_idx = np.where(np.absolute(this.dne_ad)<= np.max(np.absolute(ne0)))[0]
        na_valid_idx = np.where(np.absolute(this.nane)<= np.max(np.absolute(ne0)))[0]
        this.ne = np.zeros(this.dne_ad.shape)
        this.ne += ne0[np.newaxis,:]
        this.ne[:,ad_valid_idx] += this.dne_ad[:,ad_valid_idx]
        this.ne[:,na_valid_idx] += this.nane[:,na_valid_idx]

    def interpolate_all_on_grid_2D(this):
        """ create all interpolated quantities on given grid.
        """
        R2D = this.grid.R2D
        Z2D = this.grid.Z2D
        this.ne_on_grid = np.zeros((len(this.time_steps),R2D.shape[0],R2D.shape[1]))
        this.phi_on_grid = np.zeros(this.ne_on_grid.shape)
        this.dne_ad_on_grid = np.zeros(this.ne_on_grid.shape)
        this.nane_on_grid = np.zeros(this.ne_on_grid.shape)
        for i in range(this.ne.shape[0]):
            this.ne_on_grid[i,...] += griddata(this.points,this.ne[i,:],(Z2D,R2D),method = 'cubic', fill_value = 0)
            this.phi_on_grid[i,...] += griddata(this.points,this.phi[i,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
            this.dne_ad_on_grid[i,...] += griddata(this.points,this.dne_ad[i,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
            this.nane_on_grid[i,...] += griddata(this.points,this.nane[i,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
        this.te_on_grid = this.te0_sp(this.psi_on_grid)
        this.ti_on_grid = this.ti0_sp(this.psi_on_grid)

    def cdf_output_2D(this,output_path,filehead='fluctuation'):
        """write out cdf files for old FWR2D code use

        Arguments:
        output_path: string, the full path to put the output files
        filehead: string, the starting string of all filenames

        CDF file format:
        Dimensions:
        r_dim: int, number of grid points in R direction.
        z_dim: int, number of grid points in Z direction
        
        Variables:
        rr: 1D array, coordinates in R direction, in Meter
        zz: 1D array, coordinates in Z direction, in Meter
        bb: 2D array, total magnetic field on grids, in Tesla, shape in (z_dim,r_dim)
        ne: 2D array, total electron density on grids, in m^-3
        ti: 2D array, total ion temperature, in keV
        te: 2D array, total electron temperature, in keV
        
        """
        file_start = output_path + filehead
        for i in range(len(this.time_steps)):
            fname = file_start + str(this.time_steps[i]) + '.cdf'
            f = nc.netcdf_file(fname,'w')
            f.createDimension('z_dim',this.grid.NZ)
            f.createDimension('r_dim',this.grid.NR)

            rr = f.createVariable('rr','d',('r_dim',))
            rr[:] = this.grid.R1D[:]
            zz = f.createVariable('zz','d',('z_dim',))
            zz[:] = this.grid.Z1D[:]
            rr.units = zz.units = 'Meter'

            bb = f.createVariable('bb','d',('z_dim','r_dim'))
            bb[:,:] = this.B_on_grid[:,:]
            bb.units = 'Tesla'

            ne = f.createVariable('ne','d',('z_dim','r_dim'))
            ne[:,:] = this.ne_on_grid[i,:,:]
            ne.units = 'per cubic meter'

            te = f.createVariable('te','d',('z_dim','r_dim'))
            te[:,:] = this.te_on_grid[:,:]/1000
            te.units = 'keV'

            ti = f.createVariable('ti','d',('z_dim','r_dim'))
            ti[:,:] = this.ti_on_grid[:,:]/1000
            ti.units = 'keV'

            f.close()
        

