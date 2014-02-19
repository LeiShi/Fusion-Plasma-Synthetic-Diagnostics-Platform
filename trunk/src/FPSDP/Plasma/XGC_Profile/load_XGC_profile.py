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

def get_interp_planes(my_xgc):
    """Get the plane numbers used for interpolation for each point 
    """
    dPHI = 2 * np.pi / my_xgc.n_plane
    phi_planes = np.arange(my_xgc.n_plane)*dPHI
    if(my_xgc.CO_DIR):
        nextplane = np.searchsorted(phi_planes,my_xgc.grid.phi3D,side = 'right')
        prevplane = nextplane - 1
        nextplane[np.nonzero(nextplane == my_xgc.n_plane)] = 0
    else:
        prevplane = np.searchsorted(phi_planes,my_xgc.grid.phi3D,side = 'right')
        nextplane = prevplane - 1
        prevplane[np.nonzero(nextplane == my_xgc.n_plane)] = 0

    return (prevplane,nextplane)
    
    

def find_interp_positions(my_xgc):
    """Find the interpolation R-Z positions on previous and next planes for 3D mesh  given by my_xgc.grid.

    Argument:
    my_xgc: XGC_Loader object, containing all the detailed information for the XGC output data and desired Cartesian mesh data
    
    returns:
    double array with shape (2,2,NZ,NY,NX), contains the R and Z values on both planes that should be used to interpolate. the last 3 dimensions corresponds to each desired mesh point, the first 2 dimensions contains the 2 pairs of (Z,R,portion) values. The previous plane interpolation point is stored in [0,:,...], with order (Z,R,portion), the next plane point in [1,:,...]

    Note: previous plane means the magnetic field line comes from this plane and go through the desired mesh point, next plane means this magnetic field line lands on this plane. In XGC output files,  these planes are not necessarily stored in increasing index ordering. Direction of toroidal field determines this.  
    """

    #narrow down the search region by choosing only the points with psi values close to psi_want.
    psi = my_xgc.psi
    R = my_xgc.mesh['R']
    Z = my_xgc.mesh['Z']

    Rwant = my_xgc.grid.r3D
    Zwant = my_xgc.grid.z3D
    PHIwant = my_xgc.grid.phi3D

    NZ = Rwant.shape[0]
    NY = Rwant.shape[1]
    NX = Rwant.shape[2]

    nextnode = my_xgc.nextnode
    prevnode = my_xgc.prevnode

    prevplane,nextplane = my_xgc.prevplane,my_xgc.nextplane
    dPHI = 2*np.pi / my_xgc.n_plane
    phi_planes = np.arange(my_xgc.n_plane)*dPHI
    
    interp_positions = np.zeros((2,3,NZ,NY,NX))
   
    psi_want = griddata(my_xgc.points,my_xgc.psi,(Zwant,Rwant),method = 'cubic',fill_value = -1)
    for i in range(NZ):
        for j in range(NY):
            for k in range(NX):
                
                if( psi_want[i,j,k] < 0 ):
                    # if the desired point is outside of XGC mesh, all quantities will be set to zero except total B. Here use R,Z = -1 as the flag
                    interp_positions[i,j,k,...] += -1
                else:
                    # first, find the 2 XGC mesh points that are nearest to the desired points, one inside of the flux surface, the other outside.
                    psi_max = np.max(psi)
                    inner_search_region = np.intersect1d( np.where(( np.absolute(psi-psi_want[i,j,k])<= psi_max/100))[0],np.where(( psi-psi_want[i,j,k] <0))[0], assume_unique = True)
                    outer_search_region = np.intersect1d( np.where((psi-psi_want[i,j,k])<=psi_max/100)[0],np.where( (psi-psi_want[i,j,k])>=0 )[0])
    
                    inner_distance = np.sqrt((Rwant[i,j,k]-R[inner_search_region])**2 + (Zwant[i,j,k]-Z[inner_search_region])**2)
                    outer_distance = np.sqrt((Rwant[i,j,k]-R[outer_search_region])**2 + (Zwant[i,j,k]-Z[outer_search_region])**2)

                    #nearest2 contains the index of the 2 nearest mesh points
                    nearest2 = []
                    min1 = np.argmin(inner_distance)
                    nearest2.append( inner_search_region[min1] ) 

                    min2 = np.argmin(outer_distance)
                    nearest2.append( outer_search_region[min2] )

                    #Calculate the portion of toroidal angle between interpolation planes
                    prevp = prevplane[i,j,k]
                    nextp = nextplane[i,j,k]
                   
                    phi = my_xgc.grid.phi3D[i,j,k]
                    if(prevp != 0 or phi <= dPHI):
                        portion_p = abs(phi-phi_planes[prevp])/dPHI
                    else:
                        portion_p = abs(phi - 2*np.pi)/dPHI
                        
                    if(nextp != 0 or phi <= dPHI):    
                        portion_n = abs(phi-phi_planes[nextp])/dPHI
                    else:
                        portion_n = abs(phi - 2*np.pi)/dPHI

                    #Calculate the expected r,z positions on next and previous planes
                    #first try, use the inner nearest point alone

                    ncur = nearest2[0]
                    nnext = nextnode[ncur]
                    nprev = prevnode[ncur]
                    if(nprev != -1):
                       
                        r,z = Rwant[i,j,k],Zwant[i,j,k]
                        Rcur,Zcur = R[ncur],Z[ncur]
                        Rnext,Znext = R[nnext],Z[nnext]
                        Rprev,Zprev = R[nprev],Z[nprev]

                        dRnext = portion_n * (Rnext-Rcur)
                        dZnext = portion_n * (Znext-Zcur)
                        dRprev = portion_p * (Rprev-Rcur)
                        dZprev = portion_p * (Zprev-Zcur)
                    else:
                        #if the previous node is not found(normally because of lack of resolution), use next node alone to determine the interpolation positions.
                        r,z = Rwant[i,j,k],Zwant[i,j,k]
                        Rcur,Zcur = R[ncur],Z[nur]
                        Rnext,Znext = R[nnext],Z[nnext]

                        dRnext = portion_n * (Rnext-Rcur)
                        dZnext = portion_n * (Znext-Zcur)
                        dRprev = portion_p * (Rcur-Rnext)
                        dZprev = portion_p * (Zcur-Znext) 
                        
                    interp_positions[0,0,i,j,k] = z + dZprev
                    interp_positions[0,1,i,j,k] = r + dRprev
                    interp_positions[0,2,i,j,k] = portion_p
                    interp_positions[1,0,i,j,k] = z + dZnext
                    interp_positions[1,1,i,j,k] = r + dRnext
                    interp_positions[1,2,i,j,k] = portion_n
                    
                    

    return interp_positions




class XGC_loader():
    """Loader for a given set of XGC output files
    """

    def __init__(this,xgc_path,grid,t_start,t_end,dt):

        print 'Loading XGC output data'
        this.xgc_path = xgc_path
        this.mesh_file = xgc_path + 'xgc.mesh.h5'
        this.bfield_file = xgc_path + 'xgc.bfield.h5'
        this.time_steps = np.arange(t_start,t_end+dt,dt)
        this.grid = grid
        print 'from directory:'+ this.xgc_path
        
        if isinstance(grid,Cartesian2D):
            print '2D Grid detected.'
            this.load_mesh_2D()
            print 'mesh loaded.'
            this.load_psi_2D()
            print 'psi loaded.'
            this.load_B_2D3D()
            print 'B loaded.'
            this.load_eq_2D3D()
            print 'equilibrium loaded.'
            this.load_fluctuations_2D()
            print 'fluctuations loaded.'
            this.calc_total_ne_2D3D()
            print 'total ne calculated.'
            this.interpolate_all_on_grid_2D()
            print 'quantities interpolated on grid.\n XGC data sucessfully loaded.'
            
        elif isinstance(grid, Cartesian3D):
            print '3D grid detected.'

            grid.ToCylindrical()
            print 'cynlindrical coordinates created.'

            this.load_mesh_psi_3D()
            print 'mesh and psi loaded.'

            this.prevplane,this.nextplane = get_interp_planes(this)
            print 'interpolation planes obtained.'

            this.load_B_3D()
            print 'B loaded.'

            this.load_eq_2D3D()
            print 'equlibrium loaded.'

            this.load_fluctuations_3D()
            print 'fluctuations loaded.'

            this.calc_total_ne_2D3D()
            print 'total ne calculated.'

            this.interpolate_all_on_grid_3D()
            print 'all quantities interpolated on grid.\n XGC data sucessfully loaded.'

            
            
            
    
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

        this.nextnode = mesh['nextnode'][...]
        
        this.prevnode = np.zeros(this.nextnode.shape)
        for i in range(len(this.nextnode)):
            prevnodes = np.nonzero(this.nextnode == i)[0]
            if( len(prevnodes)>0 ):
                this.prevnode[i] = prevnodes[0]
            else:
                this.prevnode[i] = -1

        this.CO_DIR = True
        
        this.psi = mesh['psi'][...]
        this.psi_on_grid = griddata(this.points, this.psi, (this.grid.z3D,this.grid.r3D), method = 'cubic', fill_value=np.max(this.psi))

        mesh.close()

        # get the number of toroidal planes from fluctuation data file
        fluc_file0 = this.xgc_path + 'xgc.3d.' + str(this.time_steps[0]).zfill(5)+'.h5'
        fmesh = h5.File(fluc_file0,'r')
        this.n_plane = fmesh['dpot'].shape[1]

        fmesh.close()
        
        
        
        

    def load_B_2D(this):
        """Load equilibrium magnetic field data

        B_total is interpolated over Z,R plane on given 2D Cartesian grid
        """
        B_mesh = h5.File(this.bfield_file,'r')
        this.B = B_mesh['node_data[0]']['values']
        this.B_total = np.sqrt(this.B[:,0]**2 + this.B[:,1]**2 + this.B[:,2]**2)
        this.B_on_grid = griddata(this.points, this.B_total, (this.grid.Z2D,this.grid.R2D), method = 'cubic') 
        B_mesh.close()
        return 0

    def load_B_3D(this):
        """Load equilibrium magnetic field data

        B_total is interpolated over Z,R plane on given 3D Cartesian grid, since B_0 is assumed symmetric along toroidal direction
        """
        B_mesh = h5.File(this.bfield_file,'r')
        this.B = B_mesh['node_data[0]']['values']
        this.B_total = np.sqrt(this.B[:,0]**2 + this.B[:,1]**2 + this.B[:,2]**2)
        this.B_on_grid = griddata(this.points, this.B_total, (this.grid.z3D,this.grid.r3D), method = 'cubic') 
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

    def load_fluctuations_3D(this):
        """Load non-adiabatic electron density and electrical static potential fluctuations for 3D mesh. The required planes are calculated and stored in sorted array. fluctuation data on each plane is stored in the same order. 
        the mean value of these two quantities on each time step is also calculated.
        """

        this.planes = np.unique(np.array([np.unique(this.prevplane),np.unique(this.nextplane)]))
        this.planeID = {this.planes[i]:i for i in range(len(this.planes))} #the dictionary contains the positions of each chosen plane, useful when we want to get the data on a given plane known only its plane number in xgc file.
        
        this.nane = np.zeros( (len(this.time_steps),len(this.planes),len(this.mesh['R'])) )
        this.phi = np.zeros(this.nane.shape)
        this.nane_bar = np.zeros((len(this.time_steps)))
        this.phi_bar = np.zeros(this.nane_bar.shape)
        for i in range(len(this.time_steps)):
            flucf = this.xgc_path + 'xgc.3d.'+str(this.time_steps[i]).zfill(5)+'.h5'
            fluc_mesh = h5.File(flucf,'r')
            
            this.nane[i] += np.swapaxes(fluc_mesh['eden'][...][:,this.planes],0,1)
            this.phi[i] += np.swapaxes(fluc_mesh['dpot'][...][:,this.planes],0,1)

            this.nane_bar[i] += np.mean(fluc_mesh['eden'][...])
            this.phi_bar[i] += np.mean(fluc_mesh['dpot'][...])

            this.nane[i] -= this.nane_bar[i]
            this.phi[i] -= this.phi_bar[i]
            
            fluc_mesh.close()
        return 0
    
    def load_eq_2D3D(this):
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

    def calc_total_ne_2D3D(this):
        """calculate the total electron density in raw XGC grid points
        """
        ne0 = this.ne0
        te0 = this.te0
        inner_idx = np.where(np.absolute(te0)>0)[0]
        this.dne_ad = np.zeros(this.phi.shape)
        this.dne_ad[...,inner_idx] += ne0[inner_idx] * this.phi[...,inner_idx] /this.te0[inner_idx]
        ad_valid_idx = np.where(np.absolute(this.dne_ad)<= np.max(np.absolute(ne0)))
        na_valid_idx = np.where(np.absolute(this.nane)<= np.max(np.absolute(ne0)))
        this.ne = np.zeros(this.dne_ad.shape)
        this.ne += ne0[:]
        this.ne[ad_valid_idx] += this.dne_ad[ad_valid_idx]
        this.ne[na_valid_idx] += this.nane[na_valid_idx]

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

    def interpolate_all_on_grid_3D(this):
        """ create all interpolated quantities on given 3D grid.
        """
        r3D = this.grid.r3D
        z3D = this.grid.z3D
        this.ne_on_grid = np.zeros((len(this.time_steps),r3D.shape[0],r3D.shape[1],r3D.shape[2]))
        this.phi_on_grid = np.zeros(this.ne_on_grid.shape)
        this.dne_ad_on_grid = np.zeros(this.ne_on_grid.shape)
        this.nane_on_grid = np.zeros(this.ne_on_grid.shape)

        interp_positions = find_interp_positions(this)
        
        for i in range(len(this.time_steps)):
            #for each time step, first create the 2 arrays of quantities for interpolation
            prev = np.zeros( (this.grid.NZ,this.grid.NY,this.grid.NX) )
            next = np.zeros(prev.shape)
            
            #create index dictionary, for each key as plane number and value the corresponding indices where the plane is used as previous or next plane.
            prev_idx = {}
            next_idx = {}
            for j in this.planes:
                prev_idx[j] = np.where(this.prevplane == j )
                next_idx[j] = np.where(this.nextplane == j )

            #now interpolate ne on each plane for the points using it as previous or next plane.
            for j in this.planes:
                prev[prev_idx[j]] = griddata(this.points,this.ne[i,j,:],(interp_positions[0,0][prev_idx[j]], interp_positions[0,1][prev_idx[j]]),method = 'cubic', fill_value = 0)
                next[next_idx[j]] = griddata(this.points,this.ne[i,j,:],(interp_positions[1,0][next_idx[j]], interp_positions[1,1][next_idx[j]]),method = 'cubic', fill_value = 0)
            # on_grid ne is then calculated by linearly interpolating values between these two planes
            
            this.ne_on_grid[i,...] = prev * interp_positions[1,2,...] + next * interp_positions[0,2,...]
            
            #phi data as well:
            for j in this.planes:
                prev[prev_idx[j]] = griddata(this.points,this.phi[i,j,:],(interp_positions[0,0][prev_idx[j]], interp_positions[0,1][prev_idx[j]]),method = 'cubic', fill_value = 0)
                next[next_idx[j]] = griddata(this.points,this.phi[i,j,:],(interp_positions[1,0][next_idx[j]], interp_positions[1,1][next_idx[j]]),method = 'cubic', fill_value = 0)
            
            this.phi_on_grid[i,...] = prev * interp_positions[1,2,...] + next * interp_positions[0,2,...]
            #this.dne_ad_on_grid[i,...] += griddata(this.points,this.dne_ad[i,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
            #this.nane_on_grid[i,...] += griddata(this.points,this.nane[i,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
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
        

