"""Load XGC output data, interpolate electron density perturbation onto desired Cartesian grid mesh. 
"""
from ...GeneralSettings.UnitSystem import SI
from ...Geometry.Grid import Cartesian2D,Cartesian3D
from ...IO.IO_funcs import parse_num

import numpy as np
import h5py as h5
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import scipy.io.netcdf as nc
import pickle

# some external functions
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
        prevplane[np.nonzero(prevplane == my_xgc.n_plane)] = 0

    return (prevplane,nextplane)
    
    
def find_interp_positions_v2(my_xgc):
    """new version to find the interpolation positions. Using B field information and follows the exact field line.

    argument and return value are the same as find_interp_positions_v1. 
    """

    BR = my_xgc.BR_interp
    BZ = my_xgc.BZ_interp
    BPhi = my_xgc.BPhi_interp

    r3D = my_xgc.grid.r3D
    z3D = my_xgc.grid.z3D
    phi3D = my_xgc.grid.phi3D

    NZ = r3D.shape[0]
    NY = r3D.shape[1]
    NX = r3D.shape[2]

    Rwant = r3D.flatten()
    Zwant = z3D.flatten()
    Phiwant = phi3D.flatten()

    prevplane,nextplane = my_xgc.prevplane.flatten(),my_xgc.nextplane.flatten()

    dPhi = 2*np.pi/my_xgc.n_plane
    
    phi_planes = np.arange(my_xgc.n_plane)*dPhi

    if(my_xgc.CO_DIR):
        phiFWD = np.where(nextplane == 0,np.pi*2 - Phiwant, phi_planes[nextplane]-Phiwant)
        phiBWD = phi_planes[prevplane]-Phiwant
    else:
        phiFWD = phi_planes[nextplane]-Phiwant
        phiBWD = np.where(prevplane ==0,np.pi*2 - Phiwant, phi_planes[prevplane]-Phiwant)

    N_step = 20
    dphi_FWD = phiFWD/N_step
    dphi_BWD = phiBWD/N_step
    R_FWD = np.copy(Rwant)
    R_BWD = np.copy(Rwant)
    Z_FWD = np.copy(Zwant)
    Z_BWD = np.copy(Zwant)
    s_FWD = np.zeros(Rwant.shape)
    s_BWD = np.zeros(Rwant.shape)
    for i in range(N_step):
        
        print 'step {0} started'.format(i)

        RdPhi_FWD = R_FWD*dphi_FWD
        BPhi_FWD = BPhi(Z_FWD,R_FWD)
        dR_FWD = RdPhi_FWD * BR(Z_FWD,R_FWD) / BPhi_FWD
        dZ_FWD = RdPhi_FWD * BZ(Z_FWD,R_FWD) / BPhi_FWD
        s_FWD += np.sqrt(RdPhi_FWD**2 + dR_FWD**2 + dZ_FWD**2)
        R_FWD += dR_FWD
        Z_FWD += dZ_FWD

        print 'forward step completed.'
        
        RdPhi_BWD = R_BWD*dphi_BWD
        BPhi_BWD = BPhi(Z_BWD,R_BWD)
        dR_BWD = RdPhi_BWD * BR(Z_BWD,R_BWD) / BPhi_BWD
        dZ_BWD = RdPhi_BWD * BZ(Z_BWD,R_BWD) / BPhi_BWD
        s_BWD += np.sqrt(RdPhi_BWD**2 + dR_BWD**2 + dZ_BWD**2)
        R_BWD += dR_BWD
        Z_BWD += dZ_BWD
        
        print 'backward step completed.'

    interp_positions = np.zeros((2,3,NZ,NY,NX))

    interp_positions[0,0,...] = Z_BWD.reshape((NZ,NY,NX))
    interp_positions[0,1,...] = R_BWD.reshape((NZ,NY,NX))    
    interp_positions[0,2,...] = (s_BWD/(s_BWD+s_FWD)).reshape((NZ,NY,NX))
    interp_positions[1,0,...] = Z_FWD.reshape((NZ,NY,NX))
    interp_positions[1,1,...] = R_FWD.reshape((NZ,NY,NX))
    interp_positions[1,2,...] = 1-interp_positions[0,2,...]

    return interp_positions
    

def find_interp_positions_v1(my_xgc):
    """Find the interpolation R-Z positions on previous and next planes for 3D mesh  given by my_xgc.grid.

    Argument:
    my_xgc: XGC_Loader object, containing all the detailed information for the XGC output data and desired Cartesian mesh data
    
    returns:
    double array with shape (2,3,NZ,NY,NX), contains the R and Z values on both planes that should be used to interpolate. the last 3 dimensions corresponds to each desired mesh point, the first 2 dimensions contains the 2 pairs of (Z,R,portion) values. The previous plane interpolation point is stored in [0,:,...], with order (Z,R,portion), the next plane point in [1,:,...]

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
                    interp_positions[...,i,j,k] += -1
                else:
                    # first, find the 2 XGC mesh points that are nearest to the desired points, one inside of the flux surface, the other outside.
                    psi_max = np.max(psi)
                    inner_search_region = np.intersect1d( np.where(( np.absolute(psi-psi_want[i,j,k])<= psi_max/10))[0],np.where(( psi-psi_want[i,j,k] <0))[0], assume_unique = True)
                    outer_search_region = np.intersect1d( np.where((psi-psi_want[i,j,k])<=psi_max/10)[0],np.where( (psi-psi_want[i,j,k])>=0 )[0])
    
                    inner_distance = np.sqrt((Rwant[i,j,k]-R[inner_search_region])**2 + (Zwant[i,j,k]-Z[inner_search_region])**2)
                    outer_distance = np.sqrt((Rwant[i,j,k]-R[outer_search_region])**2 + (Zwant[i,j,k]-Z[outer_search_region])**2)

                    #nearest2 contains the index of the 2 nearest mesh points
                    nearest2 = []
                    min1 = np.argmin(inner_distance)
                    nearest2.append( inner_search_region[min1] ) 
                    if(outer_distance.size != 0):
                        min2 = np.argmin(outer_distance)
                        nearest2.append( outer_search_region[min2] )
                    else:
                        INNER_ONLY = True

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
                        Rcur,Zcur = R[ncur],Z[ncur]
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


class XGC_Loader_Error(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class XGC_Loader():
    """Loader for a given set of XGC output files
    """

    def __init__(self,xgc_path,grid,t_start,t_end,dt,dn_amplifier = 1.0, n_cross_section = 1,equilibrium_mesh = '2D',Full_Load = True, Fluc_Only = True):
        """The main caller of all functions to prepare a loaded XGC profile.

        Inputs:
            xgc_path:string, the directory of all the XGC output files
            grid:FPSDP.Geometry.Grid.Cartesian2D or Cartesian3D object, a user defined mesh object, find the corresponding module for details.
            tstart,tend,dt: int, the timesteps used for loading, NOTE: the time series created here MUST be a subseries of the original file numbers.
            dn_amplifier: double, the multiplier used to artificially change the fluctuation level, default to be 1, i.e. use original fluctuation data read from XGC output.
            n_cross_section: int, number of cross sections loaded for each time step, default to be 1, i.e. use only data on(near) number 0 cross-section. NOTE: this number can not be larger than the total cross-section used in the XGC simulation.
            equilibrium_mesh: string, a flag to choose from '2D' or '3D' equilibrium output style. Current FWR3D code read '2D' equilibrium data.
            Full_Load: boolean, A flag for debugging, default to be True, i.e. load all data when initializing, if set to be False, then only constants are set, no loading functions will be called during initialization, programmer can call them one by one afterwards.
            Fluc_Only: boolean, A flag determining fluctuation loading method. Default to be True. Fluc_Only == True uses newer loading method to remove equilibrium relaxation effects. Fluc_Only == False uses old version and load all the calculated density deviations from the equilibrium.
            
        """

        print 'Loading XGC output data'
        
        self.xgc_path = xgc_path
        self.mesh_file = xgc_path + 'xgc.mesh.h5'
        self.bfield_file = xgc_path + 'xgc.bfield.h5'
        self.time_steps = np.arange(t_start,t_end+1,dt)
        self.grid = grid
        self.dn_amplifier = dn_amplifier #
        self.n_cross_section = n_cross_section
        self.unit_file = xgc_path+'units.m'
        self.te_input_file = xgc_path+'te_input.in'
        self.ne_input_file = xgc_path+'ne_input.in'
        self.equilibrium_mesh = equilibrium_mesh
        
        print 'from directory:'+ self.xgc_path

        self.tstep = load_m(self.unit_file)['sml_dt']
        self.dt = self.tstep * dt
        self.t = self.time_steps * self.tstep

        if isinstance(grid, Cartesian2D):
            self.dimension = 2
        elif isinstance(grid,Cartesian3D):
            self.dimension = 3
        else:
            raise XGC_Loader_Error('grid error:grid should be either Cartesian2D or Cartesian3D')


        if (Full_Load):
            if self.dimension == 2:
                print '2D Grid detected.'
                self.load_mesh_2D()
                print 'mesh loaded.'
                self.load_psi_2D()
                print 'psi loaded.'
                self.load_B_2D()
                print 'B loaded.'
                self.load_eq_2D3D()
                print 'equilibrium loaded.'
                if (Fluc_Only):
                    self.load_fluctuations_2D_fluc_only()
                else:
                    self.load_fluctuations_2D_all()
                print 'fluctuations loaded.'
                self.calc_total_ne_2D3D()
                print 'total ne calculated.'
                self.interpolate_all_on_grid_2D()
                print 'quantities interpolated on grid.\n XGC data sucessfully loaded.'
            
            elif self.dimension == 3:
                print '3D grid detected.'
                
                grid.ToCylindrical()
                print 'cynlindrical coordinates created.'
                
                self.load_mesh_psi_3D()
                print 'mesh and psi loaded.'
            
                self.load_B_3D()
                print 'B loaded.'

                self.prevplane,self.nextplane = get_interp_planes(self)
                print 'interpolation planes obtained.'


                self.load_eq_2D3D()
                print 'equlibrium loaded.'
                
                if (Fluc_Only):
                    self.load_fluctuations_3D_fluc_only()
                else:
                    self.load_fluctuations_3D_all()
           
                print 'fluctuations loaded.'

                self.calc_total_ne_2D3D()
                print 'total ne calculated.'

            
                self.interpolate_all_on_grid_3D()
                print 'all quantities interpolated on grid.\n XGC data sucessfully loaded.'

            
            
            
    
    def change_grid(self,grid):
        """change the current grid to another grid,reload all quantities on new grid
        Argument:
        grid: Grid object, Currently must be Cartesian2D or Cartesian3D
        """
        if self.dimension == 2:
            if isinstance(grid,Cartesian2D):
                self.grid = grid
                self.interpolate_all_on_grid_2D()
            elif isinstance(grid,Cartesian3D):
                self.dimension = 3
                print 'Change grid from 2D to 3D, loading all the data files again, please wait...'
                self.grid = grid
                
                grid.ToCylindrical()
                print 'cynlindrical coordinates created.'
                
                self.load_mesh_psi_3D()
                print 'mesh and psi loaded.'

                self.load_B_3D()
                print 'B loaded.'
                
                self.prevplane,self.nextplane = get_interp_planes(self)
                print 'interpolation planes obtained.'


                self.load_eq_2D3D()
                print 'equlibrium loaded.'

                if (Fluc_Only):
                    self.load_fluctuations_3D_fluc_only()
                else:
                    self.load_fluctuations_3D_all()
                    
                print 'fluctuations loaded.'

                self.calc_total_ne_2D3D()
                print 'total ne calculated.'
            
                self.interpolate_all_on_grid_3D()
                print 'all quantities interpolated on grid.\n XGC data sucessfully loaded.'
           
            else:
                raise XGC_Loader_Error( 'NOT VALID GRID, please use either Cartesian3D or Cartesian2D grids.Grid NOT changed.')
                
        else:
            if isinstance(grid,Cartesian3D):
                self.grid = grid
                self.interpolate_all_on_grid_3D()
            elif isinstance(grid,Cartesian2D):
                self.dimension = 2
                print 'Changing from 3D to 2D grid, load all the data files again, please wait...'
                self.grid = grid
                
                self.load_mesh_2D()
                print 'mesh loaded.'
                self.load_psi_2D()
                print 'psi loaded.'
                self.load_B_2D()
                print 'B loaded.'
                self.load_eq_2D3D()
                print 'equilibrium loaded.'
                if (Fluc_Only):
                    self.load_fluctuations_2D_fluc_only()
                else:
                    self.load_fluctuations_2D_all()
                print 'fluctuations loaded.'
                self.calc_total_ne_2D3D()
                print 'total ne calculated.'
                self.interpolate_all_on_grid_2D()
                print 'quantities interpolated on grid.\n XGC data sucessfully loaded.'
            else:
                raise XGC_Loader_Error( 'NOT VALID GRID, please use either Cartesian3D or Cartesian2D grids.Grid NOT changed.')
                
    
    def load_mesh_2D(self):
        """Load the R-Z data

         
        """
        mesh = h5.File(self.mesh_file,'r')
        RZ = mesh['coordinates']['values']
        Rpts =RZ[:,0]
        Zpts = RZ[:,1]
        mesh.close()
        self.points = np.array([Zpts,Rpts]).transpose()
        self.mesh = {'R':Rpts, 'Z':Zpts}
        return 0

    def load_psi_2D(self):
        """Load psi data

        spline over Z,R
        Note that choose R as the 2nd variable in order to store it in the fastest dimension later
        """
        mesh = h5.File(self.mesh_file,'r')
        self.psi = mesh['psi'][...]
        self.psi_interp = CloughTocher2DInterpolator(self.points, self.psi,fill_value=np.max(self.psi))
        mesh.close()
        return 0

    def load_mesh_psi_3D(self):
        """load R-Z mesh and psi values, then create map between each psi value and the series of points on that surface, calculate the arc length table.
        """
        mesh = h5.File(self.mesh_file,'r')
        RZ = mesh['coordinates']['values']
        Rpts =RZ[:,0]
        Zpts = RZ[:,1]
        self.points = np.array([Zpts,Rpts]).transpose()
        self.mesh = {'R':Rpts, 'Z':Zpts}

        self.nextnode = mesh['nextnode'][...]
        
        self.prevnode = np.zeros(self.nextnode.shape)
        for i in range(len(self.nextnode)):
            prevnodes = np.nonzero(self.nextnode == i)[0]
            if( len(prevnodes)>0 ):
                self.prevnode[i] = prevnodes[0]
            else:
                self.prevnode[i] = -1
        
        self.psi = mesh['psi'][...]
        self.psi_interp = CloughTocher2DInterpolator(self.points, self.psi, fill_value=np.max(self.psi))

        mesh.close()

        # get the number of toroidal planes from fluctuation data file
        fluc_file0 = self.xgc_path + 'xgc.3d.' + str(self.time_steps[0]).zfill(5)+'.h5'
        fmesh = h5.File(fluc_file0,'r')
        self.n_plane = fmesh['dpot'].shape[1]

        fmesh.close()
        
        
        
        

    def load_B_2D(self):
        """Load equilibrium magnetic field data

        B_total is interpolated over Z,R plane on given 2D Cartesian grid
        """
        B_mesh = h5.File(self.bfield_file,'r')
        self.B = B_mesh['node_data[0]']['values']
        self.B_total = np.sqrt(self.B[:,0]**2 + self.B[:,1]**2 + self.B[:,2]**2)
        self.B_interp = CloughTocher2DInterpolator(self.points, self.B_total, fill_value = np.sign(self.B_total[0])*np.min(np.absolute(self.B_total))) 
        B_mesh.close()
        return 0

    def load_B_3D(self):
        """Load equilibrium magnetic field data

        B_total is interpolated over Z,R plane on given 3D Cartesian grid, since B_0 is assumed symmetric along toroidal direction
        """
        B_mesh = h5.File(self.bfield_file,'r')
        B = B_mesh['node_data[0]']['values']
        self.BR = B[:,0]
        self.BZ = B[:,1]
        self.BPhi = B[:,2]

        self.BR_interp = CloughTocher2DInterpolator(self.points, self.BR, fill_value = 0)
        self.BZ_interp = CloughTocher2DInterpolator(self.points, self.BZ, fill_value = 0)
        self.BPhi_interp = CloughTocher2DInterpolator(self.points, self.BPhi, fill_value = np.sign(self.BPhi[0])*np.min(np.absolute(self.BPhi)))

        
        B_mesh.close()

        #If toroidal B field is positive, then field line is going in the direction along which plane number is increasing.
        self.CO_DIR = (np.sign(self.BPhi[0]) > 0)
        return 0

    def load_fluctuations_2D_all(self):
        """Load non-adiabatic electron density and electrical static potential fluctuations
        the mean value of these two quantities on each time step is also calculated.

        Note that for full-F runs, the purturbed electron density includes both turbulent fluctuations and equilibrium relaxation, this loading method doesn't differentiate them and will read all of them.

        """
        if(self.HaveElectron):
            self.nane = np.zeros( (self.n_cross_section,len(self.time_steps),len(self.mesh['R'])) )
            self.nane_bar = np.zeros((len(self.time_steps)))
        self.phi = np.zeros((self.n_cross_section,len(self.time_steps),len(self.mesh['R'])))
        self.phi_bar = np.zeros((len(self.time_steps)))
        for i in range(len(self.time_steps)):
            flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[i]).zfill(5)+'.h5'
            fluc_mesh = h5.File(flucf,'r')
            if (i == 0):
                self.n_plane = fluc_mesh['dpot'].shape[1]
                dn = int(self.n_plane/self.n_cross_section)
                self.planes = np.arange(self.n_cross_section) * dn
                
            for j in range(self.n_cross_section):
                self.phi[j,i] += np.swapaxes(fluc_mesh['dpot'][...][:,self.planes[j]],0,1)
                self.phi_bar[i] += np.mean(fluc_mesh['dpot'][...])
                self.phi[j,i] -= self.phi_bar[i]

                if(self.HaveElectron):
                    self.nane[j,i] += np.swapaxes(fluc_mesh['eden'][...][:,self.planes[j]],0,1)
                    self.nane_bar[i] += np.mean(fluc_mesh['eden'][...])
                    self.nane[j,i] -= self.nane_bar[i]
            fluc_mesh.close()


        
        
        return 0


    def load_fluctuations_2D_fluc_only(self):
        """Load non-adiabatic electron density and electrical static potential fluctuations
        the mean value of these two quantities on each time step is also calculated.

        Since XGC-1 has full-f capability, the deviation from input equilibrium is not only fluctuations induced by turbulences, but also relaxation of the equilibrium. Since we are only interested in the former part, we need to screen out the latter effect.[*] The way of doing this is as follows:
        Since the relaxation of equilibrium should be the same across the whole flux surface, it naturally is the same along toroidal direction. Given that no large n=0 mode exists in the turbulent spectra, the toroidal average of the calculated delta-n will mainly be the equilibrium relaxation. However, this effect might be important, so we keep the time-averaged relaxation effect to add it into the input equilibrium. The final formula for density fluctuation (as well as potential fluctuation) is then:
            n_tilde = delta_n - <delta_n>_zeta ,
        where delta_n is the calculated result, and <...>_zeta denotes average in toroidal direction.
        and the effective equilibrium is given by: n0_eff = n0 + <delta_n>_zeta_t ,
        where n0 is the input equilibrium, and <...>_zeta_t denotes average over both toroidal and time.
        """
        #first we load one file to obtain the total plane number used in the simulation
        flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[0]).zfill(5)+'.h5'
        fluc_mesh = h5.File(flucf,'r')        
        self.n_plane = fluc_mesh['dpot'].shape[1]
        dn = int(self.n_plane/self.n_cross_section)#dn is the increment between two chosen cross-sections, if total chosen number is greater than total simulation plane number, an error will occur.
        self.planes = np.arange(self.n_cross_section)*dn

        if(self.HaveElectron):
            self.nane = np.zeros( (self.n_cross_section,len(self.time_steps),len(self.mesh['R'])))
            nane_all = np.zeros( (self.n_plane, len(self.time_steps), len(self.mesh['R']) ) )
        self.phi = np.zeros((self.n_cross_section,len(self.time_steps),len(self.mesh['R'])))
        phi_all = np.zeros((self.n_plane,len(self.time_steps),len(self.mesh['R'])))

        #after initializing the arrays to hold the data, we load the data from the first chosen step
        for j in range(self.n_plane):
            phi_all[j,0] += np.swapaxes(fluc_mesh['dpot'][...][:,j],0,1)
            if(self.HaveElectron):
                nane_all[j,0] += np.swapaxes(fluc_mesh['eden'][...][:,j],0,1)
        fluc_mesh.close()
        
        for i in range(1,len(self.time_steps)):
            #now we load all the data from rest of the chosen time steps. 
            flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[i]).zfill(5)+'.h5'
            fluc_mesh = h5.File(flucf,'r')
                
            for j in range(self.n_plane):
                phi_all[j,i] += np.swapaxes(fluc_mesh['dpot'][...][:,j],0,1)
                if(self.HaveElectron):
                    nane_all[j,i] += np.swapaxes(fluc_mesh['eden'][...][:,j],0,1)
            fluc_mesh.close()



        #now, all data is ready, we need to pick the chosen cross sections and do some post process. Since XGC-1 has full-f capability, the deviation from input equilibrium is not only fluctuations induced by turbulences, but also relaxation of the equilibrium. Since we are only interested in the former part, we need to screen out the latter effect.[*] The way of doing this is as follows:
        # Since the relaxation of equilibrium should be the same across the whole flux surface, it naturally is the same along toroidal direction. Given that no large n=0 mode exists in the turbulent spectra, the toroidal average of the calculated delta-n will mainly be the equilibrium relaxation. However, this effect might be important, so we keep the time-averaged relaxation effect to add it into the input equilibrium. The final formula for density fluctuation (as well as potential fluctuation) is then:
        #   n_tilde = delta_n - <delta_n>_zeta , where delta_n is the calculated result, and <...>_zeta denotes average in toroidal direction.
        # and the effective equilibrium is given by:
        #   n0_eff = n0 + <delta_n>_zeta_t , where n0 is the input equilibrium, and <...>_zeta_t denotes average over both toroidal and time.

        # first, we calculate the n_tilde, note that we have adiabatic and non-adiabatic parts. The adiabatic part is given by the potential, and will be calculated later in calc_total_ne_2D3D.
        phi_avg_tor = np.average(phi_all,axis = 0)
        if(self.HaveElectron):
            nane_avg_tor = np.average(nane_all,axis=0)
        for j in range(self.n_cross_section):
            self.phi[j,:,:] = phi_all[self.planes[j],:,:] - phi_avg_tor[:,:]
            if(self.HaveElectron):
                self.nane[j,:,:] = nane_all[self.planes[j],:,:] - nane_avg_tor[:,:]

        # then, we add the averaged relaxation modification to the input equilibrium

        self.ne0[:] += np.average(phi_avg_tor,axis = 0)
        if(self.HaveElectron):
            self.ne0[:] += np.average(nane_avg_tor,axis = 0)
        
        
        return 0

    def load_fluctuations_3D_all(self):
        """Load non-adiabatic electron density and electrical static potential fluctuations for 3D mesh. The required planes are calculated and stored in sorted array. fluctuation data on each plane is stored in the same order.
        Note that for full-F runs, the purturbed electron density includes both turbulent fluctuations and equilibrium relaxation, this loading method doesn't differentiate them and will read all of them.
        
        the mean value of these two quantities on each time step is also calculated.
        for multiple cross-section runs, data is stored under each center_plane index.
        """
        #similar to the 2D case, we first read one file to determine the total toroidal plane number in the simulation
        flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[0]).zfill(5)+'.h5'
        fluc_mesh = h5.File(flucf,'r')

        self.planes = np.unique(np.array([np.unique(self.prevplane),np.unique(self.nextplane)]))
        self.planeID = {self.planes[i]:i for i in range(len(self.planes))} #the dictionary contains the positions of each chosen plane, useful when we want to get the data on a given plane known only its plane number in xgc file.
        if(self.HaveElectron):
            self.nane = np.zeros( (self.n_cross_section,len(self.time_steps),len(self.planes),len(self.mesh['R'])) )
            self.nane_bar = np.zeros((len(self.time_steps)))
        self.phi = np.zeros( (self.n_cross_section,len(self.time_steps),len(self.planes),len(self.mesh['R'])) )
        self.phi_bar = np.zeros((len(self.time_steps)))
        for i in range(len(self.time_steps)):
            flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[i]).zfill(5)+'.h5'
            fluc_mesh = h5.File(flucf,'r')

            if(i==0):
                #self.n_plane = fluc_mesh['dpot'].shape[1]
                dn = int(self.n_plane/self.n_cross_section)
                self.center_planes = np.arange(self.n_cross_section)*dn
            for j in range(self.n_cross_section):
                self.phi[j,i] += np.swapaxes(fluc_mesh['dpot'][...][:,(self.center_planes[j] + self.planes)%self.n_plane],0,1)
                self.phi_bar[i] += np.mean(fluc_mesh['dpot'][...])
                self.phi[j,i] -= self.phi_bar[i]
                if(self.HaveElectron):
                    self.nane[i] += np.swapaxes(fluc_mesh['eden'][...][:,(self.center_planes[j] + self.planes)%self.n_plane],0,1)
                    self.nane_bar[i] += np.mean(fluc_mesh['eden'][...])
                    self.nane[i] -= self.nane_bar[i]
            fluc_mesh.close()
            
        return 0

    def load_fluctuations_3D_fluc_only(self):
        """Load non-adiabatic electron density and electrical static potential fluctuations for 3D mesh. The required planes are calculated and stored in sorted array. fluctuation data on each plane is stored in the same order. 
        the mean value of these two quantities on each time step is also calculated.

        similar to the 2D case, we take care of the equilibrium relaxation contribution. See details in the comments in 2D loading function.
        
        for multiple cross-section runs, data is stored under each center_plane index.
        """
        #similar to the 2D case, we first read one file to determine the total toroidal plane number in the simulation
        flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[0]).zfill(5)+'.h5'
        fluc_mesh = h5.File(flucf,'r')

        self.n_plane = fluc_mesh['dpot'].shape[1]
        dn = int(self.n_plane/self.n_cross_section)
        self.center_planes = np.arange(self.n_cross_section)*dn

        self.planes = np.unique(np.array([np.unique(self.prevplane),np.unique(self.nextplane)]))
        self.planeID = {self.planes[i]:i for i in range(len(self.planes))} #the dictionary contains the positions of each chosen plane, useful when we want to get the data on a given plane known only its plane number in xgc file.

        #initialize the arrays
        if(self.HaveElectron):
            self.nane = np.zeros( (self.n_cross_section,len(self.time_steps),len(self.planes),len(self.mesh['R'])) )
            nane_all = np.zeros((self.n_plane,len(self.time_steps),len(self.mesh['R'])))
        self.phi = np.zeros( (self.n_cross_section,len(self.time_steps),len(self.planes),len(self.mesh['R'])) )
        phi_all = np.zeros((self.n_plane,len(self.time_steps),len(self.mesh['R'])))

        #load all the rest of the files
        for i in range(1,len(self.time_steps)):
            flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[i]).zfill(5)+'.h5'
            fluc_mesh = h5.File(flucf,'r')
            for j in range(self.n_plane):
                phi_all[j,i] += np.swapaxes(fluc_mesh['dpot'][...][:,j],0,1)
                if(self.HaveElectron):
                    nane_all[j,i] += np.swapaxes(fluc_mesh['eden'][...][:,j],0,1)
            fluc_mesh.close()


        #similar to the 2D case, we take care of the equilibrium relaxation contribution. See details in the comments in 2D loading function.
        
        phi_avg_tor = np.average(phi_all,axis = 0)
        if self.HaveElectron:
            nane_avg_tor = np.average(nane_all,axis=0)

        for j in range(self.n_cross_section):
            self.phi[j,...] = np.swapaxes(phi_all[(self.center_planes[j] + self.planes)%self.n_plane,:,:],0,1) - phi_avg_tor[:,np.newaxis,:]
            if self.HaveElectron:
                self.nane[j,...] = np.swapaxes(nane_all[(self.center_planes[j] + self.planes)%self.n_plane,:,:],0,1) - nane_avg_tor[:,np.newaxis,:]

        self.ne0[:] += np.average(phi_avg_tor,axis=0)
        if self.HaveElectron:
            self.ne0[:] += np.average(nane_avg_tor,axis=0)
            
        return 0
    
    def load_eq_2D3D(self):
        """Load equilibrium profiles, including ne0, Te0
        """
        eqf = self.xgc_path + 'xgc.oneddiag.h5'
        eq_mesh = h5.File(eqf,'r')
        eq_psi = eq_mesh['psi_mks'][:]

        #sometimes eq_psi is stored as 2D array, which has time series infomation. For now, just use the time step 1 psi array as the unchanged array. NEED TO BE CHANGED if equilibrium psi mesh is changing over time.
        n_psi = eq_psi.shape[-1]
        eq_psi = eq_psi.flatten()[0:n_psi] #pick up the first n psi values.

        eq_ti = eq_mesh['i_perp_temperature_1d'][0,:]
        self.ti0_sp = interp1d(eq_psi,eq_ti,bounds_error = False,fill_value = 0)
        if('e_perp_temperature_1d' in eq_mesh.keys() ):
            #simulation has electron dynamics
            self.HaveElectron = True
            eq_te = eq_mesh['e_perp_temperature_1d'][0,:]
            eq_ne = eq_mesh['e_gc_density_1d'][0,:]
            self.te0_sp = interp1d(eq_psi,eq_te,bounds_error = False,fill_value = 0)
            self.ne0_sp = interp1d(eq_psi,eq_ne,bounds_error = False,fill_value = 0)

        else:
            self.HaveElectron = False
            self.load_eq_tene_nonElectronRun() 
        eq_mesh.close()
        
        self.te0 = self.te0_sp(self.psi)
        self.ne0 = self.ne0_sp(self.psi)

    def load_eq_tene_nonElectronRun(self):
        """For ion only silumations, te and ne are read from simulation input files.

        Add Attributes:
        te0_sp: interpolator for Te_0 on psi
        ne0_sp: interpolator for ne_0 on psi
        """
        te_fname = 'xgc.Te_prof.prf'
        ne_fname = 'xgc.ne_prof.prf'

        psi_te, te = np.genfromtxt(te_fname,skip_header = 1,skip_footer = 1,unpack = True)
        psi_ne, ne = np.genfromtxt(ne_fname,skip_header = 1,skip_footer = 1,unpack = True)

        psi_x = load_m(self.xgc_path + 'units.m')['psi_x']

        psi_te *= psi_x
        psi_ne *= psi_x

        self.te0_sp = interp1d(psi_te,te,bounds_error = False,fill_value = 0)
        self.ne0_sp = interp1d(psi_ne,ne,bounds_error = False,fill_value = 0)
        

    def calc_total_ne_2D3D(self):
        """calculate the total electron density in raw XGC grid points
        """
        ne0 = self.ne0
        te0 = self.te0
        inner_idx = np.where(te0>0)[0]
        self.dne_ad = np.zeros(self.phi.shape)
        self.dne_ad[...,inner_idx] += ne0[inner_idx] * self.phi[...,inner_idx] /self.te0[inner_idx]
        ad_valid_idx = np.where(np.absolute(self.dne_ad)<= np.absolute(ne0))

        self.ne = np.zeros(self.dne_ad.shape)
        self.ne += ne0[:]
        self.ne[ad_valid_idx] += self.dne_ad[ad_valid_idx]*self.dn_amplifier
        if(self.HaveElectron):
            na_valid_idx = np.where(np.absolute(self.nane)<= np.absolute(self.ne))
            self.ne[na_valid_idx] += self.nane[na_valid_idx]*self.dn_amplifier

    def interpolate_all_on_grid_2D(self):
        """ create all interpolated quantities on given grid.
        """
        R2D = self.grid.R2D
        Z2D = self.grid.Z2D

        #psi on grid
        self.psi_on_grid = self.psi_interp(Z2D,R2D)

        #B on grid
        self.B_on_grid = self.B_interp(Z2D,R2D)

        #Te and Ti on grid
        self.te_on_grid = self.te0_sp(self.psi_on_grid)
        self.ti_on_grid = self.ti0_sp(self.psi_on_grid)
        
        #total ne and fluctuations
        self.ne_on_grid = np.zeros((self.n_cross_section,len(self.time_steps),R2D.shape[0],R2D.shape[1]))
        self.phi_on_grid = np.zeros(self.ne_on_grid.shape)
        self.dne_ad_on_grid = np.zeros(self.ne_on_grid.shape)
        if self.HaveElectron:
            self.nane_on_grid = np.zeros(self.ne_on_grid.shape)

        self.ne0_on_grid = griddata(self.points,self.ne0[:],(Z2D,R2D),method = 'linear',fill_value = 0)

        for i in range(self.ne.shape[0]):
            for j in range(self.ne.shape[1]):

                self.ne_on_grid[i,j,...] += griddata(self.points,self.ne[i,j,:],(Z2D,R2D),method = 'linear', fill_value = 0)
                self.phi_on_grid[i,j,...] += griddata(self.points,self.phi[i,j,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
                self.dne_ad_on_grid[i,j,...] += griddata(self.points,self.dne_ad[i,j,:],(Z2D,R2D),method = 'cubic',fill_value = 0)
                if(self.HaveElectron):
                    self.nane_on_grid[i,j,...] += griddata(self.points,self.nane[i,j,:],(Z2D,R2D),method = 'cubic',fill_value = 0)

        self.interp_check() # after the interpolation, check if the perturbations are interpolated within a reasonable error


    def interpolate_all_on_grid_3D(self):
        """ create all interpolated quantities on given 3D grid.

        equilibrium mesh: '2D' uses 2D equilibrium mesh, '3D' uses full 3D mesh.
        """

        r3D = self.grid.r3D
        z3D = self.grid.z3D
        phi3D = self.grid.phi3D
        
        if(self.equilibrium_mesh == '3D'):
            #interpolation on 3D mesh: (currently not used in FWR3D)
            #psi on grid
            self.psi_on_grid = self.psi_interp(z3D,r3D)
        
            #B Field on grid, right now, BPhi,BZ, and BR are directly used.
            self.BX_on_grid = -self.BPhi_interp(z3D,r3D)*np.sin(phi3D)+self.BR_interp(z3D,r3D)*np.cos(phi3D)
            self.BY_on_grid = self.BZ_interp(z3D,r3D)
            self.BZ_on_grid = -self.BR_interp(z3D,r3D)*np.sin(phi3D)-self.BPhi_interp(z3D,r3D)*np.cos(phi3D)
            self.B_on_grid = np.sqrt(self.BX_on_grid**2 + self.BY_on_grid**2 + self.BZ_on_grid**2)


            #Te and Ti on grid
            self.te_on_grid = self.te0_sp(self.psi_on_grid)
            self.ti_on_grid = self.ti0_sp(self.psi_on_grid)

            #ne0 on grid
            self.ne0_on_grid = self.ne0_sp(self.psi_on_grid)
        elif(self.equilibrium_mesh == '2D'):
            #interpolation on 2D mesh: (used in FWR3D, the FWR3D code will then rotate the whole equilibrium to get the values on 3D mesh.)
            x1D = self.grid.X1D
            y1D = self.grid.Y1D
            x2D = np.zeros((self.grid.NY,self.grid.NX)) + x1D[np.newaxis,:]
            y2D = np.zeros(x2D.shape) + y1D[:,np.newaxis]

            #psi on 2D grid
            self.psi_on_grid = self.psi_interp(y2D,x2D)

            #B field on grid
            self.BX_on_grid = self.BR_interp(y2D,x2D)
            self.BY_on_grid = self.BZ_interp(y2D,x2D)
            self.BZ_on_grid = self.BPhi_interp(y2D,x2D)

            self.B_on_grid = np.sqrt(self.BX_on_grid**2 + self.BY_on_grid**2 + self.BZ_on_grid**2)
            
            #Te and Ti on grid
            self.te_on_grid = self.te0_sp(self.psi_on_grid)
            self.ti_on_grid = self.ti0_sp(self.psi_on_grid)

            #ne0 on grid
            self.ne0_on_grid = self.ne0_sp(self.psi_on_grid)            
        
        
        #ne fluctuations on 3D grid
        self.dne_ad_on_grid = np.zeros((self.n_cross_section,len(self.time_steps),r3D.shape[0],r3D.shape[1],r3D.shape[2]))
        if self.HaveElectron:
             self.nane_on_grid = np.zeros(self.dne_ad_on_grid.shape)

        
        
        interp_positions = find_interp_positions_v2(self)

        for k in range(self.n_cross_section):
            print 'center plane {0}.'.format(self.center_planes[k])
            for i in range(len(self.time_steps)):
                print 'time step {0}'.format(self.time_steps[i])
                #for each time step, first create the 2 arrays of quantities for interpolation
                prev = np.zeros( (self.grid.NZ,self.grid.NY,self.grid.NX) )
                next = np.zeros(prev.shape)
            
                #create index dictionary, for each key as plane number and value the corresponding indices where the plane is used as previous or next plane.
                prev_idx = {}
                next_idx = {}
                for j in range(len(self.planes)):
                    prev_idx[j] = np.where(self.prevplane == self.planes[j] )
                    next_idx[j] = np.where(self.nextplane == self.planes[j] )

                #now interpolate adiabatic ne on each toroidal plane for the points using it as previous or next plane.
                for j in range(len(self.planes)):
                    if(prev[prev_idx[j]].size != 0):
                        prev[prev_idx[j]] = griddata(self.points,self.dne_ad[k,i,j,:],(interp_positions[0,0][prev_idx[j]], interp_positions[0,1][prev_idx[j]]),method = 'linear', fill_value = 0)
                    if(next[next_idx[j]].size != 0):
                        next[next_idx[j]] = griddata(self.points,self.dne_ad[k,i,j,:],(interp_positions[1,0][next_idx[j]], interp_positions[1,1][next_idx[j]]),method = 'linear', fill_value = 0)
                # on_grid adiabatic ne is then calculated by linearly interpolating values between these two planes
            
                self.dne_ad_on_grid[k,i,...] = prev * interp_positions[1,2,...] + next * interp_positions[0,2,...]

                if self.HaveElectron:
                    #non-adiabatic ne data as well:
                    for j in range(len(self.planes)):
                        prev[prev_idx[j]] = griddata(self.points,self.nane[k,i,j,:],(interp_positions[0,0][prev_idx[j]], interp_positions[0,1][prev_idx[j]]),method = 'cubic', fill_value = 0)
                        next[next_idx[j]] = griddata(self.points,self.nane[k,i,j,:],(interp_positions[1,0][next_idx[j]], interp_positions[1,1][next_idx[j]]),method = 'cubic', fill_value = 0)
            
                    self.nane_on_grid[k,i,...] = prev * interp_positions[1,2,...] + next * interp_positions[0,2,...]


    def interp_check(self, tol = 0.2, toroidal_cross = 0, time = 0):
        """check if the interpolation has been carried out correctly

        use the on_grid data to interpolate onto original data locations, and compare the result with the original data, if the relative difference is larger than the tolerance, report the locations and the error.

        arguments:
            tol: double, the maximum tolerable error(relative to the original value). Default to be 20 percent.
            toroidal_cross: int, toroidal cross section number to be used, default to be 0.
            time: int, time step to be used, default to be 0.
            

        return:
            tuple (check_pass,adiabatic_error_info,non_adiabatic_error_info)
                check_pass: boolean, True if no violation founded.
                adiabatic_error_info: tuple (R,Z,error), R,Z contains original coordinates where violations occur, error is the relative error.
                non_adiabatic_error_info: tuple(R,Z,error), same meaning as in adiabatic case.
        """

        if(self.dimension == 3):
            print "3D check is currently not available. The main difficulty is that the original data is on 2D plane, but the on grid data is on 3D mesh, which doesn't guarantee a X-Y slice to be correspondent to one original plane. This makes comparison between the interpolated data and the original data impractical. A new way to test the 3D interpolation is needed.\n Caution: NO interp_check done!"
            return

        Rmin = self.grid.R1D[0]
        Rmax = self.grid.R1D[-1]
        Zmin = self.grid.Z1D[0]
        Zmax = self.grid.Z1D[-1]

        R = self.mesh['R']
        Z = self.mesh['Z']

        #find the index where original R,Z are inside the interpolated grid

        R_in_flag = np.logical_and( R>Rmin,R<Rmax )
        Z_in_flag = np.logical_and( Z>Zmin,Z<Zmax )
        idx = np.where(np.logical_and(R_in_flag,Z_in_flag))

        R_in = R[idx]
        Z_in = Z[idx]

        dne_ad_in = self.dne_ad[toroidal_cross,time,idx][0]
        nane_in = self.nane[toroidal_cross,time,idx][0]

        dne_ad_back_interp = RectBivariateSpline(self.grid.Z1D,self.grid.R1D,self.dne_ad_on_grid[toroidal_cross,time,:,:])
        nane_back_interp = RectBivariateSpline(self.grid.Z1D,self.grid.R1D,self.nane_on_grid[toroidal_cross,time,:,:])

        #compare the points
        dne_ad_interp = dne_ad_back_interp.ev(Z_in,R_in)
        dne_ad_error = np.abs( (dne_ad_in - dne_ad_interp) / dne_ad_interp )

        nane_interp = nane_back_interp.ev(Z_in,R_in)
        nane_error = np.abs( (nane_in - nane_interp) / nane_interp )
        
        #check if the error is within tolerance
        dne_ad_exceed_tol_idx = np.where(dne_ad_error > tol)
        nane_exceed_tol_idx = np.where(nane_error > tol)

        #Sometimes, tolerance is exceeded because the original data is too small. Check for these cases, an warning is raised only if the original data is greater than 1e-5 * the max value in original data.

        dne_ad_true_violate_idx = np.where(np.abs(dne_ad_in[dne_ad_exceed_tol_idx]) > np.max(np.abs(dne_ad_in))*1e-2 )
        nane_true_violate_idx = np.where(np.abs(nane_in[nane_exceed_tol_idx]) > np.max(np.abs(nane_in))*1e-2 )

        #report the violation locations
        check_pass = True
        ad_vio_count = len(dne_ad_in[dne_ad_exceed_tol_idx][dne_ad_true_violate_idx])
        na_vio_count = len(nane_in[nane_exceed_tol_idx][nane_true_violate_idx])
        if(ad_vio_count != 0):
            check_pass = False
            print "Warning: adiabatic electron density interpolation inaccurate! {0} points violated. Check the  returned value of method interp_check for detailed information.".format(ad_vio_count)
        if(na_vio_count != 0):
            check_pass = False
            print "Warning: non-adiabatic electron density interpolation inaccurate! {0} points violated. Check the  returned value of method interp_check for detailed information.".format(na_vio_count)
        
        #return the tuple containing all information

        adiabatic_error_info = (R_in[dne_ad_exceed_tol_idx][dne_ad_true_violate_idx],Z_in[dne_ad_exceed_tol_idx][dne_ad_true_violate_idx],dne_ad_in[dne_ad_exceed_tol_idx][dne_ad_true_violate_idx],dne_ad_interp[dne_ad_exceed_tol_idx][dne_ad_true_violate_idx],dne_ad_error[dne_ad_exceed_tol_idx][dne_ad_true_violate_idx])
        non_adiabatic_error_info = (R_in[nane_exceed_tol_idx][nane_true_violate_idx],Z_in[nane_exceed_tol_idx][nane_true_violate_idx],nane_in[nane_exceed_tol_idx][nane_true_violate_idx],nane_interp[nane_exceed_tol_idx][nane_true_violate_idx],nane_error[nane_exceed_tol_idx][nane_true_violate_idx])

        return (check_pass,adiabatic_error_info,non_adiabatic_error_info)
            

        
        
        
    def save(self,fname = 'xgc_profile.sav'):
        """save the original and interpolated electron density fluctuations and useful equilibrium quantities to a local .npz file

        for 2D instances,The arrays saved are:
            X1D: the 1D array of coordinates along R direction (major radius)
            Y1D: the 1D array of coordinates along Z direction (vertical)
            X_origin: major radius coordinates on original scattered grid
            Y_origin: vertical coordinates on original scattered grid
           
            dne_ad: the adiabatic electron density perturbation, in shape (NY,NX), where NX,NY are the dimensions of X1D, Y1D respectively
            nane: (if non-adiabatic electron is on in XGC simulation)the non-adiabatic electron density perturbation. same shape as dne_ad

            dne_ad_org: the adiabatic electron density perturbation on original grid
            nane_org: the non-adiabatic electron density perturbation on original grid
            
            ne0: the equilibrium electron density.
            Te0: equilibrium electron temperature
            Ti0: equilibrium ion temperature
            B0: equilibrium magnetic field (toroidal)
            
        for 3D instances, in addition to the arrays above, one coordinate is also saved:
            Z1D: 1D coordinates along R cross Z direction.

            BX: radial magnetic field
            BY: vertical magnetic field
        """
        file_name = self.xgc_path + fname
        saving_dic = {
            'dne_ad':self.dne_ad_on_grid,
            'ne0':self.ne0_on_grid,
            'dne_ad_org':self.dne_ad,
            'X_origin':self.mesh['R'],
            'Y_origin':self.mesh['Z'],
            'Te0':self.te_on_grid,
            'Ti0':self.ti_on_grid,
            'psi':self.psi_on_grid
            }
        if (self.HaveElectron):
            saving_dic['nane'] = self.nane_on_grid
            saving_dic['nane_org'] = self.nane
        if (self.dimension == 2):
            saving_dic['X1D'] = self.grid.R1D
            saving_dic['Y1D'] = self.grid.Z1D
            saving_dic['B0'] = self.B_on_grid
        else:
            saving_dic['X1D'] = self.grid.X1D
            saving_dic['Y1D'] = self.grid.Y1D
            saving_dic['Z1D'] = self.grid.Z1D
            saving_dic['B0'] = self.BZ_on_grid
            saving_dic['BX'] = self.BX_on_grid
            saving_dic['BY'] =  self.BY_on_grid
        np.savez(file_name,**saving_dic)

    def load(self, filename = 'dne_file.sav'):
        """load the previously saved xgc profile data file.
        The geometry information needs to be the same, otherwise an error will be raised.
        WARNING: Currently no serious checking is performed. The user is responsible to make sure the XGC_Loader object is initialized properly to load the corresponding saving file. 
        """

        if 'npz' not in filename:
            filename += '.npz'
        nefile = np.load(filename)
        if 'Z1D' in nefile.files:
            dimension = 3
        else:
            dimension = 2
        if(dimension != self.dimension):
            raise XGC_Loader_Error('Geometry incompatible! Trying to load {0}d data onto {1}d grid.\nMake sure the geometry setup is the same as the data file.'.format(dimension,self.dimension))

        #======== NEED MORE DETAILED GEOMETRY CHECKING HERE! CURRENT VERSION DOESN'T GUARANTEE SAME GRID. ERRORS WILL OCCUR WHEN READ SAVED FILE WITH A DIFFERENT GRID.
        #=============================================#

        self.mesh = {'R':nefile['X_origin'],'Z':nefile['Y_origin']}
        self.dne_ad = nefile['dne_ad_org']
        self.ne0_on_grid = nefile['ne0']
        self.dne_ad_on_grid = nefile['dne_ad']

        self.ne_on_grid = self.ne0_on_grid[np.newaxis,np.newaxis,:,:] + self.dne_ad_on_grid

        if 'nane' in nefile.files:
            self.HaveElectrons = True
            self.nane = nefile['nane_org']
            self.nane_on_grid = nefile['nane']
            self.ne_on_grid += self.nane_on_grid

        self.psi_on_grid = nefile['psi']
        self.te_on_grid = nefile['Te0']
        self.ti_on_grid = nefile['Ti0']

        if dimension == 2:
            self.B_on_grid = nefile['B0']
        else:
            self.BZ_on_grid = nefile['B0']
            self.BX_on_grid = nefile['BX']
            self.BY_on_grid = nefile['BY']
            self.B_on_grid = np.sqrt(self.BX_on_grid**2 + self.BY_on_grid**2 + self.BZ_on_grid**2) 
        
        


    def get_frequencies(self):
        """calculate the relevant frequencies along the plasma midplane,return them as a dictionary

        arguments:
            time_eval: boolean, a flag for time evolution. Default to be False

        return: dictionary contains all the frequency arrays
            keywords: 'f_pi','f_pe','f_ci','f_ce','f_uh','f_lh','f_lx','f_ux'

        using formula in Page 28, NRL_Formulary, 2011 and Section 2-3, Eqn.(7-9), Waves in Plasmas, Stix.
        """
        if(isinstance(self.grid,Cartesian2D) ):#2D mesh
            Zmid = (self.grid.NZ-1)/2
            ne = self.ne_on_grid[:,:,Zmid,:]*1e-6 #convert into cgs unit
            ni = ne #D plasma assumed,ignore the impurity. 
            mu = 2 # mu=m_i/m_p, in D plasma, it's 2
            B = self.B_on_grid[Zmid,:]*1e5 #convert to cgs unit
        else:#3D mesh
            Ymid = (self.grid.NY-1)/2
            Zmid = (self.grid.NZ-1)/2
            ne = self.ne_on_grid[:,:,Zmid,Ymid,:]*1e-6
            ni = ne
            mu = 2
            if(self.equilibrium_mesh == '3D'):
                B = self.B_on_grid[Zmid,Ymid,:]*1e5
            else:
                B = self.B_on_grid[Ymid,:]*1e5

        f_pi = 2.1e2*mu**(-0.5)*np.sqrt(ni)
        f_pe = 8.98e3*np.sqrt(ne)
        f_ci = 1.52e3/mu*B
        f_ce = 2.8e6*B

        f_uh = np.sqrt(f_ce**2+f_pe**2)
        f_lh = np.sqrt( 1/ ( 1/(f_ci**2+f_pi**2) + 1/(f_ci*f_ce) ) )

        f_ux = 0.5*(f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))
        f_lx = 0.5*(-f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))

        return {'f_pi':f_pi,
                'f_pe':f_pe,
                'f_ci':f_ci,
                'f_ce':f_ce,
                'f_uh':f_uh,
                'f_lh':f_lh,
                'f_ux':f_ux,
                'f_lx':f_lx}


    def cdf_output(self,output_path,eq_file = 'equilibrium.cdf',filehead = 'fluctuation'):
        """
        Wrapper for cdf_output_2D and cdf_output_3D.
        Determining 2D/3D by checking the grid property.
            
        """

        if ( isinstance(self.grid,Cartesian2D) ):
            self.cdf_output_2D(output_path,filehead)
        elif (isinstance(self.grid, Cartesian3D)):
            self.cdf_output_3D(output_path,eq_file,filehead)
        else:
            raise XGC_Loader_Error('Wrong grid type! Grid should either be Cartesian2D or Cartesian3D.') 
        

    def cdf_output_2D(self,output_path,filehead='fluctuation'):
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
        for i in range(self.n_cross_section):
            for j in range(len(self.time_steps)):
            
                fname = file_start + str(self.time_steps[j])+'_'+str(i) + '.cdf'
                f = nc.netcdf_file(fname,'w')
                f.createDimension('z_dim',self.grid.NZ)
                f.createDimension('r_dim',self.grid.NR)

                rr = f.createVariable('rr','d',('r_dim',))
                rr[:] = self.grid.R1D[:]
                zz = f.createVariable('zz','d',('z_dim',))
                zz[:] = self.grid.Z1D[:]
                rr.units = zz.units = 'Meter'

                bb = f.createVariable('bb','d',('z_dim','r_dim'))
                bb[:,:] = self.B_on_grid[:,:]
                bb.units = 'Tesla'
                
                ne = f.createVariable('ne','d',('z_dim','r_dim'))
                ne[:,:] = self.ne_on_grid[i,j,:,:]
                ne.units = 'per cubic meter'

                te = f.createVariable('te','d',('z_dim','r_dim'))
                te[:,:] = self.te_on_grid[:,:]/1000
                te.units = 'keV'
                
                ti = f.createVariable('ti','d',('z_dim','r_dim'))
                ti[:,:] = self.ti_on_grid[:,:]/1000
                ti.units = 'keV'

                f.close()

    def cdf_output_3D(self,output_path = './',eq_filename = 'equilibrium3D.cdf',flucfilehead='fluctuation'):
        """write out cdf files for FWR3D code to use

        Arguments:
        output_path: string, the full path to put the output files
        eq_filename: string, the file name for the 2D equilibrium output
        flucfilehead: string, the starting string of all 3D fluctuation filenames

        CDF file format:

        Equilibrium file:
        
        Dimensions:
        nr: int, number of grid points in radial direction.
        nz: int, number of grid points in vetical direction
        
        Variables:
        rr: 1D array, coordinates in radial direction, in m
        zz: 1D array, coordinates in vertical direction, in m
        bb: 2D array, toroidal magnetic field on grids, in Tesla, shape in (nz,nr)
        br: 2D array, magnetic field in radial direction, in Tesla
        bz: 2D array, magnetic field in vertical direction, in Tesla
        ne: 2D array, total electron density on grids, in cm^-3
        ti: 2D array, total ion temperature, in keV
        te: 2D array, total electron temperature, in keV

        Fluctuation files:

        Dimensions:
        nx: number of grid points in radial direction
        ny: number of grid points in vertical direction
        nz: number of grid points in horizontal direction

        Variables:
        xx: 1D array, coordinates in radial direction
        yy: 1D array, coordinates in vertical direction 
        zz: 1D array, coordinates in horizontal direction
        dne: 3D array, (nz,ny,nx), adiabatic electron density perturbation, real value 
        """
        eqfname = output_path + eq_filename
        f = nc.netcdf_file(eqfname,'w')
        f.createDimension('nz',self.grid.NY)
        f.createDimension('nr',self.grid.NX)
        
        rr = f.createVariable('rr','d',('nr',))
        rr[:] = self.grid.X1D[:]
        zz = f.createVariable('zz','d',('nz',))
        zz[:] = self.grid.Y1D[:]
        rr.units = zz.units = 'm'
        
        bb = f.createVariable('bb','d',('nz','nr'))
        bb[:,:] = self.BZ_on_grid[:,:]
        bb.units = 'Tesla'

        br = f.createVariable('br','d',('nz','nr'))
        br[:,:] = self.BX_on_grid[:,:]
        br.units = 'Tesla'

        bz = f.createVariable('bz','d',('nz','nr'))
        bz[:,:] = self.BY_on_grid[:,:]
        bz.units = 'Tesla'       
        
        ne = f.createVariable('ne','d',('nz','nr'))
        ne[:,:] = self.ne0_on_grid[:,:]
        ne.units = 'm^-3'
        
        te = f.createVariable('te','d',('nz','nr'))
        te[:,:] = self.te_on_grid[:,:]/1000
        te.units = 'keV'
        
        ti = f.createVariable('ti','d',('nz','nr'))
        ti[:,:] = self.ti_on_grid[:,:]/1000
        ti.units = 'keV'
        
        f.close()

        
        file_start = output_path + flucfilehead
        for j in range(self.n_cross_section):
            for i in range(len(self.time_steps)):
                fname = file_start + str(self.time_steps[i]) +'_'+ str(j)+ '.cdf'
                f = nc.netcdf_file(fname,'w')
                f.createDimension('nx',self.grid.NX)
                f.createDimension('ny',self.grid.NY)
                f.createDimension('nz',self.grid.NZ)

                xx = f.createVariable('xx','d',('nx',))
                xx[:] = self.grid.X1D[:]
                yy = f.createVariable('yy','d',('ny',))
                yy[:] = self.grid.Y1D[:]
                zz = f.createVariable('zz','d',('nz',))
                zz[:] = self.grid.Z1D[:]            
                xx.units = yy.units = zz.units = 'm'
            
                dne = f.createVariable('dne','d',('nz','ny','nx'))
                dne.units = 'm^-3'
                if(not self.HaveElectron):
                    dne[:,:,:] = self.dne_ad_on_grid[j,i,:,:,:]*self.dn_amplifier          
                else:
                    dne[:,:,:] = (self.dne_ad_on_grid[j,i,:,:,:] + self.nane_on_grid[j,i,:,:,:])*self.dn_amplifier
                f.close()
    
        

def get_ref_pos(my_xgc,freqs,mode = 'O'):
    """estimates the O-mode reflection position in R direction for given frequencies.

    Input:
        my_xgc:XGC_loader object containing the profile and fluctuation information.
        freqs:sequence of floats, all the probing frequencies in GHz.
        mode: 'O' or 'X', indication of the wave polarization.
    return:
        2D array of floats,in the shape of (time_steps,freqs). R coordinates of all the estimated reflection position on mid plane, for each timestep and frequency.
    """
    if(isinstance(my_xgc.grid,Cartesian2D)):
        R = my_xgc.grid.R1D
    else:
        R = my_xgc.grid.X1D

    plasma_freqs = my_xgc.get_frequencies()
    
    if(mode == 'O'):
        cutoff = plasma_freqs['f_pe']*1e-9
    elif(mode == 'X'):
        cutoff = plasma_freqs['f_ux']*1e-9 #convert into GHz
    else:
        print 'mode should be either O or X!'
        raise
    ref_idx = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))
    ref_pos = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))

    for j in range(cutoff.shape[0]):
        for k in range(cutoff.shape[1]):
            for i in range(len(freqs)):
    
                ref_idx[j,k,i] = np.max(np.where(cutoff[j,k,:] > freqs[i])[0])#The right most index where the wave has been cutoff

                #linearly interpolate the wave frequency to the cutoff frequency curve, to find the reflected location
                f1 = cutoff[j,k,ref_idx[j,k,i]+1]
                f2 = cutoff[j,k,ref_idx[j,k,i]]
                f3 = freqs[i]
            
                R1 = R[ref_idx[j,k,i]+1]
                R2 = R[ref_idx[j,k,i]]
            
                ref_pos[j,k,i] = R2 + (f2-f3)/(f2-f1)*(R1-R2)
        
    
    return ref_pos


