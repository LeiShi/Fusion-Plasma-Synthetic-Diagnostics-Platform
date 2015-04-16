
"""Load XGC output data and interpolate the data
   Copy of load_XGC_profile but remove all the useless stuff for BES
"""

from ...IO.IO_funcs import parse_num

import numpy as np
import h5py as h5
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import RectBivariateSpline
from load_XGC_profile import load_m, XGC_Loader_Error
import scipy.io.netcdf as nc
import pickle

def get_interp_planes_BES(my_xgc,phi3D):
    """Get the plane numbers used for interpolation for each point 
       phi3D should be between 0 and 2pi
    """

    # angle between two planes
    dPHI = 2 * np.pi / my_xgc.n_plane
    # angle of each planes
    phi_planes = np.arange(my_xgc.n_plane)*dPHI
    # previous/next plane depends on the direction of the field
    if(my_xgc.CO_DIR):
        # find next plane and previous plane
        nextplane = np.searchsorted(phi_planes,phi3D,side = 'right')
        prevplane = nextplane - 1
        # change the highest value for the periodicity
        nextplane[np.nonzero(nextplane == my_xgc.n_plane)] = 0
    else:
        prevplane = np.searchsorted(phi_planes,phi3D,side = 'right')
        nextplane = prevplane - 1
        prevplane[np.nonzero(prevplane == my_xgc.n_plane)] = 0

    return (prevplane,nextplane)


class XGC_Loader_BES():
    """Loader for a given set of XGC output files
    """

    def __init__(self,xgc_path,t_start,t_end,dt,limits,N_field):
        """The main caller of all functions to prepare a loaded XGC 
           profile for the BES diagnostic.

        Inputs:
            xgc_path       -- string, the directory of all the XGC output files
            tstart,tend,dt -- int, the timesteps used for loading, 
                              NOTE: the time series created here MUST be a subseries
                              of the original file numbers.
            limits         -- array containing the limits of the value computed [[Xmin,
                              Xmax],[Ymin,Ymax],[Zmin,Zmax]]
            N_field        -- number of step for the field line interpolation
        """

        print 'Loading XGC output data'
        
        self.xgc_path = xgc_path
        self.mesh_file = xgc_path + 'xgc.mesh.h5'
        self.bfield_file = xgc_path + 'xgc.bfield.h5'
        self.time_steps = np.arange(t_start,t_end+1,dt)
        self.unit_file = xgc_path+'units.m'
        self.te_input_file = xgc_path+'te_input.in'
        self.ne_input_file = xgc_path+'ne_input.in'
        
        print 'from directory:'+ self.xgc_path
        self.unit_dic = load_m(self.unit_file)
        self.tstep = self.unit_dic['sml_dt']*self.unit_dic['diag_1d_period']
        self.dt = self.tstep * dt
        self.t = self.time_steps * self.tstep
        self.current = 0 # index of the current time step
        # limits of the mesh
        self.Xmin = limits[0,0]
        self.Xmax = limits[0,1]
        self.Ymin = limits[1,0]
        self.Ymax = limits[1,1]
        self.Zmin = limits[2,0]
        self.Zmax = limits[2,1]

        # limits in tokamak coordinates
        self.Rmin = np.sqrt(self.Xmin**2 + self.Ymin**2)
        self.Rmax = np.sqrt(self.Xmax**2 + self.Ymax**2)
        phi = np.array([[self.Xmin,self.Ymin],[self.Xmin,self.Ymax],
                        [self.Xmax,self.Ymin],[self.Xmax,self.Ymax]])

        phi = np.arctan2(phi[:,1],phi[:,0])
        # put the value between [0,2pi]
        tempmin = np.min(phi)
        tempmax = np.max(phi)
        phi[phi<0] += 2*np.pi
        self.Phimin = np.min(phi)
        self.Phimax = np.max(phi)

        self.N_step = N_field
        self.load_mesh_psi_3D()
        print 'mesh and psi loaded.'
        
        self.load_B_3D()
        print 'B loaded.'

        # BES use only a small area, so it should be less
        # than 2-3 different planes, therefore 10 should
        # be enough
        phi = np.linspace(tempmin,tempmax,10)
        phi[phi<0] += 2*np.pi
        self.refprevplane,self.refnextplane = get_interp_planes_BES(self,phi)
        print 'interpolation planes obtained.'
        
        self.load_eq_2D3D()
        print 'equlibrium loaded.'

        self.load_next_time_step(False)

    def load_next_time_step(self,increase=True):
        """ Load all the quantities for the next time step.
            The old quantities are overwritten
            increase is used for the first time step that
            need to be keepen at 0

            can be easily change to load any time, but for BES
            loading them in order is enough
        """
        if increase:
            self.current += 1
        if self.current >= len(self.time_steps):
            raise XGC_Loader_Error('The time step is bigger than the one\
            requested')
        

        self.load_fluctuations_3D_all()        
        print 'fluctuations loaded.'
        
        self.calc_total_ne_2D3D()
        print 'total ne calculated.'

        self.compute_interpolant()
        print 'interpolant computed'
        
            
    def load_mesh_psi_3D(self):
        """load R-Z mesh and psi values, then create map between each psi 
           value and the series of points on that surface, calculate the arc length table.
        """
        # open the file
        mesh = h5.File(self.mesh_file,'r')
        RZ = mesh['coordinates']['values']
        Rpts =RZ[:,0]
        # remove the points outside the window
        self.ind = (Rpts > self.Rmin) & (Rpts < self.Rmax)
        Zpts = RZ[:,1]
        self.ind = self.ind & (Zpts > self.Zmin) & (Zpts < self.Zmax)
        self.ind = self.ind
        Rpts = Rpts[self.ind]
        Zpts = Zpts[self.ind]
        self.points = np.array([Zpts,Rpts]).transpose()

        print 'Keep: ',str(Rpts.shape[0]),'Points on a total of: '\
            ,str(self.ind.shape[0])

        self.psi = mesh['psi'][self.ind]
        # psi interpolant
        self.psi_interp = CloughTocher2DInterpolator(
            self.points, self.psi, fill_value=np.max(self.psi))

        mesh.close()

        # get the number of toroidal planes from fluctuation data file
        fluc_file0 = self.xgc_path + 'xgc.3d.' + str(self.time_steps[0]).zfill(5)+'.h5'
        fmesh = h5.File(fluc_file0,'r')
        self.n_plane = fmesh['dpot'].shape[1]

        fmesh.close()
        
        

    def load_B_3D(self):
        """Load equilibrium magnetic field data

        B_total is interpolated over Z,R plane on given 3D Cartesian grid, 
        since B_0 is assumed symmetric along toroidal direction
        """
        B_mesh = h5.File(self.bfield_file,'r')
        B = B_mesh['node_data[0]']['values']
        # keep only the values around the diagnostic
        BR = B[self.ind,0]
        BZ = B[self.ind,1]
        BPhi = B[self.ind,2]

        # interpolant of each direction of the field
        # use np.inf as a flag for outside points,
        # deal with them later in interpolation function
        self.BR_interp = CloughTocher2DInterpolator(
            self.points, BR, fill_value = np.inf)

        self.BZ_interp = CloughTocher2DInterpolator(
            self.points, BZ, fill_value = np.inf)

        self.BPhi_interp = CloughTocher2DInterpolator(
            self.points, BPhi, fill_value = np.sign(BPhi[0])*np.min(np.absolute(BPhi)))

        
        B_mesh.close()

        #If toroidal B field is positive, then field line is going in
        # the direction along which plane number is increasing.
        self.CO_DIR = (np.sign(BPhi[0]) > 0)
        return 0


    def load_fluctuations_3D_all(self):
        """Load non-adiabatic electron density and electrical static 
           potential fluctuations for 3D mesh.
           The required planes are calculated and stored in sorted array.
           fluctuation data on each plane is stored in the same order.
        Note that for full-F runs, the purturbed electron density 
        includes both turbulent fluctuations and equilibrium relaxation,
        this loading method doesn't differentiate them and will read all of them.
        
        the mean value of these two quantities on each time step is also calculated.
        for multiple cross-section runs, data is stored under each center_plane index.
        """
        #similar to the 2D case, we first read one file to determine the total
        # toroidal plane number in the simulation
        flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[0]).zfill(5)+'.h5'
        fluc_mesh = h5.File(flucf,'r')

        # list of all planes of interest
        self.planes = np.unique(np.array([np.unique(self.refprevplane),
                                          np.unique(self.refnextplane)]))
        self.planeID = {self.planes[i]:i for i in range(len(self.planes))}
        #the dictionary contains the positions of each chosen plane,
        # useful when we want to get the data on a given plane known only its plane number in xgc file.
        # the last dimension is for the mesh position
        self.nane = np.zeros( (len(self.planes),
                               len(self.points[:,0])) )
        
        self.nani = np.zeros( (len(self.planes),
                               len(self.points[:,0])) )

        self.phi = np.zeros( (len(self.planes),
                              len(self.points[:,0])) )
        
        flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[self.current]).zfill(5)+'.h5'
        fluc_mesh = h5.File(flucf,'r')
        
        self.phi += np.swapaxes(
            fluc_mesh['dpot'][self.ind,:][:,(self.planes)%self.n_plane],0,1)
        
        self.nane += np.swapaxes(
            fluc_mesh['eden'][self.ind,:][:,(self.planes)%self.n_plane],0,1)
        
        self.nani += np.swapaxes(
            fluc_mesh['iden'][self.ind,:][:,(self.planes)%self.n_plane],0,1)
        
        fluc_mesh.close()
            
        return 0

    def load_eq_2D3D(self):
        """Load equilibrium profiles
        """
        eqf = self.xgc_path + 'xgc.oneddiag.h5'
        eq_mesh = h5.File(eqf,'r')
        eq_psi = eq_mesh['psi_mks'][:]

        #sometimes eq_psi is stored as 2D array, which has time series infomation.
        # For now, just use the time step 1 psi array as the unchanged array.
        # NEED TO BE CHANGED if equilibrium psi mesh is changing over time.
        n_psi = eq_psi.shape[-1]
        eq_psi = eq_psi.flatten()[0:n_psi] #pick up the first n psi values.

        eq_ti = eq_mesh['i_perp_temperature_1d'][0,:]
        eq_ni = eq_mesh['i_gc_density_1d'][0,:]
        ni_min = np.min(eq_ni)

        # interpolant for the equilibrium of the ions
        self.ti0_sp = interp1d(eq_psi,eq_ti,bounds_error = False,fill_value = 0)
        self.ni0_sp = interp1d(eq_psi,eq_ni,bounds_error = False,fill_value = ni_min/10)
        if('e_perp_temperature_1d' in eq_mesh.keys() ):
            #simulation has electron dynamics
            eq_te = eq_mesh['e_perp_temperature_1d'][0,:]
            eq_ne = eq_mesh['e_gc_density_1d'][0,:]
            te_min = np.min(eq_te)
            ne_min = np.min(eq_ne)
            self.te0_sp = interp1d(eq_psi,eq_te,bounds_error = False,fill_value = te_min/2)
            self.ne0_sp = interp1d(eq_psi,eq_ne,bounds_error = False,fill_value = ne_min/10)

        else:
            self.load_eq_tene_nonElectronRun() 
        eq_mesh.close()

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
        """calculate the total electron and ion density in raw XGC grid points
        """
        # temperature and density (equilibrium) on the psi mesh
        te0 = self.te0_sp(self.psi)
        ne0 = self.ne0_sp(self.psi)

        ti0 = self.ti0_sp(self.psi)
        ni0 = self.ni0_sp(self.psi)

        inner_idx = np.where(te0>0)[0]
        self.dne_ad = np.zeros(self.phi.shape)
        self.dne_ad[...,inner_idx] += ne0[inner_idx]*self.phi[...,inner_idx]/te0[inner_idx]
        ad_valid_idx = np.where(np.absolute(self.dne_ad)<= np.absolute(ne0))

        self.ne = np.zeros(self.dne_ad.shape)
        self.ne += ne0[:]
        self.ne[ad_valid_idx] += self.dne_ad[ad_valid_idx]
        na_valid_idx = np.where(np.absolute(self.nane)<= np.absolute(self.ne))
        self.ne[na_valid_idx] += self.nane[na_valid_idx]

        inner_idx = np.where(ti0>0)[0]
        self.dni_ad = np.zeros(self.phi.shape)
        self.dni_ad[...,inner_idx] += ni0[inner_idx]*self.phi[...,inner_idx]/ti0[inner_idx]
        ad_valid_idx = np.where(np.absolute(self.dni_ad)<= np.absolute(ni0))


        self.ni = np.zeros(self.dni_ad.shape)
        self.ni += ni0[:]
        self.ni[ad_valid_idx] += self.dni_ad[ad_valid_idx]
        
        na_valid_idx = np.where(np.absolute(self.nani)<= np.absolute(self.ni))
        self.ni[na_valid_idx] += self.nani[na_valid_idx]

        # clear some space
        del self.nani,self.dni_ad,self.nane,self.dne_ad

    def compute_interpolant(self):
        """ Compute the interpolant for the ion and electron
            density
        """
        # list of interpolant
        self.ni_interp = []
        self.ne_interp = []
        for j in range(len(self.planes)):
            # computation of interpolant
            self.ni_interp.append(
                CloughTocher2DInterpolator(self.points,self.ni[j,:],fill_value=0.0))
            
            self.ne_interp.append(
                CloughTocher2DInterpolator(self.points,self.ne[j,:],fill_value=0.0))


    def find_interp_positions(self,r,z,phi,prev_,next_):
        """new version to find the interpolation positions.
        Using B field information and follows the exact field line.
        
        argument and return value are the same as find_interp_positions_v1. 
        """
        prevplane,nextplane = prev_.flatten(),next_.flatten()

        # angle between two planes of the simulation
        dPhi = 2*np.pi/self.n_plane

        # angle of the planes
        phi_planes = np.arange(self.n_plane)*dPhi

        # the previous/next planes depend on the direction of the field
        if(self.CO_DIR):
            # distance (angle) between the phiw wanted and the closest planes
            phiFWD = np.where(nextplane == 0,np.pi*2 - phi, phi_planes[nextplane]-phi)
            phiBWD = phi_planes[prevplane]-phi
        else:
            phiFWD = phi_planes[nextplane]-phi
            phiBWD = np.where(prevplane ==0,np.pi*2 - phi, phi_planes[prevplane]-phi)

        # number of step for each direction
        N_FWD = int(np.max(np.abs(phiFWD*self.N_step/dPhi)))
        N_BWD = int(np.max(np.abs(phiBWD*self.N_step/dPhi)))

        # angle between two steps
        dphi_FWD = phiFWD/N_FWD
        dphi_BWD = phiBWD/N_BWD
        R_FWD = np.copy(r)
        R_BWD = np.copy(r)
        Z_FWD = np.copy(z)
        Z_BWD = np.copy(z)
        s_FWD = np.zeros(r.shape)
        s_BWD = np.zeros(r.shape)

        # forward step
        for i in range(N_FWD):
            RdPhi_FWD = R_FWD*dphi_FWD
            BPhi_FWD = self.BPhi_interp(Z_FWD,R_FWD)
            dR_FWD = RdPhi_FWD * self.BR_interp(Z_FWD,R_FWD) / BPhi_FWD
            dZ_FWD = RdPhi_FWD * self.BZ_interp(Z_FWD,R_FWD) / BPhi_FWD
            
            ind = np.where(dR_FWD == np.inf)[0]
            #when the point gets outside of the XGC mesh, set BR,BZ to zero.
            dR_FWD[ind] = 0.0
            ind = np.where(dZ_FWD == np.inf)[0]
            dZ_FWD[ind] = 0.0
            
            s_FWD += np.sqrt(RdPhi_FWD**2 + dR_FWD**2 + dZ_FWD**2)
            R_FWD += dR_FWD
            Z_FWD += dZ_FWD

        # backward step
        for i in range(N_BWD):
            RdPhi_BWD = R_BWD*dphi_BWD
            BPhi_BWD = self.BPhi_interp(Z_BWD,R_BWD)
            dR_BWD = RdPhi_BWD * self.BR_interp(Z_BWD,R_BWD) / BPhi_BWD
            dZ_BWD = RdPhi_BWD * self.BZ_interp(Z_BWD,R_BWD) / BPhi_BWD
            
            ind = np.where(dR_BWD == np.inf)[0]
            dR_BWD[ind] = 0.0
            ind = np.where(dZ_BWD == np.inf)[0]
            dZ_BWD[ind] = 0.0
            
            s_BWD += np.sqrt(RdPhi_BWD**2 + dR_BWD**2 + dZ_BWD**2)
            R_BWD += dR_BWD
            Z_BWD += dZ_BWD
            
        interp_positions = np.zeros((2,3,r.shape[0]))
        
        interp_positions[0,0,...] = Z_BWD
        interp_positions[0,1,...] = R_BWD
        interp_positions[0,2,...] = (s_BWD/(s_BWD+s_FWD))
        interp_positions[1,0,...] = Z_FWD
        interp_positions[1,1,...] = R_FWD
        interp_positions[1,2,...] = 1-interp_positions[0,2,...]

        return interp_positions


    def interpolate_data(self,pos,timesteps,quant,eq):
        """ Interpolate the data to the position wanted
        
            Arguments:
            pos       --  1D array with (X,Y,Z) or 2D array with (X,Y,Z)
                          as the second index
            timesteps --  list of timesteps
            quant     --  list containing the wanted quantities (see data class
                          for more information)
            eq        --  choice between equilibrium or exact quantities
        """
        if (timesteps is not self.current) and (eq is False):
            print 'Maybe an error is made, the requested time step is not the same than the XGC one'
        

        # compute the coordinate in the torroidal system
        r = np.sqrt(np.sum(pos[...,0:2]**2,axis=-1)).flatten()
        z = pos[...,2].flatten()
        phi = np.arctan2(pos[...,1],pos[...,0]).flatten()
        # goes into the interval [0,2pi]
        phi[phi<0] += 2*np.pi
        # get the planes for theses points
        prevplane,nextplane = get_interp_planes_BES(self,phi)

        # check if asking for densities
        ni_bool = 'ni' in quant
        ne_bool = 'ne' in quant

        #psi on grid
        psi = self.psi_interp(z,r)
        # check if want equilibrium data
        if eq:
            #ne0 on grid
            if ne_bool:
                ne = self.ne0_sp(psi)
            if ni_bool:
                ni = self.ni0_sp(psi)
        #Te and Ti on grid
        if 'Te' in quant:
            Te = self.te0_sp(psi)
        if 'Ti' in quant:
            Ti = self.ti0_sp(psi)

                
        if not eq:
            if ne_bool:
                ne = np.zeros(r.shape[0])
            if ni_bool:
                ni = np.zeros(r.shape[0])
            interp_positions = self.find_interp_positions(r,z,phi,prevplane,nextplane)

            #create index dictionary, for each key as plane number and
            # value the corresponding indices where the plane is used as previous or next plane.
            prev_idx = {}
            next_idx = {}
            for j in range(len(self.planes)):
                prev_idx[j] = np.where(prevplane == self.planes[j] )
                next_idx[j] = np.where(nextplane == self.planes[j] )

            if ne_bool:
                #for each time step, first create the 2 arrays of quantities for interpolation
                prevn = np.zeros(r.shape)
                nextn = np.zeros(prevn.shape)
                                    
                for j in range(len(self.planes)):
                    # interpolation on the poloidal planes
                    prevn[prev_idx[j]] = self.ne_interp[j](
                        interp_positions[0,0][prev_idx[j]], interp_positions[0,1][prev_idx[j]])
                    
                    nextn[next_idx[j]] = self.ne_interp[j](
                        interp_positions[1,0][next_idx[j]], interp_positions[1,1][next_idx[j]])
                # interpolation along the field line
                
                ne = prevn * interp_positions[1,2,...] + nextn * interp_positions[0,2,...]
                # if ne is in the box and nan => outside the simulation data => outside
                # of the tokamak
                ne[np.isnan(ne) & (r < self.Rmax) & (r > self.Rmin) &
                   (z < self.Zmax) & (z > self.Zmin)] = 0.0
                if np.isnan(ne).any():
                    print 'r',r
                    print 'z',z
                    print 'phi',phi
                    print 'prev',prevn
                    print 'next',nextn
                    print interp_positions
                    print 'inside',ne[np.isnan(ne)]
                    
            """   NOW WE WORK WITH IONS   """
                    
            if ni_bool:
                #for each time step, first create the 2 arrays of quantities for interpolation
                prevn = np.zeros(r.shape)
                nextn = np.zeros(prevn.shape)
                
                
                for j in range(len(self.planes)):
                    # interpolation on the poloidal planes
                    prevn[prev_idx[j]] = self.ni_interp[j](
                        interp_positions[0,0][prev_idx[j]], interp_positions[0,1][prev_idx[j]])
                    
                    nextn[next_idx[j]] = self.ni_interp[j](
                        interp_positions[1,0][next_idx[j]], interp_positions[1,1][next_idx[j]])
                # interpolation along the field line
                ni = prevni * interp_positions[1,2,...] + nextni * interp_positions[0,2,...]

        ret = ()
        # put the data in the good order
        for i in quant:
            if i is 'ne':
                ret += (ne,)
            elif i is 'ni':
                ret += (ni,)
            elif i is 'Ti':
                ret += (Ti,)
            elif i is 'Te':
                ret += (Te,)
            else:
                raise XGC_Loader_Error('{} is not a valid value in the evaluation of XGC data'.format(i))
        return ret

        
    def save(self,fname = 'xgc_profile.sav'):
        """save the original and interpolated electron density fluctuations 
           and useful equilibrium quantities to a local .npz file

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
        raise NameError('Has not been updated for the BES loader')
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
        raise NameError('Has not been updated for the BES loader')

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


    def cdf_output(self,output_path,eq_file = 'equilibrium.cdf',filehead = 'fluctuation',WithBp=True):
        """
        Wrapper for cdf_output_2D and cdf_output_3D.
        Determining 2D/3D by checking the grid property.
            
        """
        raise NameError('Has not been updated for the BES loader')

    
        self.cdf_output_3D(output_path,eq_file,filehead,WithBp)
        


    def cdf_output_3D(self,output_path = './',eq_filename = 'equilibrium3D.cdf',flucfilehead='fluctuation',WithBp=True):
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
        bb: 2D array, total magnetic field on grids, in Tesla, shape in (nz,nr)
        bpol: 2D array, poloidal magnetic field on grids, in Tesla
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
        raise NameError('Has not been updated for the BES loader')

        eqfname = output_path + eq_filename
        f = nc.netcdf_file(eqfname,'w')
        f.createDimension('nz',self.grid.NY)
        f.createDimension('nr',self.grid.NX)
        
        rr = f.createVariable('rr','d',('nr',))
        rr[:] = self.grid.X1D[:]
        zz = f.createVariable('zz','d',('nz',))
        zz[:] = self.grid.Y1D[:]
        rr.units = zz.units = 'm'

        bp = np.sqrt(self.BX_on_grid[:,:]**2 + self.BY_on_grid[:,:]**2)
        
        bb = f.createVariable('bb','d',('nz','nr'))
        bb[:,:] = np.sqrt(bp**2 + self.BZ_on_grid[:,:]**2)
        bb.units = 'Tesla'

        bpol = f.createVariable('bpol','d',('nz','nr'))
        if(WithBp):        
            bpol[:,:] = bp[:,:]
        else:
            bpol[:,:] = np.zeros_like(bp)
        bpol.units = 'Tesla'  
        
        b_r = f.createVariable('b_r','d',('nz','nr'))
        b_r[:,:] = self.BX_on_grid[:,:]
        
        b_phi = f.createVariable('b_phi','d',('nz','nr'))
        b_phi[:,:] = -self.BZ_on_grid[:,:]
        
        b_z = f.createVariable('b_z','d',('nz','nr'))
        b_z[:,:] = self.BY_on_grid[:,:]
        
        
        
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

        if(not self.Equilibrium_Only):
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
                        dne[:,:,:] = self.dne_ad_on_grid[j,i,:,:,:]
                    else:
                        dne[:,:,:] = (self.dne_ad_on_grid[j,i,:,:,:] + self.nane_on_grid[j,i,:,:,:])
                    f.close()
    
        



