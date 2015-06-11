
"""Module of the XGC loader for the BES synthetic diagnostic.

It reads the data from the simulation and remove the points not used for the diagnostics (in order to keep the used memory at a low level).
Do an interpolation along the B-field.
This file is based on an other code written by Lei Shi (:download:`code <../../../../FPSDP/Plasma/XGC_Profile/load_XGC_profile.py>`).
The orginal code was doing a 3D mesh and interpolate the data from the simulation on this one, now, the code is computing one
time step at a time and interpolate only the value asked by the user.
The other difference is in the interpolation along the field line, the code now try to a number of step on each side that depends
on the distance to the previous/next plane.

"""

from FPSDP.IO.IO_funcs import parse_num

import numpy as np
import h5py as h5
from scipy.interpolate import splrep, splev
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from load_XGC_profile import load_m, XGC_Loader_Error
from FPSDP.Maths.RungeKutta import runge_kutta_explicit

import scipy.io.netcdf as nc


def get_interp_planes_local(my_xgc,phi3D):
    """Get the plane numbers used for the interpolation of each point.

    :param my_xgc: (:class:`XGC_Loader_BES`) XGC loader
    :param np.array[N] phi3D: Phi in the tokamak coordinate (should be between 0 and 2pi)

    :returns: Index of the two closest planes from phi3D
    :rtype: tuple(prev=np.array[N],next=np.array[N])
    """

    temp = np.copy(phi3D)
    # angle between two planes
    dPHI = 2 * np.pi / my_xgc.n_plane
    # angle of each planes
    phi_planes = np.arange(my_xgc.n_plane)*dPHI+my_xgc.shift
    # previous/next plane depends on the direction of the field

    temp [temp < np.min(phi_planes)] += 2*np.pi
    temp[temp > np.max(phi_planes)] -= 2*np.pi
    if(my_xgc.CO_DIR):
        # find next plane and previous plane
        nextplane = np.searchsorted(phi_planes,temp,side = 'right')
        prevplane = nextplane - 1
        # change the highest value for the periodicity
        nextplane[np.nonzero(nextplane == my_xgc.n_plane)] = 0
    else:
        prevplane = np.searchsorted(phi_planes,temp,side = 'right')
        nextplane = prevplane - 1
        prevplane[np.nonzero(prevplane == my_xgc.n_plane)] = 0

    return (prevplane,nextplane)


class XGC_Loader_local():
    """Loader classe for a local diagnostics and without a grid.

    The idea of this loader is to compute one time step at a time and calling the function 
    :func:`load_next_time_step` at the end of the time step that have two different ways to work.
    The first one is to load the next time step and the second one to give the index of the time step.
    The function :func:`interpolate_data` is used to obtain the data from any position.

    A try to limit the size of the data loaded has been made and still can be access by given a 2D array
    for the parameter limits, but it still need to be checked and will only work if the number of toroidal planes
    in the output of XGC1 is big enough (the field line integration need to stay is the same size of the plasma if the integration
    is made from one plane to the other).
    The second case is just done by giving phi min/max and will just load the planes that are concerned.
   
    :param str xgc_path: Name of the directory containing the data
    :param int t_start: Time step at which starting the diagnostics
    :param int t_end: Time step at which stoping the diagnostics
    :param int dt: Interval between two time step that the diagnostics should compute
    :param np.array[...,2] limits: Mesh limits for the diagnostics. Two different cases can happen.\
    The first is that we give an np.array[2,2], in this case, X/Y_min and X/Y_max are given (first index is for X,Y).\
    The second case is that we give a np.array[3,2], in this case,\
    the first index corresponds to the coordinates X,Y,Z and the second one to min/max
    :param float dphi: Size of the step for the field line integration (in radian)
    :param float shift: Shift for phi (default value assumed that plane number 0 is at phi=0)
    :param str kind: Order of the interpolation method (linear or cubic))

    For more detail about the shift, look at the code in :func:`get_interp_planes_local <FPSDP.Plasma.XGC_Profile.load_XGC_local.get_interp_planes_local>`
    """

    def __init__(self,xgc_path,t_start,t_end,dt,limits,dphi,shift=0,kind='linear'):
        """Copy all the input values and call all the functions that compute the equilibrium and the first
        time step.
        """

        print 'Loading XGC output data'
        
        self.xgc_path = xgc_path
        self.mesh_file = xgc_path + 'xgc.mesh.h5'
        self.bfield_file = xgc_path + 'xgc.bfield.h5'
        self.time_steps = np.arange(t_start,t_end+1,dt)
        self.unit_file = xgc_path+'units.m'
        self.te_input_file = xgc_path+'te_input.in'
        self.ne_input_file = xgc_path+'ne_input.in'
        self.shift = shift
        self.kind = kind
        
        print 'from directory:'+ self.xgc_path
        self.unit_dic = load_m(self.unit_file)
        self.tstep = self.unit_dic['sml_dt']*self.unit_dic['diag_1d_period']
        self.dt = self.tstep * dt
        self.t = self.time_steps * self.tstep
        self.current = 0 # index of the current time step
        # check if limits are asked for the mesh on the toroidal plane
        self.lim = False

        # limits of the mesh
        self.Xmin = limits[0,0]
        self.Xmax = limits[0,1]
        self.Ymin = limits[1,0]
        self.Ymax = limits[1,1]

        if limits.shape[0]==3:
            self.lim = True
            self.Zmin = limits[2,0]
            self.Zmax = limits[2,1]
            
        # limits in tokamak coordinates
        x = np.linspace(self.Xmin,self.Xmax,100)
        y = np.linspace(self.Ymin,self.Ymax,100)
        x,y = np.meshgrid(x,y)
        R = np.sqrt(x**2 + y**2)
        self.Rmin = np.min(R)
        self.Rmax = np.max(R)
        phi = np.array([[self.Xmin,self.Ymin],[self.Xmin,self.Ymax],
                        [self.Xmax,self.Ymin],[self.Xmax,self.Ymax]])
        
        phi = np.arctan2(phi[:,1],phi[:,0])
        
        # put the value between [0,2pi]
        tempmin = np.min(phi)
        tempmax = np.max(phi)
        phi[phi<0] += 2*np.pi
        self.Phimin = np.min(phi)
        self.Phimax = np.max(phi)

        self.dphi = dphi
        self.load_mesh_psi_3D()
        print 'mesh and psi loaded.'
        
        self.load_B_3D()
        print 'B loaded.'

        phi = np.linspace(tempmin,tempmax,100)
        phi[phi<0] += 2*np.pi
        self.refprevplane,self.refnextplane = get_interp_planes_local(self,phi)
        print 'interpolation planes obtained.'
        
        self.load_eq_3D()
        print 'equlibrium loaded.'

        self.load_next_time_step(False)

    def load_next_time_step(self,increase=True,t=None):
        """ Load all the quantities for the next time step.
        
        The old quantities are overwritten.
        
        :param bool increase: Define if the time step should be increase or not
        :param int t: Time step to load (index in self.time)
        """
        if increase:
            self.current += 1

        if t is not None:
            self.current = t
        if self.current >= len(self.time_steps):
            raise XGC_Loader_Error('The time step is bigger than the ones\
            requested')
        

        self.load_fluctuations_3D_all()        
        print 'fluctuations loaded.'
        
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

        Zpts = RZ[:,1]

        psi = mesh['psi']
        # psi interpolant
        fill_ = np.nan
        if not self.lim:
            fill_ = max(psi)
        if self.kind == 'linear':
            self.psi_interp = LinearNDInterpolator(
                np.array([Rpts,Zpts]).T, psi, fill_value=fill_)
        elif self.kind == 'cubic':
            self.psi_interp = CloughTocher2DInterpolator(
                np.array([Rpts,Zpts]).T, psi, fill_value=fill_)
        else:
            raise NameError("The method '{}' is not defined".format(self.kind))

        self.points = np.array([Rpts,Zpts]).transpose()

        if self.lim:
            # 0.99 and 1.01 are for increasing the window in order to take the last point
            # in it
            self.Rmin = np.max([self.Rmin,0.999*np.min(Rpts)])
            self.Rmax = np.min([self.Rmax,1.001*np.max(Rpts)])
            self.Zmin = np.max([self.Zmin,0.999*np.min(Zpts)])
            self.Zmax = np.min([self.Zmax,1.001*np.max(Zpts)])
            
        mesh.close()

        # get the number of toroidal planes from fluctuation data file
        fluc_file0 = self.xgc_path + 'xgc.3d.' + str(self.time_steps[0]).zfill(5)+'.h5'
        fmesh = h5.File(fluc_file0,'r')
        self.n_plane = fmesh['dpot'].shape[1]

        fmesh.close()
        
        

    def load_B_3D(self,eps=0.01):
        """Load equilibrium magnetic field data and compute the interpolant.
        Compute the limits by taking into account the field line interpolation
        """
        B_mesh = h5.File(self.bfield_file,'r')
        B = B_mesh['node_data[0]']['values']
        # keep only the values around the diagnostics

        # interpolant of each direction of the field
        # use np.inf as a flag for outside points,
        # deal with them later in interpolation function
        if self.kind == 'linear':
            self.B_interp = LinearNDInterpolator(
                self.points, B, fill_value = np.inf)
        elif self.kind == 'cubic':
            self.B_interp =  CloughTocher2DInterpolator(
                self.points, B, fill_value = np.inf)
        else:
            raise NameError("The method '{}' is not defined".format(self.kind))
        self.fill_Bphi = np.sign(B[0,2])*np.min(np.absolute(B[:,2]))

        #If toroidal B field is positive, then field line is going in
        # the direction along which plane number is increasing.
        self.CO_DIR = (np.sign(B[0,2]) > 0)
        B_mesh.close()

        if self.lim:
            # increase the limits for taking into account the field line intergration
            N = 20
            r = np.array([np.linspace(self.Rmin,self.Rmax,N),np.linspace(self.Rmin,self.Rmax,N)]).flatten()
            z = np.array([self.Zmin*np.ones(N),self.Zmax*np.ones(N)]).flatten()
            # case on one side of the ref plane
            phi = (1+eps)*self.shift*np.ones(r.shape)
            prev,next_ = get_interp_planes_local(self,phi)
            interp1 = self.find_interp_positions(r,z,phi,prev,next_)
            # case on the other side
            phi = (1-eps)*self.shift*np.ones(r.shape)
            prev,next_ = get_interp_planes_local(self,phi)        
            interp2 = self.find_interp_positions(r,z,phi,prev,next_)

            # find min/max
            rmin1 = np.min(interp1[:,0,...])
            rmax1 = np.max(interp1[:,0,...])
            rmin2 = np.min(interp2[:,0,...])
            rmax2 = np.max(interp2[:,0,...])
            zmin1 = np.min(interp1[:,1,...])
            zmax1 = np.max(interp1[:,1,...])
            zmin2 = np.min(interp2[:,1,...])
            zmax2 = np.max(interp2[:,1,...])

            self.Rmin = np.min([self.Rmin,rmin1,rmin2])
            self.Rmax = np.max([self.Rmax,rmax1,rmax2])
            self.Zmin = np.min([self.Zmin,zmin1,zmin2])
            self.Zmax = np.max([self.Zmax,zmax1,zmax2])
            # remove the points outside the window
            self.ind = (self.points[:,0] > self.Rmin) & (self.points[:,0] < self.Rmax)        
            self.ind = self.ind & (self.points[:,1] > self.Zmin) & (self.points[:,1] < self.Zmax)
            self.points = self.points[self.ind,:]
            print 'Keep: ',str(self.points.shape[0]),'Points on a total of: '\
                ,str(self.ind.shape[0])

        
            print 'Zlimits: [',np.min(self.points[:,1]),np.max(self.points[:,1]),']'
            print 'Rlimits: [',np.min(self.points[:,0]),np.max(self.points[:,0]),']'
        return 0


    def load_fluctuations_3D_all(self):
        """Load non-adiabatic electron density and electrical static 
        potential fluctuations for 3D mesh.
        The required planes are calculated and stored in sorted array.
        fluctuation data on each plane is stored in the same order.
        Note that for full-F runs, the perturbed electron density 
        includes both turbulent fluctuations and equilibrium relaxation,
        this loading method doesn't differentiate them and will read all of them.
        
        """
        # list of all planes of interest
        self.planes = np.unique(np.array([np.unique(self.refprevplane),
                                          np.unique(self.refnextplane)]))
        self.planeID = {self.planes[i]:i for i in range(len(self.planes))}
        #the dictionary contains the positions of each chosen plane,
        # useful when we want to get the data on a given plane known only its plane number in xgc file.
        # the last dimension is for the mesh position
        self.nane = np.zeros( (len(self.planes),
                               len(self.points[:,0])) )
        
        self.phi = np.zeros( (len(self.planes),
                              len(self.points[:,0])) )
        
        flucf = self.xgc_path + 'xgc.3d.'+str(self.time_steps[self.current]).zfill(5)+'.h5'
        fluc_mesh = h5.File(flucf,'r')

        if self.lim:
            self.phi += np.swapaxes(
                fluc_mesh['dpot'][self.ind,:][:,(self.planes)%self.n_plane],0,1)
            
            self.nane += np.swapaxes(
                fluc_mesh['eden'][self.ind,:][:,(self.planes)%self.n_plane],0,1)

        else:
            self.phi += np.swapaxes(
                fluc_mesh['dpot'][:,:][:,(self.planes)%self.n_plane],0,1)
            
            self.nane += np.swapaxes(
                fluc_mesh['eden'][:,:][:,(self.planes)%self.n_plane],0,1)
            
        fluc_mesh.close()
            
        return 0

    def load_eq_3D(self):
        """Load equilibrium profiles and compute the interpolant
        """
        eqf = self.xgc_path + 'xgc.oneddiag.h5'
        eq_mesh = h5.File(eqf,'r')
        eq_psi = eq_mesh['psi_mks'][:]
        self.psi_x = load_m(self.xgc_path + 'units.m')['psi_x']
        #sometimes eq_psi is stored as 2D array, which has time series infomation.
        # For now, just use the time step 1 psi array as the unchanged array.
        # NEED TO BE CHANGED if equilibrium psi mesh is changing over time.
        n_psi = eq_psi.shape[-1]
        eq_psi = eq_psi.flatten()[0:n_psi] #pick up the first n psi values.

        eq_ti = eq_mesh['i_perp_temperature_1d'][0,:]
        eq_ni = eq_mesh['i_gc_density_1d'][0,:]
        self.ni_min = np.min(eq_ni)
        self.ti_min = np.min(eq_ti)

        # interpolant for the equilibrium of the ions
        self.ti0_sp = splrep(eq_psi,eq_ti,k=1)
        self.ni0_sp = splrep(eq_psi,eq_ni,k=1)
        #simulation has electron dynamics
        eq_te = eq_mesh['e_perp_temperature_1d'][0,:]
        eq_ne = eq_mesh['e_gc_density_1d'][0,:]
        self.te_min = np.min(eq_te)
        self.ne_min = np.min(eq_ne)
        self.te0_sp = splrep(eq_psi,eq_te,k=1)
        self.ne0_sp = splrep(eq_psi,eq_ne,k=1)
        
        eq_mesh.close()

    def calc_total_ne_3D(self,psi,nane,pot):
        """Calculate the total electron at the wanted points.

        :param np.array[N] psi: Psi in mks unit
        :param np.array[N] nane: Fluctuation of the density
        :param np.array[N] pot: Potential 

        :returns: Total density
        :rtype: np.array[N]
        
        """
        # temperature and density (equilibrium) on the psi mesh
        te0 = splev(psi,self.te0_sp)
        # avoid temperature <= 0
        te0[te0<self.te_min/10] = self.te_min/10
        ne0 = splev(psi,self.ne0_sp)
        ne0[ne0<self.ne_min/10] = self.ne_min/10
        
        dne_ad = np.zeros(pot.shape)
        dne_ad += ne0*pot/te0
        ind = np.isfinite(dne_ad)
        ad_valid_idx = np.zeros(dne_ad.shape,dtype=bool)
        ad_valid_idx[ind] = np.absolute(dne_ad[ind])<= np.absolute(ne0[ind])

        ne = np.zeros(pot.shape)
        ne += ne0[:]
        ne[ad_valid_idx] += dne_ad[ad_valid_idx]
        ind2 = np.isfinite(nane)
        na_valid_idx = np.zeros(nane.shape,dtype=bool)
        na_valid_idx[ind2] = np.absolute(nane[ind2])<= np.absolute(ne[ind2])
        ne[na_valid_idx] += nane[na_valid_idx]
        ne[~ind | ~ind2] = ne0[~ind | ~ind2]
        return ne
    
    def compute_interpolant(self):
        """ Compute the interpolant for the ion and electron
            density
        """
        # list of interpolant
        self.interpfluc = []
        for j in range(len(self.planes)):
            # computation of interpolant
            if self.kind == 'linear':
                self.interpfluc.append(
                    LinearNDInterpolator(self.points,np.array([self.phi[j,:],self.nane[j,:]]).T,fill_value=np.nan))
            elif self.kind == 'cubic':
                self.interpfluc.append(
                    CloughTocher2DInterpolator(self.points,np.array([self.phi[j,:],self.nane[j,:]]).T,fill_value=np.nan))
            else:
                raise NameError("The method '{}' is not defined".format(self.kind))
    

    def find_interp_positions(self,r,z,phi,prev_,next_):
        """Using B field information and follows the exact field line.
        
        :param np.array[N] r: R coordinates
        :param np.array[N] z: Z coordinates
        :param np.array[N] phi: Phi coordinates
        :param np.array[N] prev_: Previous planes
        :param np.array[N] next_: Next planes

        :returns: Positions on the the previous plane and the next one
        :rtype: np.array[2,3,N] (2 for the previous/next plane, 3 for R,Z,L where L is the distance between plane and points)
        """

        prevplane,nextplane = prev_.flatten(),next_.flatten()

        # angle between two planes of the simulation
        dPhi = 2*np.pi/self.n_plane

        # angle of the planes
        phi_planes = np.arange(self.n_plane)*dPhi+self.shift
        ind = phi_planes < 0
        # put the angle of the planes inside [0,2*pi]
        while np.sum(ind) != 0:
            phi_planes[ind] = phi_planes[ind] + 2*np.pi
            ind = phi_planes < 0

        ind = phi_planes > 2*np.pi
        # put the angle of the planes inside [0,2*pi]
        while np.sum(ind) != 0:
            phi_planes[ind] = phi_planes[ind] - 2*np.pi
            ind = phi_planes > 2*np.pi
        

        # the previous/next planes depend on the direction of the field
        if(self.CO_DIR):
            # distance (angle) between the phiw wanted and the closest planes
            phiFWD = np.where(phi_planes[nextplane] < dPhi ,phi_planes[nextplane] + np.pi*2 - phi, phi_planes[nextplane]-phi)
            phi[phi<phi_planes[prevplane]] += 2*np.pi
            phiBWD = phi_planes[prevplane]-phi
        else:
            phiBWD = np.where(phi_planes[prevplane] < dPhi ,phi_planes[prevplane] + np.pi*2 - phi, phi_planes[prevplane]-phi)
            phi[phi<phi_planes[nextplane]] += 2*np.pi
            phiFWD = phi_planes[nextplane]-phi
        # angle between two steps
        R_FWD = np.copy(r)
        R_BWD = np.copy(r)
        Z_FWD = np.copy(z)
        Z_BWD = np.copy(z)
        s_FWD = np.zeros(r.shape)
        s_BWD = np.zeros(r.shape)

        # check which index need to be integrated
        ind = np.ones(r.shape,dtype=bool)
        # Coefficient of the Runge-Kutta method
        a,b,c,Nstage = runge_kutta_explicit(2)
        sign = 1.0
        # use any because all the angle are of the same sign
        if (phiFWD < 0).any():
            sign = -1.0
        # forward step
        while ind.any():
            # size of the next step for each position
            step = np.copy(phiFWD[ind])
            step[np.abs(step) > self.dphi] = sign*self.dphi
            # update the position of the next iteration
            phiFWD[ind] -= step
            K = np.zeros((r[ind].shape[0],3,Nstage))
            for i in range(Nstage):
                # compute the coordinates of this stage
                Rtemp = R_FWD[ind] + step*np.sum(a[i,:i]*K[:,0,:i],axis=1)
                Ztemp = Z_FWD[ind] + step*np.sum(a[i,:i]*K[:,1,:i],axis=1)
                Btemp = self.B_interp(Rtemp,Ztemp)
                # avoid the part outside the mesh
                indinf = np.isfinite(Btemp[:,2])

                # evaluate the function
                K[indinf,0,i] = Rtemp[indinf] * Btemp[indinf,0] / Btemp[indinf,2]
                K[indinf,1,i] = Rtemp[indinf] * Btemp[indinf,1] / Btemp[indinf,2]
                K[indinf,2,i] = np.sqrt(Rtemp[indinf]**2 + K[indinf,0,i]**2 + K[indinf,1,i]**2)


            # compute the final value of this step
            dR_FWD = step*np.sum(b[np.newaxis,:]*K[:,0,:],axis=1)
            dZ_FWD = step*np.sum(b[np.newaxis,:]*K[:,1,:],axis=1)
            ds_FWD = np.abs(step)*np.sum(b[np.newaxis,:]*K[:,2,:],axis=1)
            
            #when the point gets outside of the XGC mesh, set BR,BZ to zero.
            dR_FWD[~np.isfinite(dR_FWD)] = 0.0
            dZ_FWD[~np.isfinite(dZ_FWD)] = 0.0
            ds_FWD[~np.isfinite(ds_FWD)] = 0.0

            # update the global value
            s_FWD[ind] += ds_FWD
            Z_FWD[ind] += dZ_FWD
            R_FWD[ind] += dR_FWD
            ind = (phiFWD != 0)

        # check which index need to be integrated
        ind = np.ones(r.shape,dtype=bool)
        sign = 1.0
        if (phiBWD < 0).any():
            sign = -1.0
        # backward step
        while ind.any():
            # size of the next step for each position
            step = np.copy(phiBWD[ind])
            step[np.abs(step) > self.dphi] = sign*self.dphi
            # update the position of the next iteration
            phiBWD[ind] -= step
            K = np.zeros((r[ind].shape[0],3,Nstage))
            for i in range(Nstage):
                # compute the coordinates of this stage
                Rtemp = R_BWD[ind] + step*np.sum(a[i,:i]*K[:,0,:i],axis=1)
                Ztemp = Z_BWD[ind] + step*np.sum(a[i,:i]*K[:,1,:i],axis=1)
                Btemp = self.B_interp(Rtemp,Ztemp)
                indinf = np.isfinite(Btemp[:,2])
                # evaluate the function
                
                K[indinf,0,i] = Rtemp[indinf] * Btemp[indinf,0] / Btemp[indinf,2]
                K[indinf,1,i] = Rtemp[indinf] * Btemp[indinf,1] / Btemp[indinf,2]
                K[indinf,2,i] = np.sqrt(Rtemp[indinf]**2 + K[indinf,0,i]**2 + K[indinf,1,i]**2)


            # compute the final value of this step
            dR_BWD = step*np.sum(b[np.newaxis,:]*K[:,0,:],axis=1)
            dZ_BWD = step*np.sum(b[np.newaxis,:]*K[:,1,:],axis=1)
            ds_BWD = np.abs(step)*np.sum(b[np.newaxis,:]*K[:,2,:],axis=1)
            
            #when the point gets outside of the XGC mesh, set BR,BZ to zero.
            dR_BWD[~np.isfinite(dR_BWD)] = 0.0
            dZ_BWD[~np.isfinite(dZ_BWD)] = 0.0
            ds_BWD[~np.isfinite(ds_BWD)] = 0.0

            # update the global value
            s_BWD[ind] += ds_BWD
            Z_BWD[ind] += dZ_BWD
            R_BWD[ind] += dR_BWD
            ind = (phiBWD != 0)

        interp_positions = np.zeros((2,3,r.shape[0]))
        
        interp_positions[0,0,...] = R_BWD
        interp_positions[0,1,...] = Z_BWD
        tot = (s_BWD+s_FWD)
        ind = tot != 0.0
        interp_positions[0,2,ind] = (s_BWD[ind]/tot[ind])
        interp_positions[1,0,...] = R_FWD
        interp_positions[1,1,...] = Z_FWD
        interp_positions[1,2,...] = 1-interp_positions[0,2,...]

        return interp_positions


    def interpolate_data(self,pos,timestep,quant,eq,check=True):
        """ Interpolate the data to the position wanted
        
        :param np.array[N,3] pos: Position (in cartesian system)
        :param int timestep: Time step wanted (used to check if an error is not made)
        :param list[str] quant: Desired quantities (can be 'ni', 'ne', 'Ti', 'Te')
        :param bool eq: Choice between equilibrium or exact quantities
        :param bool check: Print message if outside the mesh

        :returns: Quantities asked in the right order
        :rtype: tuple(np.array[N],...)
        """
        if (timestep is not self.current) and (eq is False):
            print 'Maybe an error is made, the requested time step is not the same than the XGC one'
            print timestep, self.current
        

        # compute the coordinate in the torroidal system
        r = np.sqrt(np.sum(pos[...,0:2]**2,axis=-1)).flatten()
        z = pos[...,2].flatten()
        phi = np.arctan2(pos[...,1],pos[...,0]).flatten()
        # goes into the interval [0,2pi]
        phi[phi<0] += 2*np.pi
        
        # get the planes for theses points
        prevplane,nextplane = get_interp_planes_local(self,phi)
        # check if asking for densities
        ne_bool = 'ne' in quant
        
        if eq or ('Te' in quant) or ('Ti' in quant):
            #psi on grid
            psi = self.psi_interp(r,z)
        # check if want equilibrium data
        if eq:
            #ne0 on grid
            if ne_bool:
                ne = splev(psi,self.ne0_sp)
                ne[ne < self.ne_min/10] = self.ne_min/10

        #Te and Ti on grid
        if 'Te' in quant:
            Te = splev(psi,self.te0_sp)
            Te[Te < self.te_min/10] = self.te_min/10
        if 'Ti' in quant:
            Ti = splev(psi,self.ti0_sp)
            Ti[Ti < self.ti_min/10] = self.ti_min/10

                
        if not eq:
            if ne_bool:
                ne = np.zeros(r.shape[0])
                interp_positions = self.find_interp_positions(r,z,phi,prevplane,nextplane)
                if self.lim:
                    ind = (self.Zmax > interp_positions[0,1]) & (interp_positions[0,1] > self.Zmin)
                    ind = ind & ((self.Zmax > interp_positions[1,1]) & (interp_positions[1,1] > self.Zmin))
                    ind = ind & (self.Rmax > interp_positions[0,0]) & (interp_positions[0,0] > self.Rmin)
                    ind = ind & (self.Rmax > interp_positions[1,0]) & (interp_positions[1,0] > self.Rmin)
                
                #create index dictionary, for each key as plane number and
                # value the corresponding indices where the plane is used as previous or next plane.
                prev_idx = {}
                next_idx = {}
                for j in range(len(self.planes)):
                    prev_idx[j] = np.where(prevplane == self.planes[j] )
                    next_idx[j] = np.where(nextplane == self.planes[j] )

                #for each time step, first create the 2 arrays of quantities for interpolation
                prevn = np.zeros((r.shape[0],2))
                nextn = np.zeros((prevn.shape[0],2))
                
                for j in range(len(self.planes)):
                    # interpolation on the poloidal planes
                    prevn[prev_idx[j]] = self.interpfluc[j](
                        interp_positions[0,0][prev_idx[j]], interp_positions[0,1][prev_idx[j]])
                    
                    nextn[next_idx[j]] = self.interpfluc[j](
                        interp_positions[1,0][next_idx[j]], interp_positions[1,1][next_idx[j]])
                # interpolation along the field line
                phi_pot = prevn[:,0] * interp_positions[1,2,...] + nextn[:,0] * interp_positions[0,2,...]
                ne = prevn[:,1] * interp_positions[1,2,...] + nextn[:,1] * interp_positions[0,2,...]
                psi = self.psi_interp(interp_positions[0,0,...],interp_positions[0,1,...])
                psin= self.psi_interp(interp_positions[1,0,...],interp_positions[1,1,...])
                psi = psin * interp_positions[1,2,...] + psi * interp_positions[0,2,...]
                ne = self.calc_total_ne_3D(psi,ne,phi_pot)

                if self.lim:
                    ne[~ind] = np.nan
                    if check & (~ind).any():
                        print 'r',r[~ind]
                        print 'z',z[~ind]
                        print 'phi',phi[~ind]
                        print 'prev',prevn[~ind,:]
                        print 'next',nextn[~ind,:]
                        print interp_positions[:,:,~ind]

                    
        ret = ()
        # put the data in the good order
        for i in quant:
            if i is 'ne':
                ret += (ne,)
            elif i is 'Ti':
                ret += (Ti,)
            elif i is 'Te':
                ret += (Te,)
            else:
                raise XGC_Loader_Error("'{}' is not a valid value in the evaluation of XGC data".format(i))
        return ret

        


