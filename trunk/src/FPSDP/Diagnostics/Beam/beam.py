# math
import numpy as np
import math
from scipy import interpolate
# file reading
import json
import ConfigParser as psr

import FPSDP.Plasma.Collisions.collisions as col
import FPSDP.Maths.Integration as integ
from FPSDP.GeneralSettings.UnitSystem import SI


class Beam1D:
    """ Simulate a 1D beam with the help of data from simulation.

    Compute the beam density from the equilibrium or the raw data on a mesh.
    The beam density is not very sensitive of the fluctuation, therefore
    only the central line is computed and a gaussian profile is assumed (with a different
    vertical and horizontal width).
    When computing a beam density outside the mesh, an extrapolation is made (cubic).
    Two ways of computing the emission exists, a first one by assuming that
    the lifetime of the excited state is negligible (thus only the datas from the point 
    considered are used) and a second one that compute the expected value
    from an exponential decay.
    The simulation data are saved in this class.

    :param str config_file: Name of the config file

    :var str self.cfg_file: Name of the config file
    :var str self.adas_atte: Name of the ADAS21 files (beam stopping coefficient)
    :var str self.adas_emis: Name of the ADAS22 files (emission coefficient)
    
    :var self.collisions: :class:`Collisions <FPSDP.Plasma.Collisions.collisions.Collisions>` instance variable.\
    Compute all the coefficients (beam stopping and emission) for the diagnostics.

    :var list[list[int,int]] self.coll_atte: List of couple between a ADAS21 file \
    (beam stopping coefficient) and a beam component (in this order)
    :var list[list[int,int]] self.coll_emis: List of couple between a ADAS21 file \
    (emission coefficient) and a beam component (in this order)
    
    :var int self.Nlt: Number of point for the mesh in the lifetime effect
    :var float elf.t_max: Cut-off for the integral of the liftetime effect\
    (in unit of the lifetime)

    :var np.array[Ncomp] self.mass_b: Mass of a particule in the beam (one for\
    each beam component) in amu
    :var np.array[Ncomp] self.beam_comp: Beam energy of each component (in eV)

    :var float self.power: Total power of the laser (in W)
    :var np.array[Ncomp] self.frac: Fraction of energy of each beam component (in percent)
    :var np.array[3] self.pos: Position of the beam source (in cartesian system) in m
    :var np.array[3] self.direc: Direction of the beam (unit vector)
    :var float self.beam_width_h: Horizontal beam width (FWHM) in m
    :var float self.beam_width_v: Vertical beam width (FWHM) in m
    :var float self.stddev_h: Horizontal beam width (standard deviation)
    :var float self.stddev_v: Vertical beam width (standard deviation)
    :var float self.stddev2_h: Square Horizontal beam width (standard deviation)
    :var float self.stddev2_v: Square Vertical beam width (standard deviation)

    :var np.array[Ncomp] self.speed: Speed of the particles of each component\
    (does not take relativity in account)
    :var self.data: Data from a loader (Actually only XGC datas are accepted)
    :var np.array[3] self.inters: Ending 3D point of the mesh (intersection with\
    the limit of the optical system)
    :var int self.Nz: Number of point for the discretization of the beam
    :var np.array[Nz] self.dl: Distance between the origin and each point of the mesh
    :var np.array[Nz,3] self.mesh: Position of the mesh points (in cartesian system)
    :var np.array[Ncomp,Nz] self.density_beam: Particle density of each component\
    on the mesh
    :var list[tck_interp] self.nb_tck: Interpolant for each component (use cubic spline)
    :var int self.t_: Current time step
    :var bool self.eq: Equilibrium data for the attenuation
    :var np.array[3,2] self.limits: Limits for the beam
    """
    
    def __init__(self,config_file):
        """ Load everything from the config file"""
        self.cfg_file = config_file                                          #!
        config = psr.ConfigParser()
        config.read(self.cfg_file)

        # The example input is well commented
        
        # load data for collisions
        lt = json.loads(config.get('Collisions','tau'))
        self.adas_atte = json.loads(config.get('Collisions','adas_atte'))    #!
        self.adas_emis = json.loads(config.get('Collisions','adas_emis'))    #!
        n_low = json.loads(config.get('Collisions','n_low'))
        n_high = json.loads(config.get('Collisions','n_high'))
        self.collisions = col.Collisions(self.adas_atte,self.adas_emis,
                                         (n_low,n_high),lt)                  #!
        self.coll_atte = json.loads(config.get('Collisions','coll_atte'))    #!
        self.coll_emis = json.loads(config.get('Collisions','coll_emis'))    #!
        self.Nlt = int(json.loads(config.get('Collisions','Nlt')))           #!
        self.t_max = json.loads(config.get('Collisions','t_max'))            #!
        

        # load data about the beam energy
        self.mass_b = json.loads(config.get('Beam energy','mass_b'))         #!
        self.mass_b = np.array(self.mass_b)
        self.beam_comp = json.loads(config.get('Beam energy','E'))           #!
        self.beam_comp = 1000*np.array(self.beam_comp)
        self.power = float(config.get('Beam energy','power'))                #!
        self.frac = json.loads(config.get('Beam energy','f'))                #!
        self.frac = np.array(self.frac)
        if np.sum(self.frac) > 100: # allow the possibility to simulate only
            # a part of the beam
            raise NameError('Sum of f is greater than 100')

        # load data about the geometry of the beam
        self.pos = json.loads(config.get('Beam geometry','position'))        #!
        self.pos = np.array(self.pos)
        self.direc = json.loads(config.get('Beam geometry','direction'))     #!
        self.direc = np.array(self.direc)
        self.direc = self.direc/np.sqrt(np.sum(self.direc**2))
        self.beam_width_h = json.loads(
            config.get('Beam geometry','beam_width_h'))                      #!
        self.beam_width_v = json.loads(
            config.get('Beam geometry','beam_width_v'))                      #!
        self.Nz = int(config.get('Beam geometry','Nz'))                      #!

        
        # get the standard deviation at the origin
        self.stddev_h = self.beam_width_h/(2.0*np.sqrt(2.0*np.log(2.0)))     #!
        self.stddev_v = self.beam_width_v/(2.0*np.sqrt(2.0*np.log(2.0)))     #!
        self.stddev2_h = self.stddev_h**2                                    #!
        self.stddev2_v = self.stddev_v**2                                    #!

        # speed of each beam
        self.speed = np.zeros(len(self.beam_comp))                           #!
        self.speed = np.sqrt(2*self.beam_comp*SI['keV']/(1000*SI['amu']*self.mass_b))

        # equilibrium data for the attenuation
        self.eq = json.loads(config.get('Collisions','eq_atte'))                   #!
        # current time step
        self.t_ = -1                                                         #!


    def set_data(self,data,limits):
        """ Second part of the initialization (should be called manually!).

        Due to the computation of the limits (:func:`compute_limits <FPSDP.Diagnostics.BES.bes.BES.compute_limits>`),
        the data are loaded after the initialization.
        This method save the data inside the instance and compute the beam density on the mesh

        :param data: Loader of the simulation data
        :type data: e.g. :class:`XGC_Loader_BES <FPSDP.Plasma.XGC_Profile.load_XGC_BES.XGC_Loader_BES>`
        :param np.array[3,2] limits: Limits for the mesh (first index for X,Y,Z, second for min/max))
        """
        self.limits = limits                                                 #!
        self.data = data                                                     #!
        print 'Creating mesh'
        self.create_mesh()
        print 'Computing beam density'
        self.compute_beam_on_mesh()

    def get_width(self,dist):
        """ Compute the beam width at a specific distance.

        Can be modify for adding the effect of the beam divergence.
        
        :param dist: Distance from the origin of the beam
        :type dist: np.array[N]
    
        :returns: Horizontal and vertical beam width (0 for horizontal, 1 for vertical)
        :rtype: np.array[2,dist.shape]
        """
        return np.array([self.stddev_h*np.ones(dist.shape),
                         self.stddev_v*np.ones(dist.shape)])

    def create_mesh(self):
        """ Create the 1D mesh between the source of the beam and the end 
        of the mesh.

        Is called during the initialization
        """
        # intersection between end of mesh and beam
        self.inters = self.find_wall()                                       #!
        
        length = np.sqrt(np.sum((self.pos-self.inters)**2))
        # distance to the origin along the central line
        self.dl = np.linspace(0,length,self.Nz)
        self.mesh = np.zeros((self.Nz,3))                                    #!
        # the second index corresponds to the dimension (X,Y,Z)
        self.mesh = self.dl[:,np.newaxis]*self.direc + self.pos              #!
        
                    
    def find_wall(self, eps=1e-6):
        """ Find the wall (of the mesh) that will stop the beam and return
        the coordinate of the intersection with the beam.
        
        :param float eps: Ratio of increase size on each side of the box
        :returns: Position of the intersection between the end of the mesh and the beam\
        (in cartesian system)
        :rtype: np.array[3]
        """
        # X-direction
        if self.direc[0] == 0.0:
            tx = np.inf
        else:
            tx1 = (self.limits[0,1]-self.pos[0])/self.direc[0]
            tx2 = (self.limits[0,0]-self.pos[0])/self.direc[0]
            tx = max(tx1,tx2)
        
        # Y-direction
        if self.direc[1] == 0.0:
            ty = np.inf
        else:
            ty1 = (self.limits[1,1]-self.pos[1])/self.direc[1]
            ty2 = (self.limits[1,0]-self.pos[1])/self.direc[1]
            ty = max(ty1,ty2)
        
        # Z-direction
        if self.direc[2] == 0.0:
            tz = np.inf
        else:
            tz1 = (self.limits[2,1]-self.pos[2])/self.direc[2]
            tz2 = (self.limits[2,0]-self.pos[2])/self.direc[2]
            tz = max(tz1,tz2)

        t_ = np.array([tx,tz,ty])
        if not (np.isfinite(t_) & (t_>0)).any():
            raise NameError('The beam does not cross the window')
        t = np.argmin(t_)
        
        return self.pos + self.direc*t_[t]*(1-eps)

    def get_quantities(self,pos,t_,quant,eq=False, check=True):
        """ Compute the quantities from the datas
        
        Use the list of string quant for taking the good values inside the simulation datas.
        See :func:`interpolate_data <FPSDP.Plasma.XGC_Profile.load_XGC_BES.XGC_Loader_BES.interpolate_data>`

        
        :param np.array[N,3] pos: List of position where to take the quantities (in cartesian system)
        :param int t_: Time step considered
        :param list[str] quant: List containing the wanted quantities \
        (See :func:`interpolate_data <FPSDP.Plasma.XGC_Profile.load_XGC_BES.XGC_Loader_BES.interpolate_data>`\
        for more information)
        :param bool eq: Equilibrium data or not
        :param bool check: Print error message if outside the mesh or not

        :returns: The interpolated value from the simulation in the same order than quant
        :rtype: tuple[quant]
        """
        check = True
        if isinstance(t_,list):
            raise NameError('Only one time should be given')
        return self.data.interpolate_data(pos,t_,quant,eq,check)
        
        
    def compute_beam_on_mesh(self):
        r""" Compute the beam density on the mesh and the interpolant.

        Use the Gauss-Legendre quadrature of order 2 for computing the integral:

        .. math::
           n_b(P) = n_{b,0} \exp\left(-\int_0^P n_e(z)S_\text{cr}\left(E,n_e(z),T_i(z)\right)\sqrt{\frac{m}{2E}}\mathrm{d}z\right)

        where :math:`n_b(P)` is the density at the point P (along the beam central line),
        :math:`n_{b,0}` is the density at the origin, :math:`n_e` is the electron density,
        :math:`S_\text{cr}` is the beam stopping coefficient (depending on the beam energy [:math:`E_b`],
        the ion temperature [:math:`T_i`] and the electron density), :math:`m` is the mass of a particle in the beam,
        :math:`\mathrm{d}z` is along the central line.

        The initial density is computed with the help of the total power (:math:`P`):

        .. math::

           P = \int_\Omega E v n_{b,0} \mathrm{d}\sigma

        where :math:`\Omega` is the 2D space perpendicular to the beam direction and
        :math:`v = \sqrt{\frac{2E}{m}}` is the velocity of the particles.

        """
        self.t_ += 1
        self.density_beam = np.zeros((len(self.beam_comp),self.Nz))  #!
        # density of the beam at the origin
        n0 = np.zeros(len(self.beam_comp))
        n0 = np.sqrt(2*self.mass_b*SI['amu'])*self.power
        n0 *= self.frac/(self.beam_comp*SI['keV']/1000)**(1.5)
        
        # define the quadrature formula for this method
        quad = integ.integration_points(1,'GL2') # Gauss-Legendre order 2
        # can be rewritten by computing the integral of all the interval at
        # once and using cumulative sum
        for j in range(self.mesh.shape[0]):
            # density over the central line (usefull for some check)
            if j is not 0:
                temp_beam = np.zeros(len(self.beam_comp))
                for k in self.coll_atte:
                    file_nber = k[0]
                    beam_nber = k[1]
                    # limit of the integral
                    a = self.mesh[j-1,:]
                    b = self.mesh[j,:]
                    # average
                    av = (a+b)/2.0
                    # difference
                    diff = (b-a)/2.0
                    # integration point
                    pt = quad.pts[:,np.newaxis]*diff + av
                    # compute all the values needed for the integral
                    ne, T = self.get_quantities(pt,self.t_,['ne','Te'],self.eq,check=False)
                    
                    # attenuation coefficient from adas
                    S = self.collisions.get_attenutation(
                        self.beam_comp[beam_nber],ne,self.mass_b[beam_nber],
                        T,file_nber)

                    # half distance between a & b
                    norm_ = np.sqrt(np.sum(diff**2))
                    temp1 = np.sum(ne*S*quad.w)
                    temp1 *= norm_/self.speed[beam_nber]
                    temp_beam[beam_nber] += temp1
                    
                self.density_beam[:,j] = self.density_beam[:,j-1] - \
                                                temp_beam

        self.density_beam = n0[:,np.newaxis]*np.exp(self.density_beam)
        self.nb_tck = []                                                     #!
        # interpolant for the beam
        for i in range(len(self.beam_comp)):
            self.nb_tck.append(interpolate.splrep(self.dl,self.density_beam[i,:],k=1))
        
            
    def get_mesh(self):
        """ Acces method to the mesh

        :returns: self.mesh
        :rtype: np.array[Nz,3]
        """
        return self.mesh

    def get_origin(self):
        """ Acces method to the origin of the beam

        :returns: self.pos (in cartesian system)
        :rtype: np.array[3]
        """
        return self.pos

    def get_beam_density(self,pos):
        """ Compute the beam density with the help of the interpolant.
        
        Change the coordinate of the position from the cartesian system to
        the beam system and, after, interpolate the data.

        :param np.array[...,3] pos: List of position where to take the\
        beam density (in cartesian system)

        :returns: Beam density
        :rtype: np.array[:]
        """
        pos = np.reshape(pos,(-1,3))
        nb = np.zeros((self.beam_comp.shape[0],pos.shape[0]))
        # vector from beam origin to the wanted position
        dist = pos - self.get_origin()
        # result is a 1D array containing the projection of the distance
        # from the origin over the direction of the beam
        proj = np.einsum('ij,j->i',dist,self.direc)
        stddev = self.get_width(proj)
        stddev2 = stddev**2
        
        # cubic spline for finding the value along the axis
        for i in range(len(self.beam_comp)):
            for j in range(pos.shape[0]):
                nb[i,j] = interpolate.splev(proj[j],self.nb_tck[i], ext=1)
        # radius^2 on the plane perpendicular to the beam
        R2 = dist - proj[...,np.newaxis]*self.direc
        # compute the norm of each position
        xy = np.sum(R2[...,0:2]**2, axis=-1)
        z = R2[...,2]**2
        nb = nb*np.exp(-xy/(2*stddev2[0]) - z/(2*stddev2[1]))/(
            2*np.pi*stddev[0]*stddev[1])
        return nb

                    
        
    def get_emis(self,pos,t_):
        r""" Compute the emission at a given position and time step

        :math:`\varepsilon = \langle\sigma v\rangle n_b n_e`
       
        :param np.array[N,3] pos: Position in the cartesian system
        :param float t_: Time step

        :returns: :math:`\varepsilon`
        :rtype: np.array[Ncomp,N]
        """
        # first take all the value needed for the computation
        if len(pos.shape) == 1:
            pos = pos[np.newaxis,:]
        emis = np.zeros((len(self.beam_comp),pos.shape[0]))
        n_e,Te = self.get_quantities(pos,t_,['ne','Te'])
        # loop over all the type of collisions
        n_b = self.get_beam_density(pos)
        for k in self.coll_emis:
            file_nber = k[0]
            beam_nber = k[1]
            # compute the emission coefficient
            emis[beam_nber,:] += self.collisions.get_emission(
                self.beam_comp[beam_nber],n_e,self.mass_b[beam_nber],Te,file_nber)
        # compute the emissivity
        emis = emis*n_e*n_b
        return emis


    def get_emis_lifetime(self,pos,t_):
        r""" Compute the emission at a given position and time step

        For the density of excited particles, the following formula is used:
        :math:`n_\text{ex} = \frac{1}{\|v\|}\int_0^{\tau vd} \varepsilon(P-\delta \hat{v})\exp\left(-\frac{\delta}{v\tau}\right)\mathrm{d}\delta`
        where :math:`v` is the velocity of the beam particles (:math:`\hat{v} = \frac{\vec{v}}{\|\vec{v}\|}`),
        :math:`\varepsilon` is the emissivity computed in :func:`get_emis <FPSDP.Diagnostics.Beam.beam.Beam1D.get_emis>`, and
        :math:`\tau` is the lifetime.

        Therefore the emissivity is given by:
        :math:`\varepsilon_l(P) = \frac{n_\text{ex}(P)}{\tau}`
       
        :param np.array[N,3] pos: Position in the cartesian system
        :param float t_: Time step

        :returns: :math:`\varepsilon_l`
        :rtype: np.array[Ncomp,N]
        """
        pos = np.atleast_2d(pos)

        quad = integ.integration_points(1,'GL2') # Gauss-Legendre order 2
        emis = np.zeros((len(self.beam_comp),pos.shape[0]))
        # avoid the computation at each time
        ne_in, Te_in = self.get_quantities(pos,t_,['ne','Te'])
        
        #nb_in = self.get_beam_density(pos)
        for k in self.coll_emis:
            file_nber = k[0]
            beam_nber = k[1]
            # loop over all the position
            l = self.collisions.get_lifetime(ne_in,Te_in,
                                             self.beam_comp[beam_nber],
                                             self.mass_b[beam_nber],file_nber)
            dist = np.sqrt(np.sum((pos-self.pos)**2,axis=-1))
            # used for avoiding the discontinuity before the origin
            # of the beam
            up_lim = np.minimum(l*self.t_max*self.speed[beam_nber],dist)
            check_ = ~(up_lim == dist).any()
            # split the distance in interval
            #delta = integ.get_interval_exponential(up_lim,l*self.speed[beam_nber],self.Nlt,check=check_)

            step = up_lim/float(self.Nlt-1)
            delta = step[...,np.newaxis]*np.arange(0, self.Nlt)
            
            # average position (a+b)/2
            av = 0.5*(delta[...,:-1] + delta[...,1:])
            # half distance (b-a)/2
            diff = 0.5*(-delta[...,:-1] + delta[...,1:])
            # integration points at each interval
            pt = av[...,np.newaxis] + diff[...,np.newaxis]*quad.pts

            # points in 3D space
            x = pos[:,np.newaxis,np.newaxis,:] \
                - pt[...,np.newaxis]*self.direc

            n_e, Te = self.get_quantities(x,t_,['ne','Te'])

            n_e = np.reshape(n_e,pt.shape)
            Te = np.reshape(Te,pt.shape)

            n_b = self.get_beam_density(x)[beam_nber,...]
            n_b = np.reshape(n_b,pt.shape)
        
            
            f = self.collisions.get_emission(self.beam_comp[beam_nber],n_e.flatten()
                                             ,self.mass_b[beam_nber],Te.flatten(),file_nber)
            f = np.reshape(f,pt.shape)

            # should be change if l depends on position
            f = n_b*f*n_e*np.exp(-pt/(l[...,np.newaxis,np.newaxis]*
                                      self.speed[beam_nber]))/self.speed[beam_nber]
            
            if np.isnan(f).any():
                print 'isnan',np.isnan(f)
                print 'pos',pos
                raise NameError('Mesh not well computed')

            f = np.einsum('kmn,n->km',f,quad.w)
            emis[beam_nber,:] = np.sum(diff*f,axis=-1)/l
        return emis
