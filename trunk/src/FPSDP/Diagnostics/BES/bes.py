# Math
import numpy as np
import scipy as sp
# Input file
import json # for loading input
import ConfigParser as psr # for loading input
# parallel
import multiprocessing as mp
import copy_reg
import types
# beam
import FPSDP.Diagnostics.Beam.beam as be
# grid for data
import FPSDP.Geometry.Grid as Grid
# data loader
import FPSDP.Plasma.XGC_Profile.load_XGC_BES as xgc
# quadrature formula
import FPSDP.Maths.Integration as integ
from FPSDP.GeneralSettings.UnitSystem import SI
from FPSDP.Maths.Funcs import heuman, solid_angle_disk

from os.path import exists # used for checking if the input file exists
# it is not clear with configparser error


def _pickle_method(m):
    """ stuff for parallelisation"""
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

# for parallelisation
copy_reg.pickle(types.MethodType, _pickle_method)


class BES:
    """ Class computing the image of all the fibers.
    
    Load the parameter from a config file and create everything from it.
    The function :func:`get_bes()` is used for computing the intensity received by
    each fiber (number of photons per second).

    :param str input_file: Name of the config file
    :param bool parallel: Choice between the serial code or the parallel one
    :var str self.cfg_fil: Name of the config file
    :var bool self.para: Choice between the serial code and the parallel one
    :var np.array[3] self.pos_lens: Position of the lens (in the cartesian system)
    :var np.array[Nfib] self.rad_ring: Radius of the focus point for each fiber
    :var float self.rad_lens: Radius of the lens
    :var float self.inter: Cutoff distance from the focus point (in unit of the beam width)
    :var int self.Nint: Number of point for splitting the integral over the optical direction
    :var int self.Nsol: Number of interval for the evaluation of the solid angle in the mixed case
    :var np.array[Nfib,3] self.pos_foc: Position of the focus points (in the cartesian system)
    :var np.array[Nfib,3] self.op_direc: Direction of the optical line (for each fiber)
    :var np.array[Nfib] self.dist: Distance between the focus point and the lens
    :var np.array[Nfib,3] self.perp1: Second basis vector for each fiber coordinates (first is the optical line)
    :var np.array[Nfib,3] self.perp2: Third basis vector
    :var str self.type_int: choice between the full computation of the intensity or only over the central line ('1D' or '2D')
    :var float self.t_max: Cutoff time for the lifetime effect (in unit of the lifetime).\
    If set to 0, the lifetime will not be taken in account    
    :var bool self.lifetime: Choice between using the lifetime effect or not
    :var float self.tau_max: Upper limit for the lifetime of the excited particles.\
    It does not need to be exact, it is only for computing the limits of the mesh.
    :var int self.N_field: Number of interval for the field line interpolation
    :var str self.data_path: Path to the data
    :var self.beam: (:class:`Beam1D <FPSDP.Diagnostics.Beam.beam.Beam1D>`) Beam used for the diagnostic
    :var np.array[Ntime] self.time: Time steps used for the simulation
    :var tck_interp self.filter_: Interpolant of the filter
    :var float self.wl0: Wavelength of the de-excitation photons
    :var float self.Xmax: Upper limit of the X coordinate
    :var float self.Xmin: Lower limit of the X coordinate
    :var float self.Ymax: Upper limit of the Y coordinate
    :var float self.Ymin: Lower limit of the Y coordinate
    :var float self.Zmax: Upper limit of the Z coordinate
    :var float self.Zmin: Lower limit of the Z coordinate
    :var np.array[3,2] self.limits: Limits of the mesh (first index for X,Y,Z and second for max,min)
    """

    def __init__(self,input_file, parallel=False):
        """ load all the data from the input file"""
        self.cfg_file = input_file                                           #!
        if not exists(self.cfg_file):
            raise NameError('Config file not found')
        config = psr.ConfigParser()
        config.read(self.cfg_file)
        # variable for knowing if works in parallel or serie
        # only the computation of the bes intensity is done in parallel
        self.para = parallel                                                 #!
        
        # the example input file is well commented, look there for more
        # information
        
        # Optics part
        self.pos_lens = json.loads(config.get('Optics','pos_lens'))          #!
        self.pos_lens = np.array(self.pos_lens)
        self.rad_ring = json.loads(config.get('Optics','rad_ring'))          #!
        self.rad_lens = json.loads(config.get('Optics','rad_lens'))          #!
        self.inter = json.loads(config.get('Optics','int'))                  #!
        self.Nint = json.loads(config.get('Optics','Nint'))                  #!
        self.Nsol = json.loads(config.get('Optics','Nsol'))                  #!
    
        X = json.loads(config.get('Optics','X'))
        Y = json.loads(config.get('Optics','Y'))
        Z = json.loads(config.get('Optics','Z'))

        self.pos_foc = np.zeros((len(Z),3))                                  #!
        self.pos_foc[:,0] = X
        self.pos_foc[:,1] = Y
        self.pos_foc[:,2] = Z
        self.op_direc = self.pos_foc-self.pos_lens                           #!
        norm_ = np.sqrt(np.sum(self.op_direc**2,axis=1))
        if isinstance(self.rad_ring,float):
            self.rad_ring = self.rad_ring*np.ones(self.pos_foc.shape[0])
        
        # normalize the vector
        self.op_direc[:,0] /= norm_
        self.op_direc[:,1] /= norm_
        self.op_direc[:,2] /= norm_

        self.dist = np.sqrt(np.sum((self.pos_foc - self.pos_lens)**2,axis=1))#!

        self.type_int = config.get('Optics','type_int')                      #!



        self.t_max = json.loads(config.get('Collisions','t_max'))            #!
        self.lifetime = self.t_max != 0                                      #!
        
        # Data part
        self.tau_max = json.loads(config.get('Data','tau_max'))              #!
        self.N_field = json.loads(config.get('Data','N_field'))              #!
        self.data_path = config.get('Data','data_path')                      #!
        start = json.loads(config.get('Data','timestart'))
        end = json.loads(config.get('Data','timeend'))
        timestep = json.loads(config.get('Data','timestep'))
        filter_name = config.get('Data','filter')
        self.beam = be.Beam1D(input_file)                                    #!
        self.compute_limits()      # compute the limits of the mesh
        # position swap due to a difference in the axis        
        #grid3D = Grid.Cartesian3D(Xmin=self.Xmin, Xmax=self.Xmax, Ymin=self.Zmin, Ymax=self.Zmax,
        #                          Zmin=self.Ymin, Zmax=self.Ymax, NX=self.N[0], NY=self.N[2], NZ=self.N[1])
        xgc_ = xgc.XGC_Loader_BES(self.data_path, start, end, timestep,
                                  self.limits, self.N_field)
        self.time = xgc_.time_steps                                          #!

        # set the data inside the beam (2nd part of the initialization)
        self.beam.set_data(xgc_)
        print 'no check of division by zero'
        # compute the two others basis vector
        self.perp1 = np.zeros(self.pos_foc.shape)
        # put the first variable to 1
        self.perp1[:,0] = np.ones(self.pos_foc.shape[0])
        # check division by 0
        if (self.op_direc[:,1] == 0).any():
            raise NameError('Should implement the case where op_direc[:,1] == 0')
        
        self.perp1[:,1] = -self.op_direc[:,2]/self.op_direc[:,1]
        self.perp1[:,2] = np.zeros(self.pos_foc.shape[0])
        # norm of the vector for nomalization
        norm_ = np.sqrt(np.sum(self.perp1**2,axis=1))
        self.perp1[:,0] /= norm_
        self.perp1[:,1] /= norm_
        self.perp1[:,2] /= norm_

        # last basis vector
        self.perp2 = np.zeros(self.pos_foc.shape)
        for j in range(self.pos_foc.shape[0]):
            self.perp2[j,:] = np.cross(self.op_direc[j,:],self.perp1[j,:])

        # interpolant for the filter
        self.filter_ = self.load_filter(filter_name)                         #!
        # wavelength of the photon in the particles reference frame
        self.wl0 = self.beam.collisions.get_wavelength()                     #!


    def load_filter(self,filter_name):
        """ Load the data from the filter and compute the interpolant.

        

        :param str filter_name: Name of the file containing the filter datas

        :returns: Interpolant
        :rtype: tck_interp
        """
        data = np.loadtxt(filter_name)
        # wavelength in nanometer
        wl = data[:,0]
        # transmission coefficient
        trans = data[:,1]
        tck = sp.interpolate.interp1d(wl,trans,kind='cubic',fill_value=0.0)
        return tck
        
        
    def compute_limits(self, eps=0.05, dxmin = 0.1, dymin = 0.1, dzmin = 0.1):
        r""" Compute the limits of the mesh that should be loaded

        The only limitations comes from the sampling volume and the lifetime of the excited state.
        In the figure below, blue is for the beam and the lifetime effect, red for the ring and the cutoff values,
        straight black lines are for the sampling volume, and, the dashed one are the box.


        .. tikz::
           % beam
           \draw[blue] (-5,0.5) -- (7,0.5);
           \draw[blue] (-5,-0.5) -- (7,-0.5);
           \node[blue] at (-4.5,0) {Beam};
           \draw[->,thick,blue] (-3.5,0) -- (-2.5,0);
           % Sampling volume + ring
           \draw (3,-3.7) -- (2.8,-0.1);
           \draw (3.4,0.1) -- (6,-2.8);
           \draw (0.4,2.5) -- (2.8,-0.1);
           \draw (3.4,0.1) -- (3,2.9);
           \draw[red] (2.8,-0.1) -- (3.4,0.1);
           %cutoff
           \draw[red] (2.825,-1) -- (4,-0.56);
           \draw[red] (2.15,0.6) -- (3.25,1);
           %lifetime effect
           \draw[blue,->] (2.15,0.6) -- (1.65,0.6);
           \draw[blue,->] (2.825,-1) -- (2.325,-1);
           \draw[blue,->] (4,-0.56) -- (3.5,-0.56);
           \draw[blue,->] (3.25,1) -- (2.75,1);
           %box
           \draw[dashed] (1.5, 1.1) -- (1.5,-1.1) -- (4.1,-1.1) -- (4.1,1.1) -- cycle;

        :param float eps: Used for increasing the size of the box (relative size)
        :param float dxmin: Smallest size accepted for the box in X
        :param float dymin: Smallest size accepted for the box in Y
        :param float dzmin: Smallest size accepted for the box in Z
        
        """
        print('need to improve this ')
        # average beam width
        w = 0.5*(self.beam.stddev_h + self.beam.stddev_v)
        # size of the integration along the optical axis
        d = self.inter*w

        # position of the last value computed on the axis
        # the origin of this system is the center of the ring
        center_max = np.zeros((self.pos_foc.shape[0],3))
        center_max[:,0] = self.pos_foc[:,0] + \
                          d*self.op_direc[:,0]
        center_max[:,1] = self.pos_foc[:,1] + \
                          d*self.op_direc[:,1]
        center_max[:,2] = self.pos_foc[:,2] + \
                          d*self.op_direc[:,2]

        # position of the first value computed on the axis
        center_min = np.zeros((self.pos_foc.shape[0],3))
        center_min[:,0] = self.pos_foc[:,0] - \
                          d*self.op_direc[:,0]
        center_min[:,1] = self.pos_foc[:,1] - \
                          d*self.op_direc[:,1]
        center_min[:,2] = self.pos_foc[:,2] - \
                          d*self.op_direc[:,2]

        # width of the sampling volume at the two positions
        w_min = np.zeros(self.pos_foc.shape[0])
        w_max = np.zeros(self.pos_foc.shape[0])

        # compute distance from the center of the lens
        pos_optical_min = center_min - self.pos_lens
        pos_optical_max = center_max - self.pos_lens

        # compute the width
        for k in range(self.pos_foc.shape[0]):
            w_min[k] = self.get_width(pos_optical_min[k,:],k)
            w_max[k] = self.get_width(pos_optical_max[k,:],k)
        # first in X
        self.Xmax = np.max([center_max[:,0] + w_max,
                            center_min[:,0] + w_min])

        self.Xmin = np.min([center_max[:,0] - w_max,
                            center_min[:,0] - w_min])
        # second in Y
        self.Ymax = np.max([center_max[:,1] + w_max,
                            center_min[:,1] + w_min])

        self.Ymin = np.min([center_max[:,1] - w_max,
                            center_min[:,1] - w_min])
        # third in Z
        self.Zmax = np.max([center_max[:,2] + w_max,
                            center_min[:,2] + w_min])

        self.Zmin = np.min([center_max[:,2] - w_max,
                            center_min[:,2] - w_min])

        # direction of the beam
        be_dir = self.beam.direc

        # distance max used for the lifetime effect
        l = max(self.beam.speed)*self.tau_max*self.beam.t_max

        # take in account the lifetime effect
        self.Xmax = max([self.Xmax, self.Xmax - be_dir[0]*l])
        self.Xmin = min([self.Xmin, self.Xmin - be_dir[0]*l])
        self.Ymax = max([self.Ymax, self.Ymax - be_dir[1]*l])
        self.Ymin = min([self.Ymin, self.Ymin - be_dir[1]*l])
        self.Zmax = max([self.Zmax, self.Zmax - be_dir[2]*l])
        self.Zmin = min([self.Zmin, self.Zmin - be_dir[2]*l])


        # try to keep an interval big enough
        dX = self.Xmax-self.Xmin
        dY = self.Ymax-self.Ymin
        dZ = self.Zmax-self.Zmin
        if dX < dxmin:
            dX = dxmin
            av = 0.5*(self.Xmin + self.Xmax)
            self.Xmin = av - 0.5*dX
            self.Xmax = av + 0.5*dX
        if dY < dymin:
            dY = dymin
            av = 0.5*(self.Ymin + self.Ymax)
            self.Ymin = av - 0.5*dY
            self.Ymax = av + 0.5*dY
        if dZ < dzmin:
            dZ = dzmin
            av = 0.5*(self.Zmin + self.Zmax)
            self.Zmin = av - 0.5*dZ
            self.Zmax = av + 0.5*dZ
        # add a small border to the box in order to avoid
        # the values outside the mesh
        self.Xmax += dX*eps
        self.Xmin -= dX*eps
        self.Ymax += dY*eps
        self.Ymin -= dY*eps
        self.Zmax += dZ*eps
        self.Zmin -= dZ*eps

        self.limits = np.array([[self.Xmin,self.Xmax],
                                [self.Ymin,self.Ymax],
                                [self.Zmin,self.Zmax]])

        print self.limits


    def get_bes(self):
        """ Compute the image of the density turbulence.
        This function should be the only one used outside the class.
        
        :returns: Intensity collected by each fiber (number of photons)
        :rtype: np.array[Ntime, Nfib]
        """
        print self.time
        nber_fiber = self.pos_foc.shape[0]
        print 'do not take in account the wavelenght'
        print 'only the main component is used'
        I = np.zeros((len(self.time),nber_fiber))
        for i,time in enumerate(self.time):
            print('Time step number: ' + str(i+1) + '/' + str(len(self.time)))
            # first time step already loaded
            if i != 0:
                self.beam.data.load_next_time_step()
            if self.para:
                p = mp.Pool()
                a = np.array(p.map(self.intensity_para, range(nber_fiber)))
                # sum the light from the different component
                I[i,:] = a
                # serial case
                p.close()
            else:
                for j in range(nber_fiber):
                    # compute the light received by each fiber
                    t_ = self.beam.data.current
                    # sum the light from the different component
                    I[i,j] = self.intensity(i,j)
        return I
        
    def intensity_para(self,i):
        """ Same as :func:`intensity`, but have only one argument.
        The only use is for the parallelization that ask only one argument.
        Use the variable current from the data loader.

        :param int i: Index of a fiber
        :returns: Intensity received by the fiber (number of photons per second)
        :rtype: float
        """
        t_ = self.beam.data.current
        return self.intensity(t_,i)

    def to_cart_coord(self,pos,fiber_nber):
        """ Change the optical coordinate to the cartesian system

        :param np.array[N,3] pos: Position in the optical system
        :param int fiber_nber: Index of the fiber

        :returns: Position in the cartesian system
        :rtype: np.arrray[N,3]
        """
        # use the three basis vectors for computing the vectors in the
        # cartesian coordinate
        if len(pos.shape) == 1:
            pos = pos[np.newaxis,:]
            
        ret = np.zeros(pos.shape)
        ret[:,0] = self.pos_lens[0] + self.op_direc[fiber_nber,0]*pos[:,2]
        ret[:,0] += self.perp1[fiber_nber,0]*pos[:,0] + self.perp2[fiber_nber,0]*pos[:,1]
        ret[:,1] = self.pos_lens[1] + self.op_direc[fiber_nber,1]*pos[:,2]
        ret[:,1] += self.perp1[fiber_nber,1]*pos[:,0] + self.perp2[fiber_nber,1]*pos[:,1]
        ret[:,2] = self.pos_lens[2] + self.op_direc[fiber_nber,2]*pos[:,2]
        ret[:,2] += self.perp1[fiber_nber,2]*pos[:,0] + self.perp2[fiber_nber,2]*pos[:,1]
        return ret

    def get_width(self,pos,fiber_nber):
        r""" Compute the radius of the light cone.
        Assume two cones that meet at the focus disk.

        .. tikz::
           \draw (3,3) -- (-2.5,0) -- (3,-3);
           \draw[ultra thick] (3,3) -- (3,-3);
           \node at (3.5,1.5) {Lens};
           \draw[ultra thick] (0,1.35) -- (0,-1.35);
           \node at (0, 1.8) {Ring};
           \draw[dashed] (-3,3) -- (2.5,0) -- (-3,-3);
           \draw[-{Triangle[angle=45:5pt 5]}] (3.6,0) to (-3,0);
           \draw (-1.2,0) -- (-1.2,0.7);
           %first pos
           \node at (-1.5,0.3) {$r_1$};
           \node at (-1.2,-0.5) {x};
           \node at (-0.6,-0.5) {Pos$_1$};
           % second pos
           \draw (1.9,0) -- (1.9,0.33);
           %\node at (1.9,0.1) {$r_2$};
           \node at (1.9,0.6) {x};
           \node at (2.45,0.6) {Pos$_2$};
           :libs: arrows.meta

        :param np.array[N,3] pos: Position where to compute the width in the optical system
        :param int fiber_nber: Index of the fiber

        :returns: Radius of the optical cone
        :rtype: np.array[N]
        """
        
        if len(pos.shape) == 1:
            pos = pos[np.newaxis,:]

        # distance from the ring
        a = abs(pos[:,2]-self.dist[fiber_nber])
        a *= (self.rad_lens-self.rad_ring[fiber_nber])/self.dist[fiber_nber]
        return a + self.rad_ring[fiber_nber]
    
    def check_in(self,pos,fib):
        r""" Check if the position (optical coordinate) is inside the first cone (blue area)
        (if the focus ring matter or not).
        The shape of the sampling area is asume to be linear along the z-axis (optic direction).

        .. tikz::
           \draw[fill=blue] (-1,0) -- (3,2) -- (3,-2) -- cycle;
           \draw[ultra thick] (3,2) -- (3,-2);           
           \draw[thick] (0,0.5) -- (0,-0.5);
           \draw[dashed] (-3,-2) -- (1,0) -- (-3,2);
           \node at (3.5, 0) {Lens};
           \node at (0,0.9) {Ring};

        :param np.array[N,3] pos: Position in the optical system
        :param int fib: Index of the fiber

        :returns: True if inside the first cone
        :rtype: np.array[N] of bool
        """
        ret = np.zeros(pos.shape[0], dtype=bool)
        # before the focus point
        ind = pos[:,2] < self.dist[fib]
        ret[ind] = True
        # distance from the focus point along the z-axis
        a = pos[~ind,2]-self.dist[fib]
        # size of the 'ring' scaled to this position
        a = a*(self.rad_ring[fib]-self.rad_lens)/self.dist[fib] + self.rad_ring[fib]
        # distance from the axis
        R = np.sqrt(np.sum(pos[~ind,0:2]**2, axis=1))
        ind1 = a > R
        temp = np.zeros(np.sum(~ind), dtype=bool)
        temp[ind1] = True
        ret[~ind] = temp

        return ret
        
    def light_from_plane(self,z, t_, fiber_nber):
        r""" Compute the light from one plane using a method of order 10 (see report or
        Abramowitz and Stegun).
        
        .. math::
           I_\text{plane} = \iint_D f(x) \mathrm{d}\sigma \approx \sum_i \omega_i f(x_i)
        
        where :math:`f(x)` is the value obtained by :func:`get_emis_from`, D is the disk
        representing the plane, and, :math:`\omega_i` and :math:`x_i` are the weights and the points
        of the quadrature formula.

        The points are given in the figure below and the weights are :math:`\frac{1}{9}` for the center,
        :math:`\frac{16\pm\sqrt{6}}{360}` for the innermost circle (plus sign) and the outermost circle (minus sign)

        .. tikz::
           \draw (0,0) circle(3);
           \foreach \i in {1,...,10}
           {
              \fill ({2.757*cos(36*\i)},{2.757*sin(36*\i)}) circle(2pt);
              \fill ({1.788*cos(36*\i)},{1.788*sin(36*\i)}) circle(2pt);
           }
           \fill (0,0) circle(2pt);

        :param np.array[N] z: Distance from the fiber along the sightline
        :param int t_: Time step to compute (is not important for the data loader, but is used as a check)
        :param int fiber_nber: Index of the fiber

        :returns: Intensity collected by the fiber from these planes
        :rtype: np.array[N]
        """
        I = np.zeros(z.shape[0])
        if self.type_int == '2D':
            # compute the integral with a few points
            # outside the central line
            center = np.zeros((len(z),3))
            center[:,2] = z # define the center of the circle
            # redius of the sampling plane
            r = self.get_width(center,fiber_nber)
            for i,r_ in enumerate(r):
                # integration points
                quad = integ.integration_points(2, 'order10', 'disk', r_)
                pos = np.zeros((quad.pts.shape[0],3))
                pos[:,0] = quad.pts[:,0]
                pos[:,1] = quad.pts[:,1]
                pos[:,2] = z[i]*np.ones(quad.pts.shape[0])
                eps = self.get_emis_from(pos,t_,fiber_nber)
                # sum the emission of all the points with the appropriate
                # weight
                I[i] = np.sum(quad.w*eps)
        elif self.type_int == '1D':
            # just use the point on the central line
            for i,z_ in enumerate(z):
                pos = np.array([0,0,z_])
                I[i] = self.get_emis_from(pos[np.newaxis,:],t_,fiber_nber)
        else:
            raise NameError('This type of integration does not exist')
        return I

    def intensity(self,t_,fiber_nber):
        r""" Compute the light received by a fiber at one time step.
        
        Use a Gauss-Legendre quadrature formula of order 3.
        
        .. math::
           I = \int_{-d}^d f(z) \mathrm{d}z \approx 
           \sum_i \frac{b_i-a_i}{2} \sum_j \omega_j f\left(\frac{b_i-a_i}{2}x_j + \frac{a_i+b_i}{2}\right)

        where the index i is for the splitting in subintervals, j is for the Gauss-Legendre formula,
        :math:`f(z) = \frac{I_\text{plane}(z)}{A(z)}` is the light emitted by the planes normalized by its area,
        :math:`d = \text{inter} \cdot w`, inter is the cutoff in unit of the average beam width (w),
        :math:`a_i` and :math:`b_i` are the lower and upper limits for each intervals, :math:`\omega` and :math:`x_i` are
        the weights and points of the quadrature formula. See figure :func:`compute_limits` for a view of the situation.

        :param int t_: Time step to compute
        :param int fiber_nber: Index of the fiber

        :returns: Intensity of light collected by the fiber
        :rtype: float
        """
        # first define the quadrature formula
        quad = integ.integration_points(1,'GL5') # Gauss-Legendre order 5
        I = 0.0
        # compute the distance from the origin of the beam
        dist = np.dot(self.pos_foc[fiber_nber,:] - self.beam.pos,self.beam.direc)
        width = self.beam.get_width(dist)
        # compute the average beam width of the beam
        width = 0.5*(width[0] + width[1])*self.inter
        # limit of the intervals
        border = np.linspace(-width,width,self.Nint)
        # value inside the intervals
        Z = 0.5*(border[:-1] + border[1:])
        # half size of one interval
        ba2 = 0.5*(border[2]-border[1])
        for z in Z:
            # distance of the plane from the lense
            pt = z + ba2*quad.pts + self.dist[fiber_nber]
            light = self.light_from_plane(pt,t_,fiber_nber)
            # sum the weight with the appropriate pts
            I += np.sum(quad.w*light)
        # multiply by the weigth of each interval
        I *= ba2
        return I
        
    def get_emis_from(self,pos,t_,fiber_nber):
        """ Compute the total emission received from a position (takes in account the
        solid angle and the filter).
        As long as it is possible, the effect of the optic should be written here (a lot more efficient than
        keeping all the wavelength until the end).

        :todo: Improvement possible: keep in memory the solid angle for different time step

        :param np.array[N,3] pos: Position in the optical system 
        :param int t_: Time step to compute
        :param int fiber_nber: Index of the fiber

        :returns: Intensity collected from each point
        :rtype: np.array[N]
        """
        # first change coordinate: optical -> cartesian (Tokamak)
        x = self.to_cart_coord(pos,fiber_nber)
        if self.lifetime:
            # choose if we use the lifetime effect or not
            eps = self.beam.get_emis_lifetime(x,t_)/(4.0*np.pi)
        else:
            eps = self.beam.get_emis(x,t_)/(4.0*np.pi)
        # now compute the solid angle
        solid = self.get_solid_angle(pos,fiber_nber)

        # compute the effect of the filter
        dist_ = self.pos_lens[np.newaxis,:] - pos
        dist_ = dist_/np.sqrt(np.sum(dist_**2,axis=-1)[:,np.newaxis])

        costh = np.einsum('ij,j->i',dist_,self.beam.direc)

        wl = (1.0 - self.beam.speed[:,np.newaxis]*costh/SI['c'])
        wl *= self.wl0
        filt = self.filter_(wl)
        # sum the intensity of all the components
        eps = np.sum(eps*filt,axis=0)
        return eps*solid

    def get_solid_angle(self,pos,fib):
        r""" Compute the solid angle 

        Three different cases can happen:

        * Lens case
        * Ring case
        * mixed case

        In the following drawing, the vision from a particle that emits some photons is shown.
        The red circles are for the lens and the black ones are for the ring.

        .. tikz::
           % lens case
           \draw[red] (-5,0) circle(1.5);
           \draw (-5,0.3) circle(2);
           \node at (-5,2.5) {Lens Case};
           % ring case 
           \draw (0,0) circle(1.5);
           \draw[red] (0,0.3) circle(2);
           \node at (0,2.5) {Ring Case};
           % mixed case
           \draw (5,0) circle(1.5);
           \draw[red] (5,0.3) circle(1.4);
           \node at (5,2.5) {Mixted Case};


        The two first are solved with the formula of Paxton (:func:`solid_angle_disk <FPSDP.Maths.Funcs.solid_angle_disk`) and
        the last one is solved numerically.

        :param np.array[N,3] pos: Position in the optical system
        :param int fib: Index of the fiber

        :returns: Solid angle
        :rtype: np.array[N]
        """
        test = self.check_in(pos,fib)
        #ind2 = np.where(~test)[0]
        solid = np.zeros(pos.shape[0])

        # first case (look at the report about this code)
        solid[test] = solid_angle_disk(pos[test,:],self.rad_lens)
        # second case
        # first find the position of the 'intersection' between the lens and the ring
        # define a few constant (look my report for the detail, too much computation and
        # need some drawing to write them in the comments)

        if ((pos[~test,0] == 0) & (pos[~test,1] != 0)).any():
            print ~test
            print pos[~test,:]
            raise NameError('pos[:,0] == 0 gives a division by 0')
        # ratio between the point P and the distance ring-lens
        ratio = np.abs(pos[~test,2]/self.dist[fib])
        f = 1.0/(1.0-ratio)
        A = 0.5*((np.sum(pos[~test,0:2]**2,axis=1)-(self.rad_lens/f)**2)/ratio + ratio*self.rad_ring[fib]**2)/pos[~test,0]
        B = -pos[~test,1]/pos[~test,0]
        delta = 4*B**2*A**2 - 4*(A**2-self.rad_ring[fib]**2)*(B**2+1)
        ind = (delta > 0) & (~np.isnan(delta)) & (~np.isinf(delta))
        temp = np.zeros(np.sum(~test))
        if ind.any():
            # x1 = plus sign
            delta = np.sqrt(delta[ind])
            x1 = np.zeros((sum(ind),2))
            x1[:,1] = (-2*B[ind]*A[ind] + delta)/(2*(B[ind]**2+1))
            x1[:,0] = A[ind] + B[ind]*x1[:,1]

            x2 = np.zeros((sum(ind),2))
            x2[:,1] = (-2*B[ind]*A[ind] - delta)/(2*(B[ind]**2+1))
            x2[:,0] = A[ind] + B[ind]*x2[:,1]

            y1 = ((pos[~test,:][ind,0:2].T-x1.T*ratio[ind])*f[ind]).T
            y2 = ((pos[~test,:][ind,0:2].T-x2.T*ratio[ind])*f[ind]).T

            temp[ind] = self.solid_angle_mix_case(pos[~test,:][ind,:],[x1, x2],[y1, y2],fib)
        # second case
        ind = ~ind
        if ind.any():
            q = pos[~test,:][ind,:]
            q[:,2] -= self.dist[fib]
            temp[ind] = solid_angle_disk(q,self.rad_ring[fib])
        solid[~test] = temp
        if (solid < 0).any() or (solid > 4*np.pi).any():
            print('solid angle',solid)
            print('check_in',test)
            print('ind',ind)
            raise NameError('solid angle smaller than 0 or bigger than 4pi')
        return solid

    def solid_angle_mix_case(self,pos,x,y,fib):
        r""" Compute numerically the solid angle for the mixted case
        (where the lens AND the ring limit the size of the solid angle)

        The view from the emission point is given in the figure below.
        The light collected by the fiber is within the continuous lines.

        .. tikz::
           \draw [red,dashed,domain=115:180] plot ({4*cos(\x)}, {4*sin(\x)});
           \draw [red,dashed,domain=360:425] plot ({4*cos(\x)}, {4*sin(\x)});
           \draw [black,thick,domain=150:390] plot ({2*cos(\x)}, {8/3+2*sin(\x)});
           \draw [red,thick,domain=65:115] plot ({4*cos(\x)}, {4*sin(\x)});
           \draw [black,dashed,domain=30:150] plot ({2*cos(\x)}, {8/3+2*sin(\x)});
           \node at ({-5*2/3},0) {Lens};
           \node at ({2.4*2/3},{2/3}) {Ring};
           \node at ({2.66*2/3},{5.38*2/3}) {x};
           \node at ({3.2*2/3},{5.8*2/3}) {$x_2$,$y_2$};
           \node at ({-2.66*2/3},{5.38*2/3}) {x};
           \node at ({-3.2*2/3},{5.8*2/3}) {$x_1$,$y_1$};


        :param np.array[N,3] pos: Position in the optical system
        :param list[np.array[N],..] x: Position of the intersection on the ring (list contains 2 elements) 
        :param list[np.array[N],..] y: Position of the intersection on the lens (list contains 2 elements)
        :param int fib: Index of the fiber
        """
        # first the contribution of the ring
        omega = self.solid_angle_seg(pos-np.array([0,0,self.dist[fib]]),x,
                                     self.rad_ring[fib],0)
        # second the contribution of the lens
        omega +=self.solid_angle_seg(pos,y,self.rad_lens,1)
        return omega


    def solid_angle_seg(self,pos,x,r,islens):
        r""" Compute the solid angle of a disk where a segment has been removed.

        First, the numerical integration will be carried out over the biggest area of the disk,
        and, in a second time, if necessary, the integral over the full disk is computed
        (with the analytical formula) and subtracted by the numerical integral.

        The idea is to compute numerically the 2D integral by splitting the domain in 
        sector of the same angle and doing a Gauss-Legendre quadrature formula over
        each dimension.

        In a first time, the maximum radius (that will depends on the coordinate :math:`\omega`)
        has to be compute.

        In this figure, we want to compute the area between the black line and the blue one

        .. tikz::
           \draw [red,dashed,domain=115:180] plot ({6*cos(\x)}, {6*sin(\x)});
           \draw [red,dashed,domain=360:425] plot ({6*cos(\x)}, {6*sin(\x)});
           \draw [black,thick,domain=150:390] plot ({3*cos(\x)}, {4+3*sin(\x)});
           \draw [red,thick,domain=65:115] plot ({6*cos(\x)}, {6*sin(\x)});
           \draw [black,dashed,domain=30:150] plot ({3*cos(\x)}, {4+3*sin(\x)});
           \draw [domain=-10:80] plot ({0.8*cos(\x)}, {4+0.8*sin(\x)});
           \node at (1,4.6) {$\theta$};
           \node at (-5,0) {Lens};
           \node at (2.4,1) {Ring};
           \node at (0,0) {x};
           \node at (0,4) {x};
           \draw (0,4) -- (0.51,6.94);
           \draw (0,4) -- ({3*cos(-10)}, {4+3*sin(-10)});
           \node at (2.66,5.38) {x};
           \node at (3.2,5.8) {$x_2$,$y_2$};
           \node at (-2.66,5.38) {x};
           \node at (-3.2,5.8) {$x_1$,$y_1$};
           \draw [blue] (-2.66,5.38) -- (2.66,5.38);
           \node at (0.25,5.4) {x};
           \node at (0.8,5.2) {$r_{max}$};

        
        :todo: improvement: remove useless computation of rmax
        :param np.array[N,3] pos: Position in the optical system
        :param list[np.array[N],..] x: Position of the intersection on the ring (list contains 2 elements) 
        :param float r: Radius of the disk (should be centered at (0,0,0) and the perpendicular should be along the z-axis)
        :param bool islens: True if the computation is for the lens (change of sign if it is the case)
        """

        # split the two intersections in two variables
        x1 = x[0]
        x2 = x[1]
        
        # limits (in angle) considered for the integration
        theta = np.linspace(0,2*np.pi,self.Nsol)
        quadr = integ.integration_points(1,'GL5') # Gauss-Legendre order 5
        quadt = integ.integration_points(1,'GL5') # Gauss-Legendre order 5

        # mid point of the limits in theta
        av = 0.5*(theta[:-1] + theta[1:])
        # half size of the intervals in theta
        diff = 0.5*np.diff(theta)
        # array containing all the value of theta that will be computed
        th = ((diff[:,np.newaxis]*quadt.pts).T + av).T

        # perpendicular vector to x1->x2
        perp = -pos[:,0:2]

        # indices where we want to compute the big part
        ind = np.einsum('ij,ij->i',perp,x1) > 0
        if islens:
            ind = ~ind
        perp[~ind] = -perp[~ind]
        perp = (perp.T/np.sqrt(np.sum(perp**2,axis=1))).T

        # unit vector for each angle
        delta = np.array([np.cos(th),np.sin(th)])
        delta = np.rollaxis(delta,0,3)
        # now detla[Nsol-1,quadt,dim]

        # compute the scalar product (=> the cos)
        cospsi = np.einsum('ak,ijk->aij',perp,delta)
        # index where the line can cross the segment
        ind2 = cospsi > 0

        # distance between line
        d = np.abs(x1[:,0]*x2[:,1]-x2[:,0]*x1[:,1])/np.sqrt(np.sum((x2-x1)**2,axis=1))

        #print('useless computations')
        #:todo: This can be improved
        # compute the distance along theta where the segment is crossed
        rmax = ((1.0/cospsi).T*d).T
        # if the line cannot be cross, therefore the computation can raise some trouble
        # => set it manually to the good value
        rmax[~ind2] = r
        # take the min between the intersection with the segment and the circle
        rmax = np.minimum(r,rmax)

        # array containing the evaluation of the function that will be integrated
        R = np.zeros((pos.shape[0],self.Nsol-1,quadt.pts.shape[0],
                      quadr.pts.shape[0],3))
        # radius that will be computed
        temp = (0.5*rmax[...,np.newaxis]*(quadr.pts+1.0))
        R[...,0] = pos[:,np.newaxis,np.newaxis,np.newaxis,0]\
                   + temp*delta[...,np.newaxis,0]
        R[...,1] = pos[:,np.newaxis,np.newaxis,np.newaxis,1]\
                   + temp*delta[...,np.newaxis,1]
        R[...,2] = pos[:,np.newaxis,np.newaxis,np.newaxis,2]

        # compute the norm of the vector
        R = np.sum(R**2,axis=4)**(-1.5)
        # sum over all the index (theta and r quadrature formula, and, sector)
        omega = np.sum(0.5*diff*np.sum(rmax*np.sum(temp*R*quadr.w,axis=3)*quadt.w,axis=2),axis=1)

        # multiply by the scalar product between the position and the normal (to the disk) vector 
        omega *= np.abs(pos[:,2])

        # change the area that we want to compute
        omega[~ind] = solid_angle_disk(pos[~ind,:],r)-omega[~ind]
        return omega














class BES_ideal:
    """ Take the output of the simulation and just 
        compute the fluctuation
        A lot of copy and paste from the BES class, therefore look there for
        the comments
    """
    def __init__(self,input_file,mesh=False):
        """ load all the data from the input file mesh is used for knowing
            if the focus points are used or if a mesh is created one the max/min
            value of the focus points
        """
        self.cfg_file = input_file                                           #!
        if not exists(self.cfg_file):
            raise NameError('Config file not found')
        config = psr.ConfigParser()
        config.read(self.cfg_file)

        self.mesh = mesh

        X = json.loads(config.get('Optics','X'))
        Y = json.loads(config.get('Optics','Y'))
        Z = json.loads(config.get('Optics','Z'))

        self.pos_foc = np.zeros((len(Z),3))                                  #!
        self.pos_foc[:,0] = X
        self.pos_foc[:,1] = Y
        self.pos_foc[:,2] = Z

        # Data part
        self.N_field = json.loads(config.get('Data','N_field'))              #!
        self.data_path = config.get('Data','data_path')                      #!
        start = json.loads(config.get('Data','timestart'))
        end = json.loads(config.get('Data','timeend'))
        timestep = json.loads(config.get('Data','timestep'))
        self.compute_limits()      # compute the limits of the mesh

        xgc_ = xgc.XGC_Loader_BES(self.data_path, start, end, timestep,
                                  self.limits, self.N_field)
        self.time = xgc_.time_steps                                          #!

        self.data = xgc_
        if (self.time != xgc_.time_steps).any():
            raise NameError('Time steps wrong')
        

    def compute_limits(self, eps=1, dxmin = 0.1, dymin = 0.1, dzmin = 0.5):
        """ find min/max of the focus points """
        # first in X
        self.Xmax = np.max(self.pos_foc[:,0])

        self.Xmin = np.min(self.pos_foc[:,0])
        # second in Y
        self.Ymax = np.max(self.pos_foc[:,1])

        self.Ymin = np.min(self.pos_foc[:,1])
        # third in Z
        self.Zmax = np.max(self.pos_foc[:,2])

        self.Zmin = np.min(self.pos_foc[:,2])

        # try to keep an interval big enough
        dX = self.Xmax-self.Xmin
        dY = self.Ymax-self.Ymin
        dZ = self.Zmax-self.Zmin
        if dX < dxmin:
            dX = dxmin
            av = 0.5*(self.Xmin + self.Xmax)
            self.Xmin = av - 0.5*dX
            self.Xmax = av + 0.5*dX
        if dY < dymin:
            dY = dymin
            av = 0.5*(self.Ymin + self.Ymax)
            self.Ymin = av - 0.5*dY
            self.Ymax = av + 0.5*dY
        if dZ < dzmin:
            dZ = dzmin
            av = 0.5*(self.Zmin + self.Zmax)
            self.Zmin = av - 0.5*dZ
            self.Zmax = av + 0.5*dZ
        self.Xmax += dX*eps
        self.Xmin -= dX*eps
        self.Ymax += dY*eps
        self.Ymin -= dY*eps
        self.Zmax += dZ*eps
        self.Zmin -= dZ*eps
        self.limits = np.array([[self.Xmin,self.Xmax],
                                [self.Ymin,self.Ymax],
                                [self.Zmin,self.Zmax]])


    def get_bes(self):
        """ Compute the image of the turbulence in density
            This function should be the only one used outside the class
        """
        if self.mesh:
            print('Need to edit for special case')
            a = np.linspace(self.Xmin,self.Xmax,100)
            b = np.linspace(self.Zmin,self.Zmax,100)
            a,b = np.meshgrid(a,b)
            a = np.reshape(a,-1)
            b = np.reshape(b,-1)
            self.pos_foc = np.zeros((a.shape[0],3))
            self.pos_foc[:,0] = a
            self.pos_foc[:,2] = b
            
        print self.time
        nber_fiber = self.pos_foc.shape[0]
        I = np.zeros((len(self.time),nber_fiber))
        for i,time in enumerate(self.time):
            print('Time step number: ' + str(i+1) + '/' + str(len(self.time)))
            if i != 0:
                self.data.load_next_time_step()
            for j in range(nber_fiber):
                # compute the light received by each fiber
                t_ = self.data.current
                I[i,j] = self.intensity(i,j)
        return I


    def intensity(self,t_,fiber_nber):
        """ Compute the light received by the fiber
        """
        I = self.data.interpolate_data(self.pos_foc[fiber_nber,:],t_,['ne'],False)[0]
        return I
 
