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
# data loader
import FPSDP.Plasma.XGC_Profile.load_XGC_local as xgc
# hdf5
import h5py as h5
# quadrature formula and utility functions
import FPSDP.Maths.Integration as integ
from FPSDP.GeneralSettings.UnitSystem import SI
from FPSDP.Maths.Funcs import heuman, solid_angle_disk,\
    solid_angle_seg, compute_threshold_solid_angle
    


from os.path import exists # used for checking if the input file exists
# it is not clear with configparser error

from sys import exc_info

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
    
    Load the parameter from a config file (example in :download:`bes.in <../../../../FPSDP/Diagnostics/BES/bes.in>`
    with a lot of comments for explaining all the parameters) and create everything from it.
    The function :func:`get_bes <FPSDP.Diagnostics.BES.bes.BES.get_bes>` is used for computing the photon radiance received by
    each fiber (number of photons per second).

    :param str input_file: Name of the config file
    :param bool multiprocessing: Choice between the serial code or the parallel one
    :var str self.cfg_fil: Name of the config file
    :var bool self.multiprocessing: Choice between the serial code and the parallel one
    :var np.array[3] self.pos_lens: Position of the lens (in the cartesian system) in meter
    :var np.array[Nfib] self.rad_foc: Radius of the focus point for each fiber in meter
    :var float self.rad_lens: Radius of the lens in meter
    :var float self.inter: Cutoff distance from the focus point (in unit of the beam width).\
    Look at the figure in :func:`compute_limits <FPSDP.Diagnostics.BES.bes.BES.compute_limits>`, the two outer red lines for the limit. 
    :var int self.Nint: Number of point for splitting the integral over the optical direction
    :var int self.Nsolid: Number of intervals for the evaluation of the solid angle in the mixed case\
    Look at the figure in :func:`get_solid_angle  <FPSDP.Diagnostics.BES.bes.BES.get_solid_angle>`.
    :var int self.Nr: Number of intervals for the radial integration of the solid angle
    :var np.array[Nfib,Nint-1,2,21] self.solid: Solid angle for all the position that are computed.\
    The axis number 2 is the number of points used by interval in the integration along the field line and\
    the axis number 3 is the number of points used by integral over a disk.
    :var np.array[Nfib,3] self.pos_foc: Position of the focus points in meter (in the cartesian system)
    :var np.array[Nfib,3] self.op_direc: Direction of the optical line (for each fiber)
    :var np.array[Nfib] self.dist: Distance between the focus point and the lens in meter
    :var np.array[Nfib] self.lim_op: Distance for switching between different case in the solid angle (see :func:`find_case`).
    :var np.array[Nfib,3] self.perp1: Second basis vector for each fiber coordinates (first is the optical line)
    :var np.array[Nfib,3] self.perp2: Third basis vector
    :var str self.type_int: choice between the full computation of the photo radiance or only over the central line ('1D' or '2D')
    :var float self.t_max: Cutoff time for the lifetime effect (in unit of the lifetime).\
    If set to 0, the lifetime will not be taken in account    
    :var bool self.lifetime: Choice between using the lifetime effect or not
    :var float self.tau_max: Upper limit for the lifetime of the excited particles.\
    It does not need to be exact, it is only for computing the limits of the mesh.
    :var int self.dphi: Toroidal step for the field line interpolation (in radians)
    :var str self.data_path: Directory path to the data
    :var self.beam: (:class:`Beam1D <FPSDP.Diagnostics.Beam.beam.Beam1D>`) Beam used for the diagnostic
    :var np.array[Ntime] self.time: Time steps used for the simulation
    :var tck_interp self.filter_: Interpolant of the filter
    :var float self.lambda0: Wavelength of the de-excitation photons
    :var float self.Xmax: Upper limit of the X coordinate of the mesh
    :var float self.Xmin: Lower limit of the X coordinate of the mesh
    :var float self.Ymax: Upper limit of the Y coordinate of the mesh
    :var float self.Ymin: Lower limit of the Y coordinate of the mesh
    :var float self.Zmax: Upper limit of the Z coordinate of the mesh
    :var float self.Zmin: Lower limit of the Z coordinate of the mesh
    :var np.array[3,2] self.limits: Limits of the mesh (first index for X,Y,Z and second for max,min)

    The following graph shows the most important call during the initialization of the BES class.
    The red arrows show the call order and the black ones show what is inside the function.

    .. graphviz::
     
       digraph bes_init{
       compound=true;
       // BES.__INIT__

       subgraph cluster_besinit { label="BES.__init__"; "Beam1D.__init__"->compute_limits->
         "XGC_Loader_local.__init__"->"Beam1D.set_data"->"BES.load_filter"->"Collisions.get_wavelength"[color="red"]
       }
    
       A [label="load_XGC_local.get_interp_planes_local"];
       B [label="load_XGC_local.get_interp_planes_local"];


       // BEAM1D.__INIT__
       "Beam1D.__init__"->"Collisions.__init__" [lhead=cluster_Beam1D];
       subgraph cluster_Beam1D { label="Beam1D.__init__"; "Collisions.__init__";}

       "Collisions.__init__"->"Collisions.read_adas"[lhead=cluster_collisions];

       // COLLISIONS.__INIT__
       subgraph cluster_collisions { label="Collisions.__init__"; "Collisions.read_adas";}
       "Collisions.read_adas"->"ADAS_file.__init__" [lhead="cluster_read_adas"];
       subgraph cluster_read_adas { label="Collisions.read_adas"; "ADAS_file.__init__";}
       


       // XGC_LOADER_local.__INIT__
       "XGC_Loader_local.__init__"->"XGC_Loader_local.load_mesh_psi_3D" [lhead=cluster_XGC];

       subgraph cluster_XGC { label="XGC_Loader_local.__init__"; "XGC_Loader_local.load_mesh_psi_3D"->
       "XGC_Loader_local.load_B_3D"->A->"XGC_Loader_local.load_eq_3D"
       ->"XGC_Loader_local.load_next_time_step"[color="red"];}
       
       // XGC_LOADER_local.LOAD_NEXT_TIME_STEP
       "XGC_Loader_local.load_next_time_step"->"XGC_Loader_local.load_fluctuations_3D_all"[lhead=cluster_next];
       subgraph cluster_next { label="XGC_Loader_local.load_next_time_step"; "XGC_Loader_local.load_fluctuations_3D_all"->
       "XGC_Loader_local.calc_total_ne_3D"->"XGC_Loader_local.compute_interpolant"[color="red"];}

       // BEAM.SET_DATA
       "Beam1D.set_data"->"Beam1D.create_mesh" [lhead=cluster_set_data];    

       subgraph cluster_set_data { label="Beam1D.set_data"; "Beam1D.create_mesh"->"Beam1D.compute_beam_on_mesh"[color="red"];
       }
       "Beam1D.create_mesh"->"Beam1D.find_wall"[lhead=cluster_create_mesh];

       subgraph cluster_create_mesh { label="Beam1D.create_mesh"; "Beam1D.find_wall";}
       

       // BEAM.COMPUTE_BEAM_ON_MESH
       "Beam1D.compute_beam_on_mesh"->"Integration.integration_points"[lhead=cluster_compute_beam];

       subgraph cluster_compute_beam { label="Beam1D.compute_beam_on_mesh"; "Integration.integration_points"->
       "Beam1D.get_quantities"->"Collisions.get_attenuation"[color="red"]; }

       "Beam1D.get_quantities"->"XGC_Loader_local.interpolate_data" [lhead=cluster_quantities];
        subgraph cluster_quantities { label="Beam1D.get_quantities"; "XGC_Loader_local.interpolate_data"}

       // XGC_LOADER_local.INTERPOLATE_DATA
       "XGC_Loader_local.interpolate_data"->B [lhead=cluster_interpolate];
       subgraph cluster_interpolate { label="XGC_Loader_local.interpolate_data"; B->
       "XGC_Loader_local.find_interp_positions"[color="red"];
       }

       }    

    """

    def __init__(self,input_file, multiprocessing=False):
        """ load all the data from the input file"""
        self.cfg_file = input_file                                           #!
        if not exists(self.cfg_file):
            raise NameError('Config file not found')
        config = psr.ConfigParser()
        config.read(self.cfg_file)
        # variable for knowing if works in parallel or serie
        # only the computation of the bes intensity is done in parallel
        self.multiprocessing = multiprocessing                               #!
        self.data_path = config.get('Data','data_path')                      #!
        start = json.loads(config.get('Data','timestart'))
        # the example input file is well commented, look there for more
        # information
        
        # Optics part
        self.pos_lens = json.loads(config.get('Optics','pos_lens'))          #!
        self.pos_lens = np.array(self.pos_lens)
        self.rad_foc = json.loads(config.get('Optics','rad_foc'))            #!
        self.rad_lens = json.loads(config.get('Optics','rad_lens'))          #!
        self.inter = json.loads(config.get('Optics','int'))                  #!
        self.Nint = json.loads(config.get('Optics','Nint'))                  #!
        self.Nsolid = json.loads(config.get('Optics','Nsolid'))              #!
        self.Nr = json.loads(config.get('Optics','Nr'))                      #!

        if self.Nint/self.inter < 2:
            print 'WARNING: The accuracy of the optical integral is assumed to be too small'
        
        R = json.loads(config.get('Optics','R'))
        R = np.array(R)
        phi = json.loads(config.get('Optics','phi'))
        Z = json.loads(config.get('Optics','Z'))
        plane = json.loads(config.get('Optics','plane'))
        # compute the value of phi in radian
        print 'should be changed if use another code than xgc'
        name = self.data_path + 'xgc.3d.' + str(start).zfill(5)+'.h5'
        nber_plane = h5.File(name,'r')
        nphi = nber_plane['nphi'][:]
        shift = np.mean(phi) - 2*np.pi*plane/nphi[0]

        
        self.pos_foc = np.zeros((len(Z),3))                                  #!
        self.pos_foc[:,0] = R*np.cos(phi)
        self.pos_foc[:,1] = R*np.sin(phi)
        self.pos_foc[:,2] = Z
        self.op_direc = self.pos_foc-self.pos_lens                           #!
        
        norm_ = np.sqrt(np.sum(self.op_direc**2,axis=1))
        if isinstance(self.rad_foc,float):
            self.rad_foc = self.rad_foc*np.ones(self.pos_foc.shape[0])
        
        # normalize the vector
        self.op_direc[:,0] /= norm_
        self.op_direc[:,1] /= norm_
        self.op_direc[:,2] /= norm_

        self.dist = np.sqrt(np.sum((self.pos_foc - self.pos_lens)**2,axis=1))#!

        self.type_int = config.get('Optics','type_int')                      #!
        
        self.lim_op = np.zeros((len(Z),2))                                   #!
        self.lim_op[:,0] = self.dist*self.rad_lens/(self.rad_lens+self.rad_foc)
        self.lim_op[:,1] = self.dist*self.rad_lens/(self.rad_lens-self.rad_foc)


        self.t_max = json.loads(config.get('Collisions','t_max'))            #!
        self.lifetime = self.t_max != 0                                      #!
        
        # Data part
        self.tau_max = json.loads(config.get('Collisions','tau'))            #!
        self.dphi = json.loads(config.get('Data','dphi'))                    #!
        end = json.loads(config.get('Data','timeend'))
        timestep = json.loads(config.get('Data','timestep'))
        filter_name = config.get('Data','filter')
        order = config.get('Data','interpolation')
        self.beam = be.Beam1D(input_file)                                    #!
        self.compute_limits()      # compute the limits of the mesh
        lim = json.loads(config.get('Data','limits'))

        if  lim:
            limit = self.limits
        else:
            limit = self.limits[:2,:]
        xgc_ = xgc.XGC_Loader_local(self.data_path, start, end, timestep,
                                    limit, self.dphi,shift,order)
        self.time = xgc_.time_steps                                          #!

        # set the data inside the beam (2nd part of the initialization)
        self.beam.set_data(xgc_,self.limits)

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
        tck = sp.interpolate.interp1d(wl,trans,kind='linear',fill_value=0.0)
        return tck
        
        
    def compute_limits(self, eps=0, dxmin = 0.05, dymin = 0.05, dzmin = 0.05):
        r""" Compute the limits of the mesh that should be loaded

        The only limitations comes from the sampling volume and the lifetime of the excited state.
        In the figure below, blue is for the beam and the lifetime effect, red for the ring and the cutoff values (:keyword:`self.inter`),
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

        :todo: Improve this function in order to have a smaller window        
        """

        # average beam width
        # assume a configuration where the beam as a width close to a constant
        # and the focus point are close from each other
        width = [self.beam.stddev_h, self.beam.stddev_v]

        w = (width[0]*np.sum(self.op_direc[0,0:2]) + width[1]*self.op_direc[0,2])*self.inter
        w /= np.abs(np.dot(self.beam.direc,self.op_direc[0,:]))
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
            w_min[k] = self.get_width(pos_optical_min[k,2],k)
            w_max[k] = self.get_width(pos_optical_max[k,2],k)
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


        # check if use equilibrium data for the attenuation
        if not self.beam.eq:
            self.Xmax = max([self.Xmax,self.beam.pos[0]])
            self.Xmin = min([self.Xmin,self.beam.pos[0]])
            self.Ymax = max([self.Ymax,self.beam.pos[1]])
            self.Ymin = min([self.Ymin,self.beam.pos[1]])
            self.Zmax = max([self.Zmax,self.beam.pos[2]])
            self.Zmin = min([self.Zmin,self.beam.pos[2]])

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

        print 'The limits are: ',self.limits


    def get_bes(self):
        """ Compute the image from the synthetic diagnostics.
        This function should be the only one used outside the class.
        
        :returns: Photon radiance collected by each fiber (number of photons by seconds, by steradians and by square meters)
        :rtype: np.array[Ntime, Nfib]

        The following graph shows the most important call during the computation of the intensity collected by the fibers.
        The red arrows show the call order and the black ones show what is inside the function.
        When two red arrows comes from the same point, it means that a if condition is present.

        .. graphviz::

           digraph get_bes{
           compound=true;

           // list of variable that will be present at least twice
           A [label="BES.intensity"];
           B [label="BES.intensity"];

           Aint [label="Integration.integration_points"];
           Bint [label="Integration.integration_points"];
           Cint [label="Integration.integration_points"];
           Dint [label="Integration.integration_points"];

           Aquant [label="Beam1D.get_quantities"];
           Bquant [label="Beam1D.get_quantities"];

           Abeam [label="Beam1D.get_beam_density"];
           Bbeam [label="Beam1D.get_beam_density"];

           Aemis [label="Collisions.get_emission"];
           Bemis [label="Collisions.get_emission"];

           // BES.GET_BES
           subgraph cluster_get_bes { label="BES.get_bes"; "XGC_Loader_local.load_next_time_step"->"Beam1D.compute_beam_on_mesh"->
           "BES.intensity_para"[color="red"];
           "Beam1D.compute_beam_on_mesh"->A[color="red"];
           }

           "BES.intensity_para"->B[lhead=cluster_para];
           subgraph cluster_para { label="BES.intensity_para"; B;}

           // BES.INTENSITY
           A->Aint[lhead=cluster_intensity];
           B->Aint[lhead=cluster_intensity];
           subgraph cluster_intensity { label="BES.intensity"; Aint->"BES.light_from_plane"[color="red"]}

           // BES.LIGHT_FROM_PLANE
           "BES.light_from_plane"->Bint[lhead=cluster_light]
           subgraph cluster_light { label="BES.light_from_plane"; Bint->"BES.get_emis_from"[color="red"]}

           // BES.GET_EMIS_FROM
           "BES.get_emis_from"->"BES.to_cart_coord"[lhead=cluster_emis_from];
           subgraph cluster_emis_from { label="BES.get_emis_from"; "BES.to_cart_coord"->"Beam1D.get_emis_lifetime"
           ->"BES.get_solid_angle"->"BES.filter"[color="red"];
           "BES.to_cart_coord"->"Beam1D.get_emis"->"BES.get_solid_angle"[color="red"];
           }

           // BEAM1D.GET_EMIS_LIFETIME
           "Beam1D.get_emis_lifetime"->Cint[lhead=cluster_lifetime];
           subgraph cluster_lifetime { label="Beam1D.get_emis_lifetime"; Cint->Aquant->"Collisions.get_lifetime"->
           Abeam->Aemis[color="red"];}

           // BEAM1D.GET_EMIS
           "Beam1D.get_emis"->Bquant[lhead=cluster_emis];
           subgraph cluster_emis { label="Beam1D.get_emis"; Bquant->Bbeam->
           Bemis[color="red"]}


           // BEAM1D.GET_QUANTITIES
           Aquant->"XGC_Loader_local.interpolate_data" [lhead=cluster_quantities];
           Bquant->"XGC_Loader_local.interpolate_data" [lhead=cluster_quantities];
           subgraph cluster_quantities { label="Beam1D.get_quantities"; "XGC_Loader_local.interpolate_data"}

           // XGC_LOADER_local.INTERPOLATE_DATA
           "XGC_Loader_local.interpolate_data"-> "load_XGC_local.get_interp_planes_local"[lhead=cluster_interpolate];
           
           subgraph cluster_interpolate { label="XGC_Loader_local.interpolate_data"; 
           "load_XGC_local.get_interp_planes_local"->"XGC_Loader_local.find_interp_positions"[color="red"];
           }


           // BES.GET_SOLID_ANGLE
           "BES.get_solid_angle"->"BES.find_case"[lhead=cluster_solid_angle];
           subgraph cluster_solid_angle { label="BES.get_solid_angle";
           "BES.find_case"->"Funcs.solid_angle_disk"->"BES.solid_angle_mix_case"[color="red"]; 
           }
        
           "Funcs.solid_angle_disk"->"Funcs.heuman"[lhead=cluster_heuman];
           subgraph cluster_heuman { label="Funcs.solid_angle_disk"; "Funcs.heuman";}


           // BES.SOLID_ANGLE_MIX_CASE
           "BES.solid_angle_mix_case"->"BES.solid_angle_seg"[lhead=cluster_solid_angle_mix_case];
           subgraph cluster_solid_angle_mix_case { label="BES.solid_angle_mix_case";
           "BES.solid_angle_seg";}

           // BES.SOLID_ANGLE_SEG
           "BES.solid_angle_seg"->Dint[lhead=cluster_solid_angle_mix_case];
           subgraph cluster_seg { label="BES.solid_angle_seg"; Dint}



           }
        """
        if self.multiprocessing:
            p = mp.Pool(maxtasksperchild=5)
        print self.time
        nber_fiber = self.pos_foc.shape[0]

        # solid angle
        print 'Should be changed if the order of the methods change'
        self.solid = np.zeros((nber_fiber,self.Nint-1,2,21) )
        I = np.zeros((len(self.time),nber_fiber))
        for i,time in enumerate(self.time):
            print('Time step number: ' + str(i+1) + '/' + str(len(self.time)))
            # first time step already loaded
            if i != 0:
                self.beam.data.load_next_time_step()
                if not self.beam.eq:
                    self.beam.compute_beam_on_mesh()
            if self.multiprocessing:
                a = np.array(p.map(self.intensity_para, range(nber_fiber)))
                I[i,:] = a
            else:
                for j in range(nber_fiber):
                    # compute the light received by each fiber
                    I[i,j] = self.intensity(i,j)
        return I

    def get_psin(self,pt):
        """ Compute the :math:`\Psi_n`. 
        
        :math:`\Psi_n` is equal to 0 on the magnetic axis and to 1 on the separatrix.

        :param np.array[N,3] pt: Positions in the cartesian system
        :return: :math:`\Psi_n`
        :rtype: np.array[N]
        """
        R = np.sqrt(np.sum(self.pos_foc[:,0:2]**2,axis=1))
        return self.beam.data.psi_interp(self.pos_foc[:,2],R)/self.beam.data.psi_x
        
    def intensity_para(self,i):
        """ Same as :func:`intensity <FPSDP.Diagnostics.BES.bes.BES.intensity>`, but have only one argument.
        The only use is for the parallelization that ask only one argument.
        Use the variable :keyword:`self.beam.data.current` from the data loader.

        :param int i: Index of a fiber
        :returns: Photon radiance received by the fiber (number of photons by seconds, by steradians and by square meters)
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

    def get_width(self,z,fib):
        r""" Compute the radius of the light cone.

        The following figure shows two cases of the computation of the radius (:math:`R_1` and :math:`R_2`).

        .. tikz:: [line cap=round,line join=round,x=1.0cm,y=1.0cm]
           \clip(-2.95,-3.88) rectangle (12.16,4.13);
           \draw [domain=-2.95:12.16] plot(\x,{(-0-0*\x)/8});
           \draw [line width=1.5pt] (0,1.12)-- (0,0);
           \draw [line width=1.5pt] (2.6,3.14)-- (2.6,-3.14);
           \draw [line width=1.5pt] (0,0)-- (0,-1.12);
           \draw (0,-1.12)-- (2.6,-3.14);
           \draw (2.6,-3.14)-- (7.4,-1.08);
           \draw (2.6,3.14)-- (7.4,1.08);
           \draw (2.6,3.14)-- (0,1.12);
           \draw [line width=1.5pt] (7.4,1.08)-- (7.4,-1.08);
           \draw [dash pattern=on 3pt off 3pt] (2.6,-3.14)-- (7.4,1.08);
           \draw [dash pattern=on 3pt off 3pt] (2.6,3.14)-- (7.4,-1.08);
           \draw [domain=7.4:12.156934491978609] plot(\x,{(--30.13-4.88*\x)/5.55});
           \draw [domain=7.4:12.156934491978609] plot(\x,{(-28.55--4.63*\x)/5.26});
           \draw [color=blue] (5.3,1.98)-- (5.3,0);
           \draw [color=blue] (9.19,2.65)-- (9.19,0);
           \draw (-1.2,0.53) node[anchor=north west] {Fiber};
           \draw (1.4,1.6) node[anchor=north west] {Lens};
           \draw (7.3,-0.6) node[anchor=north west] {Focus point};
           \begin{scriptsize}
           \draw[color=blue] (5.51,1.21) node {$R_2$};
           \draw[color=blue] (9,1.41) node {$R_1$};
           \end{scriptsize}
        
        :param np.array[N] z: Position where to compute the width in the optical system
        :param int fib: Index of the fiber

        :returns: Radius of the optical cone
        :rtype: np.array[N]
        """
        if isinstance(z,float):
            z = np.atleast_1d(z)
        r = np.zeros(z.shape)
        # index before focus point
        ind = z<=self.dist[fib]
        # case before the focus point
        r[ind] = self.rad_lens + (self.rad_foc[fib]-self.rad_lens)*z[ind]/self.dist[fib]
        # case after it
        r[~ind] = -self.rad_lens + (self.rad_foc[fib] + self.rad_lens)*z[~ind]/self.dist[fib]
        return r

    def get_width_cases(self,z,fib):
        r""" Compute the radius of the limits between all the cases.

        The four different cases are shown in the following figure (:math:`R_i`).

        .. tikz:: [line cap=round,line join=round,x=0.75cm,y=0.75cm]
           \clip (-2,-8.14) rectangle (20,6.3);
           \draw [domain=-4.3:18] plot(\x,{(-0-0*\x)/12});
           \draw (0.34,1)-- (4.82,4.14);
           \draw (4.82,4.14)-- (10.62,1.76);
           \draw (10.62,-1.76)-- (4.82,-4.14);
           \draw (4.82,-4.14)-- (0.34,-1);
           \draw [line width=1.6pt] (0.34,-1)-- (0.34,1);
           \draw [line width=1.6pt] (4.82,4.14)-- (4.82,-4.14);
           \draw [line width=1.6pt] (10.62,-1.76)-- (10.62,1.76);
           \draw [dash pattern=on 5pt off 5pt] (4.82,4.14)-- (10.62,-1.76);
           \draw [domain=10.62:18] plot(\x,{(--53.52-6.02*\x)/5.92});
           \draw [domain=10.62:18] plot(\x,{(-31.28--3.52*\x)/3.46});
           \draw [dash pattern=on 5pt off 5pt] (10.62,1.76)-- (4.82,-4.14);
           \draw [color=blue] (6.38,2.56)-- (6.38,0);
           \draw [dash pattern=on 5pt off 5pt,domain=10.62:18] plot(\x,{(--26.24-1.76*\x)/4.29});
           \draw [dash pattern=on 5pt off 5pt,domain=10.62:18] plot(\x,{(-26.24--1.76*\x)/4.29});
           \draw [color=blue] (9.67,0.8)-- (9.67,0);
           \draw [color=blue] (12.26,1.09)-- (12.26,0);
           \draw [color=blue] (17.36,1)-- (17.36,0);
           \draw (-1.1,0.86) node[anchor=north west] {Fiber};
           \draw (3.5,1.62) node[anchor=north west] {Lens};
           \draw (10.5,-0.1) node[anchor=north west] {Focus point};
           \begin{scriptsize}
           \draw[color=blue] (6.1,1.42) node {$R_1$};
           \draw[color=blue] (10,0.54) node {$R_2$};
           \draw[color=blue] (12,0.7) node {$R_3$};
           \draw[color=blue] (17,0.66) node {$R_4$};
           \end{scriptsize}

        :param np.array[N] z: Position where to compute the width in the optical system
        :param int fib: Index of the fiber

        :returns: Radius of the optical cone
        :rtype: np.array[N]
        """
        if isinstance(z,float):
            z = np.atleast_1d(z)
        r = np.zeros(z.shape)
        # index before focus point
        ind = z<=self.dist[fib]
        # slope of the linear function
        a = (self.rad_lens+self.rad_foc[fib])/self.dist[fib]
        # case before the focus point
        r[ind] = np.abs(self.rad_lens - a*z[ind])
        # case after it
        a = (self.rad_lens-self.rad_foc[fib])/self.dist[fib]
        r[~ind] = np.abs(self.rad_lens - a*z[~ind])
        return r

    def find_case(self,r,z,fib):
        r""" Compute the case for the solid angle.

        In the following figure, the three different cases are shown.
        In red, the area of the ring case is shown, in blue, the mixed case and, in green, the lens case 
        (see :func:`get_solid_angle <FPSDP.Diagnostics.BES.bes.BES.get_solid_angle>` for more a drawing of the different cases).

        .. tikz:: [line cap=round,line join=round,x=0.75cm,y=0.75cm]
           \clip(-2,-5.71) rectangle (32,5.71);
           \fill[line width=0pt,color=blue,fill=blue,fill opacity=0.1] (4.82,4.14) -- (10.62,1.76) -- (8.89,0) -- cycle;
           \fill[line width=0pt,color=blue,fill=blue,fill opacity=0.1] (8.89,0) -- (4.82,-4.14) -- (10.62,-1.76) -- cycle;
           \fill[line width=0pt,color=blue,fill=blue,fill opacity=0.1] (20.98,12.3) -- (10.62,1.76) -- (14.91,0) -- (20.98,2.49) -- cycle;
           \fill[line width=0pt,color=blue,fill=blue,fill opacity=0.1] (14.91,0) -- (10.62,-1.76) -- (20.98,-12.3) -- (20.98,-2.49) -- cycle;
           \fill[line width=0pt,color=red,fill=red,fill opacity=0.1] (4.82,4.14) -- (8.89,0) -- (4.82,-4.14) -- cycle;
           \fill[line width=0pt,color=red,fill=red,fill opacity=0.1] (20.98,2.49) -- (14.91,0) -- (20.98,-2.49) -- cycle;
           \fill[line width=0pt,color=green,fill=green,fill opacity=0.1] (10.62,1.76) -- (8.89,0) -- (10.62,-1.76) -- (14.91,0) -- cycle;
           \draw [domain=-2.61:28.08] plot(\x,{(-0-0*\x)/12});
           \draw (0.34,1)-- (4.82,4.14);
           \draw (4.82,4.14)-- (10.62,1.76);
           \draw (10.62,-1.76)-- (4.82,-4.14);
           \draw (4.82,-4.14)-- (0.34,-1);
           \draw [line width=1.6pt] (0.34,-1)-- (0.34,1);
           \draw [line width=1.6pt] (4.82,4.14)-- (4.82,-4.14);
           \draw [line width=1.6pt] (10.62,-1.76)-- (10.62,1.76);
           \draw [dash pattern=on 5pt off 5pt] (4.82,4.14)-- (10.62,-1.76);
           \draw [domain=10.62:28.075140428510473] plot(\x,{(--53.52-6.02*\x)/5.92});
           \draw [domain=10.62:28.075140428510473] plot(\x,{(-31.28--3.52*\x)/3.46});
           \draw [dash pattern=on 5pt off 5pt] (10.62,1.76)-- (4.82,-4.14);
           \draw [dash pattern=on 5pt off 5pt,domain=10.62:28.075140428510473] plot(\x,{(--26.24-1.76*\x)/4.29});
           \draw [dash pattern=on 5pt off 5pt,domain=10.62:28.075140428510473] plot(\x,{(-26.24--1.76*\x)/4.29});
           \draw (-1.2,0.9) node[anchor=north west] {Fiber};
           \draw (3.3,1.64) node[anchor=north west] {Lens};
           \draw (10.5,-0.05) node[anchor=north west] {Focus point};
           \draw (20.98,-5.71) -- (20.98,10.58);           

        :param np.array[N] r: Distance between the central axis and the point
        :param np.array[N] z: Distance between the point and the lens
        :param int fib: Index of the fiber

        :returns: 0 if inside the first area, 1 if in the second one and 2 in the third
        :rtype: np.array[N] of int
        """
        # create an array with only the first case
        ret = np.zeros(r.shape[0], dtype=int)
        # compute the radius of the cone at each position
        r_cone = self.get_width_cases(z,fib)

        # if the radius is bigger than cone => case 2
        ret[r>r_cone] = 2
        # if before intersection => case 1
        ret[(z < self.lim_op[fib,0]) & (r<=r_cone)] = 1
        # if after the intersection 2 => case 1
        ret[(z > self.lim_op[fib,1]) & (r<=r_cone)] = 1

        return ret
        
    def light_from_plane(self,z, t_, fiber_nber,zind,comp_eps=False):
        r""" Compute the light from one plane using a method of order 10 (see report or
        Abramowitz and Stegun) or by making the assumption of a constant emission on the plane.
        
        .. math::
           I_\text{plane} = \frac{\iint_D f(x) \mathrm{d}\sigma}{\iint_D \Omega(x)\mathrm{d}\sigma}
           \approx \frac{\sum_i \omega_i f(x_i)}{\sum_i \omega_i \Omega(x_i)}
        
        where :math:`f(x) = F(\varepsilon(x))\Omega(x)`, :math:`\Omega(x)` is the solid angle, :math:`F(x)` is the filter,
        D is the disk representing the plane, and, :math:`\omega_i` and :math:`x_i` are the weights and the points
        of the quadrature formula.

        The filter is computed at this point in order to simplify the code.

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
        :param int zind: Index of the z integration

        :returns: Intensity collected by the fiber from these planes
        :rtype: np.array[N]
        """
        I = np.zeros(z.shape[0])
        if self.type_int == '2D':
            # compute the integral with a few points
            # outside the central line
            r = self.get_width(z,fiber_nber)
            for i,r_ in enumerate(r):
                # integration points
                quad = integ.integration_points(2, 'order10', 'disk', r_)
                pos = np.zeros((quad.pts.shape[0],3))
                pos[:,0] = quad.pts[:,0]
                pos[:,1] = quad.pts[:,1]
                pos[:,2] = z[i]*np.ones(quad.pts.shape[0])
                eps = self.get_emis_from(pos,t_,fiber_nber)
                
                # now compute the solid angle
                if comp_eps or (self.solid[fiber_nber,zind,i,:] == 0).any():
                    # if an error of size is thrown, look at the line that create the array
                    # the best explaination is that someone as change the order of a method
                    self.solid[fiber_nber,zind,i,:] = self.get_solid_angle(pos,fiber_nber)
                # compute the filter
                filt = self.get_filter(pos)
                
                # sum the intensity of all the components
                eps = np.sum(eps*filt,axis=0)

                # sum the emission of all the points with the appropriate
                # weight (quadrature formula and solid angle)
                wsol = quad.w*self.solid[fiber_nber,zind,i,:]
                I[i] = np.sum(wsol*eps)/np.sum(wsol)
        elif self.type_int == '1D':
            # just use the point on the central line
            for i,z_ in enumerate(z):
                pos = np.array([0,0,z_])
                filt = self.get_filter(pos)

                I[i] = np.sum(self.get_emis_from(pos[np.newaxis,:],t_,fiber_nber)*filt,axis=0)
        else:
            raise NameError('This type of integration does not exist')
        return I

    def get_filter(self,pos):
        """ Compute the wavelenght and the transmittance for each position.

        Use the Doppler effect for the computation of the wavelength:

        :param np.array[N,3] pos: Position in the optical system 

        :returns: Transmittance
        :rtype: np.array[Nbeam,N]
        
        """
        # compute the effect of the filter
        dist_ = self.pos_lens[np.newaxis,:] - pos
        dist_ = dist_/np.sqrt(np.sum(dist_**2,axis=-1)[:,np.newaxis])

        # compute the angle between the beam and the optic
        costh = np.einsum('ij,j->i',dist_,self.beam.direc)

        # compute the wavelength
        wl = (1.0 - self.beam.speed[:,np.newaxis]*costh/SI['c'])
        wl *= self.wl0
        # interpolate
        return self.filter_(wl)
        
        

    def intensity(self,t_,fiber_nber,comp_eps=False):
        r""" Compute the light received by a fiber at one time step.
        
        Use a Gauss-Legendre quadrature formula of order 4.
        
        .. math::
           I = \int_{-d}^d f(z) \mathrm{d}z \approx 
           \sum_i \frac{b_i-a_i}{2} \sum_j \omega_j f\left(\frac{b_i-a_i}{2}x_j + \frac{a_i+b_i}{2}\right)

        where the index i is for the splitting in subintervals, j is for the Gauss-Legendre formula,
        :math:`f(z)` is the function computed by :func:`light_from_plane <FPSDP.Diagnostics.BES.bes.BES.light_from_plane>`,
        :math:`d = \text{inter} \cdot w`, inter is the cutoff in unit of the average beam width (w),
        :math:`a_i` and :math:`b_i` are the lower and upper limits for each intervals (not linear spacing 
        [look at :func:`get_interval_gaussian <FPSDP.Maths.Integration.get_interval_gaussian>`]),
        :math:`\omega_j` and :math:`x_j` are the weights and points of the quadrature formula.
        See figure :func:`compute_limits <FPSDP.Diagnostics.BES.bes.BES.compute_limits>` for a view of the situation.

        The computation of the intervals assume that the focus point is exactly at the center of the beam and that :math:`f(x)` is a gaussian.
        

        :param int t_: Time step to compute
        :param int fiber_nber: Index of the fiber

        :returns: Photon radiance collected by the fiber (number of photons by seconds, by steradians and by square meters)
        :rtype: float
        """
        # first define the quadrature formula
        quad = integ.integration_points(1,'GL2') # Gauss-Legendre order 4
        I = 0.0
        # compute the distance from the origin of the beam
        dist = np.dot(self.pos_foc[fiber_nber,:] - self.beam.pos,self.beam.direc)
        width = self.beam.get_width(dist)
        # compute the average beam width of the beam
        width = (width[0]*np.sum(self.op_direc[fiber_nber,0:2]) + width[1]*self.op_direc[fiber_nber,2])*self.inter
        width /= np.abs(np.dot(self.beam.direc,self.op_direc[fiber_nber,:]))
        # limit of the intervals
        border = np.linspace(-width*self.inter,width*self.inter,self.Nint)
	#border = integ.get_interval_gaussian(width*self.inter,width,self.Nint)
        # value inside the intervals
        Z = 0.5*(border[:-1] + border[1:])
        # half size of one interval
        ba2 = 0.5*(border[1:]-border[:-1])
        for i,z in enumerate(Z):
            # distance of the plane from the lense
            pt = z + ba2[i]*quad.pts + self.dist[fiber_nber]
            light = self.light_from_plane(pt,t_,fiber_nber,i,comp_eps)
            # sum the weight with the appropriate pts
            I += np.sum(quad.w*light)*ba2[i]
        # multiply by the weigth of each interval
        return I
        
    def get_emis_from(self,pos,t_,fiber_nber):
        """ Compute the total emission of each position.

        :param np.array[N,3] pos: Position in the optical system 
        :param int t_: Time step to compute
        :param int fiber_nber: Index of the fiber

        :returns: Photon radiance emitted by each point
        :rtype: np.array[N]
        """
        # first change coordinate: optical -> cartesian (Tokamak)
        x = self.to_cart_coord(pos,fiber_nber)
        if self.lifetime:
            # choose if we use the lifetime effect or not
            eps = self.beam.get_emis_lifetime(x,t_)/(4.0*np.pi)
        else:
            eps = self.beam.get_emis(x,t_)/(4.0*np.pi)
        return eps

    
    def get_solid_angle(self,pos,fib):
        r""" Compute the solid angle 

        Three different cases can happen:

        * Lens case
        * Ring case
        * mixed case

        In the following drawing, the vision from a particle that emits a photon is shown.
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
           \node at (5,2.5) {Mixed Case};


        The two first are solved with the formula of Paxton (:func:`solid_angle_disk <FPSDP.Maths.Funcs.solid_angle_disk>`) and
        the last one is solved numerically.

        For finding in which case a point is, the function :func:`find_case <FPSDP.Diagnostics.BES.bes.BES.find_case>`])is used.

        For finding the intersections, the following system is solved [assuming that the coordinate system is the optical one]:
        
        .. math::
           \left\{ \begin{array}{ccc}
           x_1^2 + x_2^2 & = & r_r^2 \\
           y_1^2 + y_2^2 & = & r_l^2 \\
           \frac{{\bf y}-{\bf P}}{z} & = & \frac{{\bf y}-{\bf x}}{L}\\
           \end{array}\right.

        where :math:`x_i` (:math:`y_i`) are the coordinates of the intersection on the ring (lens),
        :math:`{\bf P}` is the point where we want to compute the solid angle, :math:`r_r` (:math:`r_l`) 
        is the radius of the ring (lens), :math:`z` is the last coordinate of :math:`{\bf P}` 
        (thus the distance to the lens) and :math:`L` is the one for :math:`{\bf x}`.

        :param np.array[N,3] pos: Position in the optical system
        :param int fib: Index of the fiber

        :returns: Solid angle
        :rtype: np.array[N]
        """
        r = np.sqrt(np.sum(pos[:,0:2]**2,axis=1))
        z = pos[:,2]
        # check for different case
        test = self.find_case(r,pos[:,2],fib)
        solid = np.zeros(pos.shape[0])

        #---------------------
        # first case (look in find_case for a drawing)
        ind = test==0
        solid[ind] = solid_angle_disk(pos[ind,:],self.rad_lens)

        #---------------------
        # second case
        ind = test==1
        d = pos[ind,:]
        # change the origin of the central axis
        d[:,2] = np.abs(d[:,2]-self.dist[fib])
        solid[ind] = solid_angle_disk(d,self.rad_foc[fib])

        #--------------------
        # last case
        # first computation of the intersection
        ind = test==2
        if (pos[ind,0] == 0).any():
            raise NameError('Should implement the other version (switch pos[ind,0] <-> pos[ind,1])')

        # few values defined in my report
        ratio = np.abs(z[ind]/self.dist[fib])
        f = 1.0/(1.0-ratio)
        A = 0.5*((r[ind]**2-(self.rad_lens/f)**2)/ratio + ratio*self.rad_foc[fib]**2)/pos[ind,0]
        B = -pos[ind,1]/pos[ind,0]

        # \Delta when computing the solution of a quadratic function
        delta = np.sqrt(4*B**2*A**2 - 4*(A**2-self.rad_foc[fib]**2)*(B**2 + 1))
        # first intersection point on the focus point [Npt,{x,y}]
        x1 = np.zeros((np.sum(ind),2))
        # second one
        x2 = np.zeros((np.sum(ind),2))
        x1[:,1] = (-2*B*A + delta)/(2*(B**2+1))
        x2[:,1] = (-2*B*A - delta)/(2*(B**2+1))
        x1[:,0] = A + B*x1[:,1]
        x2[:,0] = A + B*x2[:,1]

        # same but with the lens
        y1 = f[:,np.newaxis]*(pos[ind,:2]-ratio[:,np.newaxis]*x1)
        y2 = f[:,np.newaxis]*(pos[ind,:2]-ratio[:,np.newaxis]*x2)

        solid[ind] = self.solid_angle_mix_case(pos[ind,:],[x1,x2],[y1,y2],fib)

        #--------------------
        if (solid < 0).any() or (solid > 4*np.pi).any():
            print np.sqrt(np.sum(x1**2,axis=1))
            print np.sqrt(np.sum(y2**2,axis=1))
            print('solid angle',solid)
            print('find_case',test)
            print('ind',ind)
            print ('x',x1,x2)
            print('y',y1,y2)
            raise NameError('solid angle smaller than 0 or bigger than 4pi')
        return solid

    def solid_angle_mix_case(self,pos,x,y,fib):
        r""" Compute numerically the solid angle for the mixed case
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
        :param list[x1,x2] x: Position of the intersection on the ring (x1 and x2 are np.array[N]) 
        :param list[y1,y2] y: Position of the intersection on the lens (y1 and y2 are np.array[N])
        :param int fib: Index of the fiber
        :return: Solid angle
        :rtype: np.array[N]
        """
        # first the contribution of the ring
        omega1 = solid_angle_seg(pos-np.array([0,0,self.dist[fib]]),x,
                                self.rad_foc[fib],0,self.Nsolid,
                                self.Nr)

        # second the contribution of the lens
        omega2 = solid_angle_seg(pos,y,self.rad_lens,1,self.Nsolid,
                               self.Nr)

        # remove the part where the numerical error is too big
        ind1 = omega1 < 0
        ind2 = omega2 < 0
        omega = np.zeros(omega1.shape)
        omega[~ind1] += omega1[~ind1]
        omega[~ind2] += omega2[~ind2]
        
        if (-omega1[ind1]/omega[ind1] > 1e-3).any():
            print omega
            print omega1
            raise NameError('solid angle negative 1')

        if (-omega2[ind2]/omega[ind2] > 1e-3).any():
            print omega
            print omega2
            raise NameError('solid angle negative 2')

        return omega












class BES_ideal:
    """ Take the output of the simulation and just 
    compute the density fluctuation at the focus points.
    
    A lot of copy and paste from the BES class, therefore look there for
    the comments

    :param str input_file: name of the BES config file
    :param bool mesh: Use a mesh done from min/max of the focus points (True)\
    or the focus points (False)
    """
    def __init__(self,input_file,mesh=False):
        """ load all the data from the input file.

        mesh is used for knowing
        if the focus points are used or if a mesh is created one the max/min
        value of the focus points
        """
        self.cfg_file = input_file                                           #!
        if not exists(self.cfg_file):
            raise NameError('Config file not found')
        config = psr.ConfigParser()
        config.read(self.cfg_file)
        self.mesh = mesh

        self.data_path = config.get('Data','data_path')                      #!
        start = json.loads(config.get('Data','timestart'))
    
        R = json.loads(config.get('Optics','R'))
        R = np.array(R)
        phi = json.loads(config.get('Optics','phi'))
        Z = json.loads(config.get('Optics','Z'))
        plane = json.loads(config.get('Optics','plane'))
        # compute the value of phi in radian
        print 'should be changed if use another code than xgc'
        name = self.data_path + 'xgc.3d.' + str(start).zfill(5)+'.h5'
        nber_plane = h5.File(name,'r')
        nphi = nber_plane['nphi'][:]
        shift = np.mean(phi) - 2*np.pi*plane/nphi[0]

        
        self.pos_foc = np.zeros((len(Z),3))                                  #!
        self.pos_foc[:,0] = R*np.cos(phi)
        self.pos_foc[:,1] = R*np.sin(phi)
        self.pos_foc[:,2] = Z
        
        # Data part
        self.dphi = json.loads(config.get('Data','dphi'))                    #!
        end = json.loads(config.get('Data','timeend'))
        timestep = json.loads(config.get('Data','timestep'))
        order = config.get('Data','interpolation')
        self.compute_limits()      # compute the limits of the mesh
        xgc_ = xgc.XGC_Loader_local(self.data_path, start, end, timestep,
                                    self.limits, self.dphi,shift,order)
        self.time = xgc_.time_steps                                          #!


        self.data = xgc_
        

    def compute_limits(self, eps=1, dxmin = 0.1, dymin = 0.1, dzmin = 0.5):
        """ find min/max coordinates of the focus points """
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
        """ Compute the image of the density turbulence at the focus points.

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

    def get_psin(self,pt):
        """ Compute the psin value. 
        
        Psin is equal to 0 on the magnetic axis and to 1 on the separatrix.
        :param np.array[N,3] pt: Positions in the cartesian system
        :return: Psin
        :rtype: np.array[N]
        """
        R = np.sqrt(np.sum(self.pos_foc[:,0:2]**2,axis=1))
        return self.data.psi_interp(self.pos_foc[:,2],R)/self.data.psi_x

    def intensity(self,t_,fiber_nber):
        """ Compute the light received by the fiber
        """
        I = self.data.interpolate_data(self.pos_foc[fiber_nber,:],t_,['ne'],False)[0]
        return I
 
