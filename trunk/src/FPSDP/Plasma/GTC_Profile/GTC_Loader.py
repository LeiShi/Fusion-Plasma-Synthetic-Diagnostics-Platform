# -*- coding: utf-8 -*-
"""
GTC loading module

Read in equilibrium and fluctuation data from GTC output files, interpolate and 
extrapolate necessary data onto a Cartesian grid specified by user. 
Data structure is compatible with Diagnostic modules.



Created on Mon Sep 21 14:07:39 2015

@author: lshi
"""
import numpy as np
import json
import re
import os
from ...Geometry.Grid import Cartesian2D, Cartesian3D
from ...IO import f90nml
from ...Maths.Funcs import poly3_curve
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator, interp1d
from matplotlib.tri import triangulation, cubic_interp

def get_interp_planes(loader):
    """Get the plane numbers used for interpolation for each point 
    """
    dPHI = 2 * np.pi / loader.n_plane
    phi_planes = np.arange(loader.n_plane)*dPHI
    if(loader.CO_DIR):
        nextplane = np.searchsorted(phi_planes,loader.grid.phi3D,side = 'right')
        prevplane = nextplane - 1
        nextplane[np.nonzero(nextplane == loader.n_plane)] = 0
    else:
        prevplane = np.searchsorted(phi_planes,loader.grid.phi3D,side = 'right')
        nextplane = prevplane - 1
        prevplane[np.nonzero(prevplane == loader.n_plane)] = 0

    return (prevplane,nextplane)

def calculate_needed_planes(my_gtc):
    pass

def check_time_availability(path,required_times,fname_pattern):
    """
    We use regular expression to extract the time step information from file names. Note that if non-default file names are used, we need to modify the regular expression pattern.
    More information about regular expression module in python, check out the documentation :https://docs.python.org/2/library/re.html 
    
    :param string path: the path within which file names are being checked
    :param required_times: time steps that are expected to be existing
    :type required_times: python or numpy array of int.
    :param fname_pattern: regular expression pattern that fits the filenames, and extract the time information with the group name *time*.        
    :type fname_pattern: raw string
    """
    
    fname_re = re.compile(fname_pattern)
    fnames = os.listdir(path)
    time_all = []
    for name in fnames:
        m = fname_re.match(name)
        if m:
            time_all.append(int(m.group('time')))
        
    time_all = np.sort(time_all)
    for t in required_times:
        if t not in time_all:
            raise GTC_Loader_Error(('Time {} not available!'.format(t),time_all))
    print 'All time available.'
    return time_all
    

class GTC_Loader_Error(Exception):
    def __init__(self,message):
        self.message = message
    def __str__(self):
        return str(self.message)

class GTC_Loader:
    """Main class for loading GTC data. 
    Instances are initialized with a given "grid" containing user specified mesh,
    and a path containing all GTC output files. After initialization, GTC data
    on the mesh will be ready to use.
    """
    
    def __init__(self,gtc_path,grid,tsteps,fname_pattern_2D = r'snap(?P<time>\d+)_fpsdp.json', fname_pattern_3D = r'PHI_(?P<time>\d+)_\d+.ncd', Mode = 'Full'):
        """Initialize a GTC loader with the following parameters:
        
            :param string gtc_path: The path where GTC output files are located. 
            :param grid: User defined spatial grid. All GTC data will be interpolated onto this grid.
            :type grid: FPSDP.Geometry.Grid object. Only Cartesian2D and Cartesian3D are supported for now.
            :param tsteps: time steps requested. Use GTC simulation step counting. Will be checked at the beginning to make sure all requested time steps have been outputted. 
            :type tsteps: list or 1D numpy array of int
            :param fname_pattern_2D: (Optional) Regular Expression to fit 2D output files. Do not change unless you have changed the output file name convention.
            :type fname_pattern_2D: raw string
            :param fname_pattern_3D: (Optional) Regular Expression to fit 3D output files. Do not change unless you have changed the output file name convention.
            :type fname_pattern_3D: raw string
            :param string Mode:(Optional) choice among 3 possible iniialization modes: 
                **full**(Default): load both equilibrium and fluctuations, and interpolate them on given *grid*
                **eq_only**: load only equilibrium, and interpolate it on given *grid*
                **least**: DO NOT load any GTC output files, only initialize the loader with initial parameters. This mode is mainly used for debug.
        """        
        
        
        self.path = gtc_path
        self.grid = grid
        self.tsteps = tsteps
        
        if isinstance(grid, Cartesian2D):
            print('2D grid detected.')
            self.dimension = 2
            #First, use :py:func:`check_time_availability` to analyze the existing files and check if all required time steps are available.
            #If any requested time steps are not there, raise an error and print out all existing time steps.
            try:
                self.time_all = check_time_availability(self.path,self.tsteps,fname_pattern_2D)
            except GTC_Loader_Error as e:
                print e.message[0]
                print 'Available time steps are:'
                print str(e.message[1])
                raise 
        elif isinstance(grid, Cartesian3D):
            print('3D grid detected.')
            self.dimension = 3
            try:
                self.time_all = check_time_availability(os.path.join(self.path,'phi_dir'),self.tsteps,fname_pattern_3D)
            except GTC_Loader_Error as e:
                print e.message[0]
                print 'Available time steps are:'
                print str(e.message[1])
                raise 
        else:
            raise(GTC_Loader_Error('Invalid grid: Only Cartesian2D or Cartesian3D grids are supported.'))
            
        
        if ((Mode == 'full') or (Mode == 'eq_only')):
            #read gtc.in.out and gtc.out, obtain run specifications like: adiabatic/non-adiabatic electrons, electrostatic/electromagnetic, time step, ion gyro-radius, and snap output frequency.
            self.load_gtc_specifics() 
        
            # read grid_fpsdp.json, create triangulated mesh on GTC grids and extended grids for equilibrium.
            self.load_grid() 
        
            #read equilibriumB_fpsdp.json and equilibrium1D_fpsdp.json, create interpolators for B_phi,B_R,B_Z on extended grids, and interpolators for equilibrium density and temperature over 'a'.
            self.load_equilibrium() 
        
            #interpolate equilibrium quantities
            self.interpolate_eq()
            
            if(Mode == 'full'):
                #For fluctuations, 2D and 3D loaders are different
                if(self.dimension == 2):
                    #2D is simple, read snap{time}_fpsdp.json and interpolate the data onto 2D grid
                    self.load_fluctuations_2D()
                    self.interpolate_fluc_2D()
                    
                if(self.dimension == 3):
                    #3D is much harder to do. First, we calculate the needed cross-section numbers
                    # get interpolation plane numbers for each grid point
                    self.prevplane,self.nextplane = get_interp_planes(self)
                    # calculate needed planes
                    planes = calculate_needed_planes(self)
                    # load fluctuations on those cross-sections            
                    self.load_fluctuations_3D(planes)
        
                    # interpolate onto our 3D mesh
                    self.interpolate_fluc_3D()
            
    def load_gtc_specifics(self):
        """ read relevant GTC simulation settings from gtc.in.out and gtc.out files.
        Create Attributes:
            :var bool HaveElectron: if True, GTC is simulating non-adiabatic electrons. Otherwise only adiabatic electrons are used.
            :var bool isEM: if True, GTC is in Electro-magnetic mode. Otherwise in electro-static mode.
            :var double dt: GTC simulation time step length, unit: second
            :var double rho_i: typical ion gyro-radius in GTC simulation, unit:meter
            :var int snapstep: GTC snapshot output time, every *snapstep* timesteps, a *snap###_fpsdp.json* file is written.
        """
        
        GTCin_fname = self.path+'gtc.in.out'
        gtcin_nml = f90nml.read(GTCin_fname)        
        
        self.HaveElectron = (gtcin_nml['input_parameters']['nhybrid'] == 1)
        self.isEM = (gtcin_nml['input_parameters']['magnetic'] == 1)
        self.snapstep = gtcin_nml['input_parameters']['msnap']
        
        
        #NEED INFO on gtc.in.out and gtc.out
        
    def load_grid(self):
        """ Read in Grid information from grid_fpsdp.json file.
        Create Attributes:
            :var R_gtc: R coordinates for GTC simulation grid. unit: meter
            :vartype R_gtc: 1D double array of length N
            :var Z_gtc: R coordinates for GTC simulation grid. unit: meter
            :vartype Z_gtc: 1D double array of length N
            :var points_gtc: (R,Z) pairs for each GTC simulation grid point
            :vartype points_gtc: (N,2) shape double array
            
            :var a_gtc: corresponding normalized flux radial coordinates on GTC grid
            :vartype a_gtc: 1D double array of length N
            :var theta_gtc: corresponding flux poloidal coordinates on GTC grid
            :vartype theta_gtc: 1D double array of length N

            :var R_eq: R cooridnate on larger mesh for interpolating equilibrium
            :vartype R_eq: 1D double array of length N_eq
            :var Z_eq: Z cooridnate on larger mesh for interpolating equilibrium
            :vartype Z_eq: 1D double array of length N_eq
            :var points_eq: (R,Z) pairs for each grid point on equilibrium mesh
            :vartype points_eq: (N_eq,2) shape double array

            :var a_eq: corresponding *a* values 
            :vartype a_eq: 1D double array of length N_eq
            
            :var R_LCFS: R coordinates for sampled last closed flux surface, i.e. a=1
            :vartype R_LCFS: 1D double array of length N_LCFS
            :var Z_LCFS: Z coordinates for sampled last closed flux surface, i.e. a=1
            :vartype Z_LCFS: 1D double array of length N_LCFS
            
            :var double R0: R coordinate of magnetic axis
            :var double Z0: Z coordinate of magnetic axis
            
            :var Delaunay_gtc: trangulation of GTC grid on (R,Z) plane, created by scipy.spatial.Delaunay
            :vartype Delaunay_gtc: scipy.spatial.Delaunay object
            :var Delaunay_eq: trangulation of equilibrium grid on (R,Z) plane, created by scipy.spatial.Delaunay
            :vartype Delaunay_eq: scipy.spatial.Delaunay object
            
            :var a_eq_interp: 2D linear interpolator of *a* on (R,Z) plane, using *a_eq* and *Delaunay_eq*. Out_of_bound value is set to be *nan* and will be dealt with later.
            :vartype a_eq_interp: scipy.interpolate.LinearNDInterpolator object
            
            
        """
        grid_fname = self.path+'grid_fpsdp.json'
        with open(grid_fname) as gridfile:
            raw_grids = json.load(gridfile)
            
        self.R_gtc = np.array(raw_grids['R_gtc'])
        self.Z_gtc = np.array(raw_grids['Z_gtc'])
        self.points_gtc = np.transpose(np.array([self.Z_gtc,self.R_gtc]))
        self.a_gtc = np.array(raw_grids['a_gtc'])
        self.theta_gtc = np.array(raw_grids['theta_gtc'])
        self.R_eq = np.array(raw_grids['R_eq'])
        self.Z_eq = np.array(raw_grids['Z_eq'])
        self.points_eq = np.transpose(np.array([self.Z_eq,self.R_eq]))
        self.a_eq = np.array(raw_grids['a_eq'])
        self.R_LCFS = np.array(raw_grids['R_LCFS'])
        self.Z_LCFS = np.array(raw_grids['Z_LCFS'])
        self.R0= raw_grids['R0']
        self.Z0 = raw_grids['Z0']
        
        #use Delaunay Triangulation package provided by **scipy** to do the 2D triangulation on GTC mesh
        self.Delaunay_gtc = Delaunay(self.points_gtc)
        
        #For equilibrium mesh, we use Triangulation package provided by **matplotlib** to do a cubic interpolation on *a* values and B field. The points outside the convex hull of given set of points will be treated later.
        self.triangulation_eq = triangulation.Triangulation(self.Z_eq,self.R_eq)
        
        #interpolate flux surface coordinate "a" onto grid_eq, save the interpolater for future use.
        #Default fill value is "nan", points outside eq mesh will be dealt with care
        self.a_eq_interp = cubic_interp(self.triangulation_eq,self.a_eq)
        
    def load_equilibrium(self, SOL_width = 0.1, extrapolation_points = 20):
        """ read in equilibrium field data from equilibriumB_fpsdp.json and equilibrium1D_fpsdp.json
        
        :param double SOL_width: *(optional)*, width of the scrape-off layer outside the closed flux surface region, used to determine the equilibrium decay length. Default to be 0.1. Unit: minor radius *a*=1
        :param int extrapolation_points: *(optional)* number of sample points added within SOL region, used for extrapolation of equilibrium density and temperature. Default to be 20.        
        
        Create Attributes:
            :var B_phi: :math:`\\Phi` component of equilibrium magnetic field on eq grid
            :vartype B_phi: 1D double array of length N_eq            
            :var B_R: *R* component of equilibrium magnetic field on eq grid
            :vartype B_R: 1D double array of length N_eq 
            :var B_Z: *Z* component of equilibrium magnetic field
            :vartype B_Z: 1D double array of length N_eq 
            
            :var B_phi_interp: interpolated B_phi. Out_of_bound value is set to be *nan* and will be dealt with later.
            :vartype B_phi_interp: scipy.interpolate.LinearNDInterpolator object
            :var B_R_interp: interpolated B_R. Out_of_bound value is set to be *nan* and will be dealt with later.
            :vartype B_R_interp: scipy.interpolate.LinearNDInterpolator object
            :var B_Z_interp: interpolated B_Z. Out_of_bound value is set to be *nan* and will be dealt with later.
            :vartype B_Z_interp: scipy.interpolate.LinearNDInterpolator object
            
            :var a_1D: *a* coordinate for 1D interpolation, includes *a* array read from *equilibrium1D_fpsdp.json* file, and extension on larger *a* values. 
            :vartype a_1D: 1D double array of length *N_1D*
            :var ne0_1D: equilibrium electron density on *a_1D*, read in from *equilibrium1D_fpsdp.json* file, and use 3rd order polynomial to extrapolate outside values. Unit: :math:`m^{-3}`
            :vartype ne0_1D: 1D double array of length *N_1D*
            :var Te0_1D: equilbirium electron temperature on *a_1D*, read in from *equilibrium1D_fpsdp.json* file, and use 3rd order polynomial to extrapolate outside values. Unit: keV
            :vartype Te0_1D: 1D double array of length *N_1D*
            
            :var ne0_interp:linear interpolator created with *ne0_1D* on *a_1D*. Out_of_bound value set to 0.
            :vartype ne0_interp: scipy.interpolate.interp1d object
            :var Te0_interp:linear interpolator created with *Te0_1D* on *a_1D*. Out_of_bound value set to 0.
            :vartype ne0_interp: scipy.interpolate.interp1d object
        """
        
        eqB_fname = self.path+ 'equilibriumB_fpsdp.json'
        with open(eqB_fname,'r') as eqBfile:
            raw_eqB = json.load(eqBfile)
            
        self.B_phi = np.array(raw_eqB['B_phi'])
        self.B_R = np.array(raw_eqB['B_R'])
        self.B_Z = np.array(raw_eqB['B_Z'])
        
        self.B_phi_interp = cubic_interp(self.triangulation_eq,self.B_phi) #outside points will be masked and dealt with later.
        self.B_R_interp = cubic_interp(self.triangulation_eq,self.B_R)
        self.B_Z_interp = cubic_interp(self.triangulation_eq,self.B_Z)

        #Now reading in 1D equilibrium quantities        
        eq1D_fname = self.path+'equilibrium1D_fpsdp.json'
        with open(eq1D_fname,'r') as eq1Dfile:
            raw_eq1D = json.load(eq1Dfile)
        
        # a, ne0, Te0 inside LCFS, sorted in *a* increasing order
        raw_a = np.array(raw_eq1D['a'])
        sorting = np.argsort(raw_a)
        sorted_a = raw_a[sorting]
        raw_ne0 = np.array(raw_eq1D['ne0'])
        sorted_ne0 = raw_ne0[sorting]
        raw_Te0 = np.array(raw_eq1D['Te0'])
        sorted_Te0 = raw_Te0[sorting]
        
        
        # outside needs extrapolation
        a_out = np.linspace(1,1+SOL_width,extrapolation_points)      
        # ne0 and Te0 will be extrapolated using 3rd order polynomial curves that fits the value and derivative at a=1, and decays to 0 at SOL outter edge.        
        # first, calculate the derivative using simple finite difference scheme
        dne0 = (sorted_ne0[-1]-sorted_ne0[-2])/(sorted_a[-1]-sorted_a[-2])
        dTe0 = (sorted_Te0[-1]-sorted_Te0[-2])/(sorted_a[-1]-sorted_a[-2])
        #set up the polynomial curve
        ne_curve = poly3_curve(sorted_a[-1],sorted_ne0[-1],a_out[-1],0,dne0,0)
        Te_curve = poly3_curve(sorted_a[-1],sorted_Te0[-1],a_out[-1],0,dTe0,0)
        #evaluate extrapolated curve at sample points
        ne0_out = ne_curve(a_out)
        Te0_out = Te_curve(a_out)        
        #append outside values to original sorted array
        self.a_1D = np.append(sorted_a,a_out)
        self.ne0_1D = np.append(sorted_ne0,ne0_out)
        self.Te0_1D = np.append(sorted_Te0,Te0_out)
        #set up interpolators using extrapolated samples, points outside the extended *a* range can be safely set to 0.
        self.ne0_interp = interp1d(self.a_1D,self.ne0_1D, bounds_error = False, fill_value = 0)
        self.Te0_interp = interp1d(self.a_1D,self.Te0_1D, bounds_error = False, fill_value = 0)
        
    def interpolate_eq(self):
        """Interpolate equilibrium quantities on given grid. 
        *B_R*, *B_Z*, *B_phi* and *a* are interpolated over (Z_eq,R_eq) mesh, and *ne0*, *Te0* are interpolated on *a* space.
        For interpolation over (Z_eq,R_eq),Grid points outside Equilibrium mesh(i.e. outside LCFS) will be approximated using the following method:
            For an outside point :math:`(Z_{out},R_{out})`, we search for the closest vertex on the convex hull of the interpolation set, :math:`(Z_n,R_n)`, and the corresponding :math:`a=a_n`. 
            From the cubic interpolation, we can obtain the derivatives of *a* respect to Z and R at :math:`(Z_n,R_n)`, :math:`\partial a/\partial Z` and :math:`\partial a/\partial R`.
            Now the *a* value at :math:`(Z_{out}, R_{out})` will be approximated by:
                ..math::
        
                    a(Z_{out},R_{out}) = a_n + (Z_{out}-Z_n) \cdot \frac{\partial a}{\partial Z} + (R_{out}-R_n) \cdot \frac{\partial a}{\partial R}
            
            This first order approximation is good if :math:`(Z_{out},R_{out})` is not far from :math:`(Z_n,R_n)`. In our case, since we are assuming :math:`n_e` and :math:`T_e` are rapidly decaying in *a* outside the LCFS, this approximation is good enough.
        """
        
        #outside points are obtained by examining the mask flag from the returned masked array of "cubic_interp"
        Zwant = self.grid.Z2D
        Rwant = self.grid.R2D        
        self.a_on_grid = self.a_eq_interp(Zwant,Rwant)
        out_mask = np.copy(self.a_on_grid.mask)
        
        Zout = Zwant[out_mask]
        Rout = Rwant[out_mask]
        
        #boundary points are obtained by applying ConvexHull on equilibrium grid points
        hull = ConvexHull(self.points_eq)
        p_boundary = self.points_eq[hull.vertices]
        Z_boundary = p_boundary[:,0]
        R_boundary = p_boundary[:,1]
        
        #Now let's calculate *a* on outside points, first, get the nearest boundary point for each outside point
        nearest_indices = []
        for i in range(len(Zout)):
            Z = Zout[i]
            R = Rout[i]
            nearest_indices.append (np.argmin((Z-Z_boundary)**2 + (R-R_boundary)**2) )
            
        # Then, calculate *a* based on the gradient at these nearest points
        Zn = Z_boundary[nearest_indices]
        Rn = R_boundary[nearest_indices]
        #The value *a* and its gradiant at this nearest point can by easily obtained            
        an = self.a_eq_interp(Zn,Rn)            
        gradaZ,gradaR = self.a_eq_interp.gradient(Zn,Rn)
        
        a_out = an + (Zout-Zn)*gradaZ + (Rout-Rn)*gradaR
        
        # Finally, assign these outside values to the original array
        self.a_on_grid[out_mask] = a_out
        
        #Now we are ready to interpolate ne and Te on our grid
        self.ne0_on_grid = self.ne0_interp(self.a_on_grid)
        self.Te0_on_grid = self.Te0_interp(self.a_on_grid)
        
        #B_R,B_Z and B_phi can be interpolated exactly like *a*
        self.BR_on_grid = self.B_R_interp(Zwant,Rwant)
        self.BZ_on_grid = self.B_Z_interp(Zwant,Rwant)
        self.Bphi_on_grid = self.B_phi_interp(Zwant,Rwant)
        
        BRn = self.B_R_interp(Zn,Rn)
        gradBR_Z, gradBR_R = self.B_R_interp.gradient(Zn,Rn)
        BR_out = BRn + (Zout-Zn)*gradBR_Z + (Rout-Rn)*gradBR_R
        self.BR_on_grid[out_mask] = BR_out
        
        BZn = self.B_Z_interp(Zn,Rn)
        gradBZ_Z, gradBZ_R = self.B_Z_interp.gradient(Zn,Rn)
        BZ_out = BZn + (Zout-Zn)*gradBZ_Z + (Rout-Rn)*gradBZ_R
        self.BZ_on_grid[out_mask] = BZ_out
        
        Bphin = self.B_phi_interp(Zn,Rn)
        gradBphi_Z, gradBphi_R = self.B_phi_interp.gradient(Zn,Rn)
        Bphi_out = Bphin + (Zout-Zn)*gradBphi_Z + (Rout-Rn)*gradBphi_R
        self.Bphi_on_grid[out_mask] = Bphi_out
        
    def load_fluctuations_2D(self):
        """ Read fluctuation data from **snap{time}_fpsdp.json** files
        Read data into an array with shape (NT,Ngrid_gtc), NT the number of requested timesteps, corresponds to *self.tstep*, Ngrid_gtc is the GTC grid number on each cross-section.
        Create Attribute:
            :var phi: fluctuating electro-static potential on GTC grid for each requested time step, unit: V
            :vartype phi: double array with shape (NT,Ngrid_gtc)
            :var Te: electron temperature fluctuation, unit: keV
            :vartype Te: double array with shape (NT,Ngrid_gtc)
            if *HaveElectron*:
            :var nane: non-adiabatic electron density fluctuation, unit: :math:`m^{-3}`
            :vartype nane: double array with shape (NT,Ngrid_gtc)                
            if *isEM*:
            :var A_par: parallel vector potential fluctuation
            :vartype A_par:double array with shape (NT,Ngrid_gtc)
            
        """
        NT = len(self.tsteps)
        Ngrid_gtc = len(self.R_gtc)
        self.phi = np.empty((NT,Ngrid_gtc))
        self.Te = np.empty_like(self.phi)        
        if self.HaveElectron:
            self.nane = np.empty_like(self.phi)
        if self.isEM:
            self.Apar = np.empty_like(self.phi)
        
        for i in range(NT):
            snap_fname = os.path.join(self.path,'snap{0:0>7}_fpsdp.json'.format(self.tsteps[i]))
            with open(snap_fname,'r') as snap_file:
                raw_snap = json.load(snap_file)
            
            self.phi[i] = np.array(raw_snap['phi'])
            self.Te[i] = np.array(raw_snap['dTe'])
            if self.HaveElectron:
                self.nane[i] = np.array(raw_snap['dne'])
            if self.isEM:
                self.Apar[i] = np.array(raw_snap['dAp'])
                
        
                
        
    def interpolate_fluc_2D(self):
        """Interpolate 2D fluctuations onto given Cartesian grid. Grids outside the GTC simulation domain will be given 0.
        
        """
        NT = len(self.tsteps)
        NZ = self.grid.NZ
        NR = self.grid.NR
        
        points_on_grid =np.transpose( np.array([self.grid.Z2D,self.grid.R2D]), (1,2,0))        
        
        self.phi_on_grid = np.empty((NT,NZ,NR))
        self.Te_on_grid = np.empty_like(self.phi_on_grid)
        if self.HaveElectron:
            self.nane_on_grid = np.empty_like(self.phi_on_grid)
        if self.isEM:
            self.Apar_on_grid = np.empty_like(self.phi_on_grid)
        for i in range(NT):
            phi_interp = LinearNDInterpolator(self.Delaunay_gtc,self.phi[i],fill_value = 0)
            self.phi_on_grid[i] = phi_interp(points_on_grid)
            Te_interp = LinearNDInterpolator(self.Delaunay_gtc,self.Te[i],fill_value = 0)
            self.Te_on_grid[i] = Te_interp(points_on_grid)
            if self.HaveElectron:
                nane_interp = LinearNDInterpolator(self.Delaunay_gtc,self.nane[i],fill_value = 0)
                self.nane_on_grid[i] = nane_interp(points_on_grid)
            if self.isEM:   
                Apar_interp = LinearNDInterpolator(self.Delaunay_gtc,self.Apar[i],fill_value = 0)
                self.Apar_on_grid[i] = Apar_interp(points_on_grid)
            

        
            
        
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
