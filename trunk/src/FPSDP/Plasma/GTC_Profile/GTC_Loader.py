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
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, interp1d

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

class GTC_Loader_Error(Exception):
    def __init__(self,message):
        self.message = message
    def __str__(self):
        return self.message

class GTC_Loader:
    """Main class for loading GTC data. 
    Instances are initialized with a given "grid" containing user specified mesh,
    and a path containing all GTC output files. After initialization, GTC data
    on the mesh will be ready to use.
    """
    
    def __init__(self,gtc_path,grid,tstart,tend,tstep):
        if isinstance(grid, Cartesian2D):
            print('2D grid detected.')
            self.dimension = 2
        elif isinstance(grid, Cartesian3D):
            print('3D grid detected.')
            self.dimension = 3
        else:
            raise(GTC_Loader_Error('Invalid grid: Only Cartesian2D or Cartesian3D grids are supported.'))
            
        self.path = gtc_path
        self.grid = grid
        
        #read gtc.in.out and gtc.out, obtain run specifications like: adiabatic/non-adiabatic electrons, electrostatic/electromagnetic, time step, ion gyro-radius, and snap output frequency.
        self.load_gtc_specifics() 
        
        # read grid_fpsdp.json, create triangulated mesh on GTC grids and extended grids for equilibrium.
        self.load_grid() 
        
        #read equilibriumB_fpsdp.json and equilibrium1D_fpsdp.json, create interpolators for B_phi,B_R,B_Z on extended grids, and interpolators for equilibrium density and temperature over 'a'.
        self.load_equilibrium() 
        
        #interpolate equilibrium quantities
        self.interpolate_eq()
        
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
        self.points_gtc = np.transpose(np.array([self.R_gtc,self.Z_gtc]))
        self.a_gtc = np.array(raw_grids['a_gtc'])
        self.theta_gtc = np.array(raw_grids['theta_gtc'])
        self.R_eq = np.array(raw_grids['R_eq'])
        self.Z_eq = np.array(raw_grids['Z_eq'])
        self.points_eq = np.transpose(np.array([self.R_eq,self.Z_eq]))
        self.a_eq = np.array(raw_grids['a_eq'])
        self.R_LCFS = np.array(raw_grids['R_LCFS'])
        self.Z_LCFS = np.array(raw_grids['Z_LCFS'])
        self.R0= raw_grids['R0']
        self.Z0 = raw_grids['Z0']
        
        #use Delaunay Triangulation package to do the 2D triangulation
        self.Delaunay_gtc = Delaunay(self.points_gtc)
        self.Delaunay_eq = Delaunay(self.points_eq)
        
        #interpolate flux surface coordinate "a" onto grid_eq, save the interpolater for future use.
        #Default fill value is "nan", points outside eq mesh will be dealt with care
        self.a_eq_interp = LinearNDInterpolator(self.Delaunay_eq,self.a_eq)
        
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
        
        self.B_phi_interp = LinearNDInterpolator(self.Delaunay_eq,self.B_phi) #fill value is default to *nan*, this will be used as a flag of out_of_bound points, and will be dealt with later.
        self.B_R_interp = LinearNDInterpolator(self.Delaunay_eq,self.B_R)
        self.B_Z_interp = LinearNDInterpolator(self.Delaunay_eq,self.B_Z)

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
        #set up interpolators using extrapolated samples
        self.ne0_interp = interp1d(self.a_1D,self.ne0_1D)
        self.Te0_interp = interp1d(self.a_1D,self.Te0_1D)
        
    def load_2D_fluc(self,fname_pattern = r'snap(\d+)_fpsdp.json'):
        
        
        
        
        
        
        
        
        
        
