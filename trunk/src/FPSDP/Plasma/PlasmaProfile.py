# -*- coding: utf-8 -*-
"""
This module defines the plasma profile class.

All profile generators -- analytical, experimental, or simulational-- must 
provide output of plasma profile object, which contains all useful information
of certain synthetic diagnostic requires.

Created on Mon Jan 18 13:27:04 2016

@author: lei
"""
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..Geometry.Grid import Grid
from ..GeneralSettings.UnitSystem import UnitSystem, cgs

class IonClass(object):
    """General class for a kind of ions
    
    :param string name: name of the ion kind
    :param int mass: mass of the ion, in unit of atomic mass
    :param unit charge: electrical charge of the ion, in unit of elementary 
                        charge
    """
    
    def __init__(self, mass, charge, name='temperary_ion_kind'):
        """
        """        
        
        self._name = name
        self.mass = mass
        self.charge = charge
        
    def __str__(self):
        
        return 'Ion kind: {}\nMass: {} proton mass\nCharge: {} elementary \
charge'.format(self._name, self.mass, self.charge)

    def info(self):
        print str(self)


# some pre-defined ion classes

HYDROGEN = IonClass(1, 1, 'H+')
DEUTERIUM = IonClass(2, 1, 'D+')
TRITIUM = IonClass(3, 1, 'T+')
# more ion species can be added below

class PlasmaProfile(object):
    """Base class for all plasma profiles required for synthetic diagnostics.
        
    In general, a profile can have no plasma quantities, but must have a grid
    layout to contain possible quantities.
    
    :param grid: Grid for the profiles
    :type grid: :py:class:`..Geometry.Grid.Grid` object
    """
    
    def __init__(self, grid, unitsystem):
        assert isinstance(grid, Grid)
        assert isinstance(unitsystem, UnitSystem)
        self.grid = grid
        self.unit_system = unitsystem
        self._name = 'General plasma profile'
        
    def physical_quantities(self):
        return 'none'
        
    def __str__(self):
        return '{}:\n\nUnit System:{}\nGrid:{}\nPhysical Quantities:\n{}\n'.\
                format(self._name, str(self.unit_system),str(self.grid), 
                       self.physical_quantities())
        
class ECEI_Profile(PlasmaProfile):
    """Plasma profile for synthetic Electron Cyclotron Emission Imaging.
    
    ECEI needs the following plasma quantities:
    
    :var ne0: equilibrium electron density
    :var Te0: equilibrium electron temperature
    :var B0: equilibrium magnetic field
    :var dne: *optional*, electron density perturbation
    :var Te_para:  *optional*, fluctuated electron temperature parallel to B
    :var Te_perp: *optional*, fluctuatied electron temperature perpendicular 
                  to B
    :var
    
    These should all be passed in compatible with the ``grid`` specification.
    :raises AssertionError: if any of the above quantities are not compatible
    """
    def __init__(self, grid, ne0, Te0, B0, time=None, dne=None, dTe_para=None, 
                 dTe_perp=None, dB=None, unitsystem = cgs):
        assert isinstance(grid, Grid)
        assert isinstance(unitsystem, UnitSystem)
        # test if all equilibrium quantities has same shape as the grid
        assert ne0.shape == grid.shape
        assert Te0.shape == grid.shape
        assert B0.shape == grid.shape
        # test if all perturbed quantities has shape as first dim = len(time), 
        # and the rest == grid.shape
        self.has_dne = False
        if dne is not None:
            assert time is not None
            assert dne.shape[0] == len(time)
            assert dne.shape[1:] == grid.shape[:]
            self.has_dne = True
            self.time = time
            self.dne = dne
            
        self.has_dTe_para = False
        if dTe_para is not None:
            assert time is not None
            assert dTe_para.shape[0] == len(time)
            assert dTe_para.shape[1:] == grid.shape[:]
            self.has_dTe_para = True
            self.time = time
            self.dTe_para = dTe_para
            
        self.has_dTe_perp = False
        if dTe_perp is not None:
            assert time is not None
            assert dTe_perp.shape[0] == len(time)
            assert dTe_perp.shape[1:] == grid.shape[:]
            self.has_dTe_perp = True
            self.time = time
            self.dTe_perp = dTe_perp
            
        self.has_dB = False
        if dB is not None:
            assert time is not None
            assert dB.shape[0] == len(time)
            assert dB.shape[1:] == grid.shape[:]
            self.has_dB = True
            self.time = time
            self.dB = dB
        self._name = 'Electron Cyclotron Emission Imaging Plasma Profile'
        self.unit_system = unitsystem
        self.grid = grid
        self.ne0 = ne0
        self.Te0 = Te0
        self.B0 = B0
        
        
    def setup_interps(self, equilibrium_only = True):
        """setup interpolators for frequent evaluation of profile quantities on
        given locations.
        
        """
        mesh = self.grid.get_mesh()
        self.Te0_sp = RegularGridInterpolator(mesh, self.Te0)
        self.ne0_sp = RegularGridInterpolator(mesh, self.ne0)
        self.B0_sp = RegularGridInterpolator(mesh, self.B0)
        if not equilibrium_only:
            if (self.has_dne):
                self.dne_sp = []
                for i in range(len(self.time)):
                    self.dne_sp.append(RegularGridInterpolator(mesh, 
                                                         self.dne[i]))
            if (self.has_dTe_para):
                self.dTe_para_sp = []
                for i in range(len(self.time)):
                    self.dTe_para_sp.append( RegularGridInterpolator(mesh, 
                                                         self.dTe_para[i]))
            if (self.has_dTe_perp):
                self.dTe_perp_sp = []
                for i in range(len(self.time)):
                    self.dTe_perp_sp.append( RegularGridInterpolator(mesh, 
                                                         self.dTe_perp[i]))
                                                         
            if (self.has_dB):
                self.dB_sp = []
                for i in range(len(self.time)):
                    self.dB_sp.append( RegularGridInterpolator(mesh, 
                                                         self.dB[i]))

    def get_ne0(self, coordinates):
        """return ne0 interpolated at *coordinates*
        
        :param coordinates: Coordinates given in (Z,Y,X)*(for 3D)* or (Z,R) 
                            *(for 2D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        """
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        transpose_axes = range(1,coordinates.ndim)
        transpose_axes.append(0)
        points = np.transpose(coordinates, transpose_axes)
        try:
            return self.ne0_sp(points)
        except AttributeError:
            print 'ne0_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
            mesh = self.grid.get_mesh()
            ne0_sp = RegularGridInterpolator(mesh, self.ne0)
            return ne0_sp(points)
            

    def get_Te0(self, coordinates):
        """return Te0 interpolated at *coordinates*
        
        :param coordinates: Coordinates given in (Z,Y,X)*(for 3D)* or (Z,R) 
                            *(for 2D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        """
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        transpose_axes = range(1,coordinates.ndim)
        transpose_axes.append(0)
        points = np.transpose(coordinates, transpose_axes)
        try:
            return self.Te0_sp(points)
        except AttributeError:
            print 'Te0_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
            mesh = self.grid.get_mesh()
            Te0_sp = RegularGridInterpolator(mesh, self.Te0)
            return Te0_sp(points)


    def get_B0(self, coordinates):
        """return B0 interpolated at *coordinates*
        
        :param coordinates: Coordinates given in (Z,Y,X)*(for 3D)* or (Z,R) 
                            *(for 2D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        """
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        transpose_axes = range(1,coordinates.ndim)
        transpose_axes.append(0)
        points = np.transpose(coordinates, transpose_axes)
        try:
            return self.B0_sp(points)
        except AttributeError:
            print 'B0_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
            mesh = self.grid.get_mesh()
            B0_sp = RegularGridInterpolator(mesh, self.B0)
            return B0_sp(points)  
            

    def get_dne(self, coordinates, time=None):
        """return dne interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X)*(for 3D)* or (Z,R) 
                            *(for 2D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid* 
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dne time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dne
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time))
        elif isinstance(time, int):
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dne_sp[time](points)
                
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
                mesh = self.grid.get_mesh()            
                dne_sp = RegularGridInterpolator(mesh, self.dne[time])
                result = dne_sp(points)
            return result
        else:
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for t in time:
                    result[t] = self.dne_sp[t](points)
                
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
                mesh = self.grid.get_mesh()            
                for t in time:
                    dne_sp = RegularGridInterpolator(mesh, self.dne[t])
                    result[t] = dne_sp(points)
            return result
            
    def get_dB(self, coordinates, time=None):
        """return dB interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X)*(for 3D)* or (Z,R) 
                            *(for 2D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dne time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dB
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time))
            
        elif isinstance(time, int):
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dB_sp[time](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
                mesh = self.grid.get_mesh()            
                dB_sp = RegularGridInterpolator(mesh, self.dB[time])
                result = dB_sp(points)
            return result
        
        else:
            # Note that the first dimension in coordinates is the number of spatial
            # axis. We can simply overwrite it with time steps to get the desired
            # shape of result. 
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for t in time:
                    result[t] = self.dB_sp[t](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
                mesh = self.grid.get_mesh()            
                for t in time:
                    dB_sp = RegularGridInterpolator(mesh, self.dB[t])
                    result[t] = dB_sp(points)
            return result


    def get_dTe_perp(self, coordinates, time=None):
        """return dTe_perp interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X)*(for 3D)* or (Z,R) 
                            *(for 2D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dTe_perp time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dTe_perp
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time))        
        elif isinstance(time, int):
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dTe_perp_sp[time](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
                mesh = self.grid.get_mesh()            
                dte_sp = RegularGridInterpolator(mesh, self.dTe_perp[time])
                result = dte_sp(points)
            return result
                
        else:
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for t in time:
                    result[t] = self.dTe_perp_sp[t](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
    generated. If this message shows up a lot of times, please consider calling \
    setup_interp function first.'
                mesh = self.grid.get_mesh()            
                for t in time:
                    dte_sp = RegularGridInterpolator(mesh, self.dTe_perp[t])
                    result[t] = dte_sp(points)
            return result

            
    def get_dTe_para(self, coordinates, time=None):
        """return dTe_para interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X)*(for 3D)* or (Z,R) 
                            *(for 2D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dTe_para time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dTe_para
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time))  
            
        elif isinstance(time, int):
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dTe_para_sp[time](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interp function first.'
                mesh = self.grid.get_mesh()            
                dte_sp = RegularGridInterpolator(mesh, self.dTe_para[time])
                result = dte_sp(points)
            return result        
        else:
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for t in time:
                    result[t] = self.dTe_para_sp[t](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
    generated. If this message shows up a lot of times, please consider calling \
    setup_interp function first.'
                mesh = self.grid.get_mesh()            
                for t in time:
                    dte_sp = RegularGridInterpolator(mesh, self.dTe_para[t])
                    result[t] = dte_sp(points)
            return result
            
    def get_ne(self, coordinates, eq_only=True, time=None):
        """wrapper for getting electron densities
        
        If eq_only is True, only equilibirum density is returned
        otherwise, the total density is returned.
        """
        if eq_only:
            return self.get_ne0(coordinates)
        else:
            if self.has_dne:
                return self.get_ne0(coordinates) + self.get_dne(coordinates, 
                                                                time)
            else:
                raise ValueError('get_ne is called with eq_only=False, but no \
electron density perturbation data available.')
                
    def get_B(self, coordinates, eq_only=True, time=None):
        """wrapper for getting magnetic field strength
        
        If eq_only is True, only equilibirum density is returned
        otherwise, the total density is returned.
        
        if time is None, all available time steps for perturbations are 
        returned. Otherwise the given time steps are returned.
        """
        if eq_only:
            return self.get_B0(coordinates)
        else:
            if self.has_dB:
                return self.get_B0(coordinates) + self.get_dB(coordinates, 
                                                              time)
            else:
                raise ValueError('get_B is called with eq_only=False, but no \
electron density perturbation data available.')

    def get_Te(self, coordinates, eq_only=True, perpendicular = True, 
               time=None):
        """wrapper for getting electron densities
        
        If eq_only is True, only equilibirum density is returned
        otherwise, the total density is returned.
        
        if time is None, all available time steps for perturbations are 
        returned. Otherwise the given time steps are returned.
        """
        if eq_only:
            return self.get_Te0(coordinates)
        elif perpendicular:
            if self.has_dTe_perp:
                return self.get_Te0(coordinates) + \
                       self.get_dTe_perp(coordinates, time)
            else:
                raise ValueError('get_Te is called with eq_only=False, \
perpendicular=True but no electron perpendicular temperature perturbation data\
 available.')
        else:
            if self.has_dTe_para:
                return self.get_Te0(coordinates) + \
                       self.get_dTe_para(coordinates, time)
            else:
                raise ValueError('get_Te is called with eq_only=False, \
perpendicular=True but no electron perpendicular temperature perturbation data\
 available.')


    
    def physical_quantities(self):
        
        result = 'Equilibrium:\n\
    Electron density: ne0\n\
    Electron temperature: Te0\n\
    Magnetic field: B0\n\
Fluctuation:\n'
        if self.has_dne:
            result += '    Electron density: dne\n'
        if self.has_dTe_para:
            result += '    Electron parallel temperature: dTe_para\n'
        if self.has_dTe_perp:
            result += '    Electron perpendicular temperature: dTe_perp\n'
        if self.has_dB:
            result += '    Magnetic field magnitude: dB\n'
        return result
