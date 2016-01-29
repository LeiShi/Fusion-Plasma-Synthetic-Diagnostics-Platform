# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:54:24 2016

@author: lei

This module contains mathmatical formula to evaluate Gaussian light beam 
propating in vacuum at certain angle
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.lib.scimath import sqrt

from .CoordinateTransformation import rotate

class LightBeam(object):
    r"""Abstract base class for light beams
    
    Derived classes must substantiate __call__ method which returns the 
    electric field at given coordinates
    """
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __call__(self, coords):
        pass

class GaussianBeam(LightBeam):
    r"""Gaussian light beam propagating in uniform medium
    
    :param float wave_length: the wave length :math:`\lambda`
    :param float waist_x: The x' coordinate of waist location in Lab frame
    :param float waist_y: The y' coordinate of waist location in Lab frame 
    :param float waist_z: The z' coordinate of waist location in Lab frame
                          (Optional, Default to be 0)  
    :param float tilt_v: the tilt-angle in vertical plane in radian. 0 is 
                         defined 
                         as along negative x' axis, and tilt_y>0 if the beam
                         is tilted upwards, i.e. towards positive y' direction.
                         (Optional, Default to be 0)
    :param float tilt_h: the tilt-angle in x'-z' plane in radian. 0 is defined 
                         as along negative x' axis, and tilt_z>0 if the beam
                         is tilted upwards, i.e. towards positive z' direction
                         (Optional, Default to be 0)
    :param float rotation: (Optional Default to be 0) 
                           the rotated angle of the elliptic Gaussian. It is 
                           defined as the angle between local y axis and lab y'
                           axis if local z axis has been rotated to align with
                           lab -x' axis. This angle is the tilted angle of the
                           ellipse in the transvers plane. 
    :param float w_0y: waist width in the vertical-like direction (eventually 
                       aligns with y' axis)
    :param float w_0z: waist width in the horizontal-like direction (eventually
                       aligns with z' axis) (Optional, default to be same as 
                       w_0y)  
    :param complex E0: The complex amplitude at beam frame origin
                         
    Attributes:
    
        rayleigh_range:
            Rayleigh range of the beam. Defined below.
            
            tuple (z_Ry, z_Rz)
        
        divergence:
            Divergence of the beam. Defined below
            
            tuple (theta_y, theta_z)
    
    Methods:
        
        curvature(z):
            returns the curvature of the wave front surface at given central 
            light path locations.
            
        __call__(coords):
            returns the E field complex amplitude at given locations in lab 
            frame. coords must be a length 3 list containing [Z3D, Y3D, X3D]
            coordinates data. All of the coordinate arrays must have the same
            shape, and the returned array will have the same shape.
        
        
        
    
    Definition
    ==========
    
    A Gaussian light beam is a monochromatic electromagnetic wave whose 
    transverse magnetic and electric field amplitude profiles are given by 
    the Gaussian function.    
    
    All notations here will follow the convention in [1]_.    
    
    Coordinates
    -----------
    
    We will use two set of frames here.
    
    1. Beam frame
        In Beam frame :math:`{x, y, z}`, :math:`z` is the central light path 
        direction, :math:`x` and :math:`y` are the two transverse directions 
        which align with the elliptical axis of the Gaussian profile. In 
        circular Gaussian case, :math:`r \equiv \sqrt{x^2+y^2}` is used.
        
    2. Lab frame
        Lab frame :math:`{x', y', z'}` will be used as in usual convention. 
        :math:`x` will be along major radius direction, :math:`y` the vertical 
        direction, and z locally toroidal direction.
        
    Parameters
    ----------
    
    Several parameters are usually used to describe a Gaussian beam:
    
    waist width :math:`w_0`:
        The waist, also called *focus*, is the narrowest location in the beam.
        The waist width :math:`w_0` is defined as the 1/e half-width in 
        transverse directions of field amplitude at the waist. Elliptical 
        shaped Gaussian beams may have different waist width at different 
        
    Rayleigh range :math:`z_R`:
        Rayleigh range is the distance along the beam from the waist where 1/e
        half-width becomes :math:`\sqrt{2}w_0`. It is related to :math:`w_0` as
        
        .. math::
            z_R = \frac{\pi w_0^2}{\lambda},
            
        where :math:`\lambda` is the wave length.
        
    Beam divergence :math:`\theta`:
        Far away from waist, the 1/e half-width is proportional to z. If we 
        draw a line along the 1/e half-width, it asymptotically has a constant
        angle with the central light path. This is defined as the beam 
        divergence. When it's small (paraxial case), it has the following 
        relation to the waist width.
        
        .. math::
            \theta \approx \frac{\lambda}{\pi w_0} \quad 
            (\lambda \ll w_0)}.
    
    Electric Field Expression
    ==========================
    
    The electric field complex amplitude :math:`E(x, y, z)`can be evaluated 
    using the following formula [1]_:
    
    .. math::
        E(x, y, z)=\mathrm{i}\sqrt{z_{Rx}z_{Ry}}E_0 u_x(x,z) u_y(y,z),
    
    where

    .. math::    
        u_x(x,z) = \frac{1}{\sqrt{q_x(z)}} 
                   \exp(-\mathrm{i} k \frac{x^2}{2q_x(z)} ),
                   
        u_y(y,z) = \frac{1}{\sqrt{q_y(z)}} 
                   \exp(-\mathrm{i} k \frac{y^2}{2q_y(z)} ),
                   
    and :math:`k \equiv 2\pi/\lambda` is the magnitude of the wave vector,
    :math:`q(z) \equiv z+\mathrm{i}z_R` the complex beam parameter.
    
    Coordinate Transformation
    =========================
    
    The input lab coordinates will be transformed into beam coordinates first, 
    then the complex amplitude is evaluated in beam frame using the above 
    formula.
    
    The coordinate transformation consists of 5 operations.
    
    translation:
        from lab frame origin to the beam frame origin, which is the waist 
        location.
    rotation along y' axis:
        by the angle of *tilt_h* 
    rotation along z' axis:
        by the angle of -*tilt_v*
    rotation along x' axis:
        by the angle of -*rotation*
    substitution:
        -x'->z, y'->y, z'->x
            
    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_beam        
        
    """
    
    def __init__(self, wave_length, waist_x, waist_y, w_0y, waist_z=0, 
                 w_0z=None, tilt_v=0, tilt_h=0, rotation=0, E0 = 1):
        self.wave_length = wave_length
        self.waist_loc = [waist_x, waist_y, waist_z]
        self.w_0y = w_0y
        if w_0z is not None:
            self.w_0z = w_0z
        else:
            self.w_0z = w_0y
        self.tilt_v = tilt_v
        self.tilt_h = tilt_h
        self.rotation = rotation
        self.E0 = E0
    
    @property    
    def reighlay_range(self):
        zry = np.pi*self.w_0y*self.w_0y/self.wave_length
        zrz = np.pi*self.w_0z*self.w_0z/self.wave_length
        return (zry, zrz)
        
    @property
    def divergence(self):
        thetay = self.wave_length/(np.pi*self.w_0y)
        thetaz = self.wave_length/(np.pi*self.w_0z)
        
        return (thetay, thetaz)
        
    def curvature(self, z):
        r""" curvature of the wave front at z
        
        .. math::
            \kappa = \frac{1}{R(z)}
            
            R(z) = z\left[ 1+ \left(\frac{z}{z_R}\right)^2 \right]
        
        :param z: z values to evaluate the curvature
        :type z: ndarray of float of shape (nz,)
        
        :return: curvature in y(y') and x(z') direction
        :rtype: ndarray of float with shape (nz, 2)
        """
        z = z[..., np.newaxis]
        zr = self.reighlay_range
        R = z * ( 1+ z*z/(zr*zr) )
        return 1/R
        
    def __call__(self, coords):
        r""" evaluate complex eletric field amplitude at coords
        
        :param coords: coordinates of points in lab frame
        :type coords: list of ndarrays, NOTE: the order is [Z, Y, X]
        
        :return: complex electric field amplitude at coords
        :rtype: ndarray of complex with same shape as coords[0]
        """        
        zn = np.copy(coords[0])
        yn = np.copy(coords[1])
        xn = np.copy(coords[2])        
                
        # Coordinate transform into beam frame
        
        # step 1, move to beam frame origin
        
        xn = xn-self.waist_loc[0]
        yn = yn-self.waist_loc[1]
        zn = zn-self.waist_loc[2]
        
        # step 2, rotate along y' axis
        
        xn, yn, zn = rotate('y', self.tilt_h, [xn, yn, zn])
        
        # step 3, rotate along z' axis
        
        xn, yn, zn = rotate('z', -self.tilt_v, [xn, yn, zn])
        
        # step 4, rotate along x' axis
        
        xn, yn, zn = rotate('x', -self.rotation, [xn, yn, zn])
        
        # step 5, coordinate substitution
        
        z = -xn
        y = yn
        x = zn
        
        # now, we can evaluate electric field
        zry, zrx = self.reighlay_range
        qx = z + 1j*zrx
        qy = z + 1j*zry
        ux = 1/sqrt(qx)*np.exp(-1j*(2*np.pi/self.wave_length)*x*x/(2*qx))
        uy = 1/sqrt(qy)*np.exp(-1j*(2*np.pi/self.wave_length)*y*y/(2*qy))
        
        return 1j*self.E0*sqrt(zrx*zry)*ux*uy
        
        
        

