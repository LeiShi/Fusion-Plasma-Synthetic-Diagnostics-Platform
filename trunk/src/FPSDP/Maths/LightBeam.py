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
from ..GeneralSettings.UnitSystem import cgs

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
                       (Optional, default is 1)
    :param float P_total: The total power of the beam. Optional, if given, *E0*
                          argument will be ignored. The amplitude of the beam
                          will be determined by P_total.
                         
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
        shaped Gaussian beams may have different waist width on different 
        directions. i.e. :math:`w_{0x}` and :math:`w_{0y}`.
        
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
            (\lambda \ll w_0).
    
    Electric Field Expression
    ==========================
    
    The electric field complex amplitude :math:`E(x, y, z)`can be evaluated 
    using the following formula [1]_:
    
    .. math::
        E(x, y, z)=\frac{E_0}{u_x(0,0)u_y(0,0)} \mathcal{e}^{ikz} u_x(x,z) 
                    u_y(y,z),
    
    where

    .. math::    
        u_x(x,z) = \frac{1}{\sqrt{q_x(z)}} 
                   \exp\left(\mathrm{i} k \frac{x^2}{2q_x(z)} \right),
                   
        u_y(y,z) = \frac{1}{\sqrt{q_y(z)}} 
                   \exp\left(\mathrm{i} k \frac{y^2}{2q_y(z)} \right),
                   
    and :math:`k \equiv 2\pi/\lambda` is the magnitude of the wave vector,
    :math:`q(z) \equiv z-\mathrm{i}z_R` the complex beam parameter. Note that
    since we are using a different convention for :math:`\omega` compared to 
    that used in Ref [1]_, namely, we assume the wave goes like 
    :math:`e^{-i\omega t}` instead of :math:`e^{i\omega t}`, we need to take 
    complex conjugate on spatial terms to keep the wave propagating in positive
    z direction. This applies to the definition of :math:`q` terms and the sign 
    in front of :math:`ik` part.
    
    Calculation of the Amplitude from Total Power
    =============================================
    
    If ``P_total`` is given, then the center amplitude :math:`E_0` will be 
    determined by it.
    
    The time-averaged Poynting flux is (in Gaussian unit):
    
    .. math::
        S_m = \frac{c}{8\pi}E_m \times B_m^*,
    
    where :math:`E_m` and :math:`B_m` are the magnitude of the field.    
    
    For our Gaussian beam at waist plane, :math:`E_m = B_m` and they are 
    perfectly in phase. So, 
    
    .. math::
        S_m = \frac{c}{8\pi}|E_m|^2.
        
    The total power is then
    
    .. math::
        P_{total} = \int\int S_m \mathrm{d}x \, \mathrm{d}y 
                = \frac{c}{8\pi}|E_0|^2\int\int \exp\left(-\frac{x^2}{w_{0x}^2}
                    - \frac{y^2}{w_{0y}^2}\right) \mathrm{d}x \,\mathrm{d}y.
                    
    The Gaussian integration over x and y gives us :math:`\pi`, so we have the 
    relation between :math:`E_0` and :math:`P_{total}`:
    
    .. math::
        E_0 = \sqrt{\frac{8 P_{total}}{c}}
        
    We choose :math:`E_0` to be real for simplicity.
    
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
                 w_0z=None, tilt_v=0, tilt_h=0, rotation=0, E0=1, 
                 P_total=None):
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
        if (P_total is not None):
            self.P_total = P_total
            self.E0 = np.sqrt(8*P_total/cgs['c']) 
    
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
        k = 2*np.pi/self.wave_length
        qx = z - 1j*zrx
        qy = z - 1j*zry
        ux = 1/sqrt(qx)*np.exp(1j*k*x*x/(2*qx))
        uy = 1/sqrt(qy)*np.exp(1j*k*y*y/(2*qy))
        
        return self.E0*ux*uy*np.exp(1j*k*z)*np.sqrt(zry*zrx)
        
        
        

