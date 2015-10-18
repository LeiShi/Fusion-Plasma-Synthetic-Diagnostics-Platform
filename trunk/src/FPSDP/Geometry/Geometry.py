# -*- coding: utf-8 -*-
"""
Geometry module

Provide common analytic tokamak geometrys, including Slab, Cylindrical, and Toroidal geometry. 

Created on Sun Oct 18 11:13:42 2015

@author: lei
"""

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

class Geometry(object):
    """Abstract base class for Geometry objects
    Defines the following methods:
        to_major_cylindrical(r,theta,phi): for given (r,theta, phi) coordinates, return the corresponding Major Cylindrical coordinates (R,Z,Phi)
        to_cartesian: for given (r, theta, phi) coordinates, return the corresponding Cartesian coordinates (X,Y,Z)
        from_major_cylindrical(R,Z,Phi) : return (r, theta, phi) for given (R,Z,Phi)
        from_cartesian: return (r,theta,phi) for given (X,Y,Z)
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, name):
        self._name = name
        
    def __str__(self):
        return self._name
    
    @abstractmethod
    def to_major_cylindrical(self,r,theta,phi):
        """Calculate Major cylindrical Coordinates corresponding to the given (r,theta,phi) coordinates 
        :param r: radial-like coordinates in chosen geometry
        :type r: numpy array of float
        :param theta: poloidal-like coordinates in chosen geometry
        :type theta: numpy array of float, same shape as *r*
        :param phi: toroidal-like coordinate in chosen geometry
        :type phi: numpy array of float, same shape as *r*
        
        :return RPhiZ: coordinates in Major cylindrical Coordinates
        :rtype RPhiZ: tuple, (R,Phi,Z), each is a numpy array of float, same shape as *r*
        """
        pass
    
    @abstractmethod
    def  to_cartesian(self,r,theta,phi):
        """Calculate Cartesian Coordinates corresponding to the given (r,theta,phi) coordinates 
        :param r: radial-like coordinates in chosen geometry
        :type r: numpy array of float
        :param theta: poloidal-like coordinates in chosen geometry
        :type theta: numpy array of float, same shape as *r*
        :param phi: toroidal-like coordinate in chosen geometry
        :type phi: numpy array of float, same shape as *r*
        
        :return XYZ: coordinates in Cartesian Coordinates
        :rtype XYZ: tuple, (X,Y,Z), each is a numpy array of float, same shape as *r*
        """
        pass
    
    @abstractmethod
    def from_major_cylindrical(self,R,Phi,Z):
        """Calculate local geometry (r,theta,phi) Coordinates corresponding to the given Major cylindrical coordinates 
        :param R: Major Radial coordinate in tokamak major cylindrical coordinates
        :type R: numpy array of float
        :param Phi: Toroidal coordinate in tokamak major cylindrical coordinates
        :type Phi: numpy array of float, same shape as *R*
        :param Z: Vertical coordinate in tokamak major cylindrical coordinates
        :type Z: numpy array of float, same shape as *R*
        
        :return rtp: coordinates in local geometry coordinates
        :rtype rtp: tuple, (r,theta,phi), each is a numpy array of float, same shape as *R*
        """
        pass
    
    @abstractmethod
    def from_cartesian(self,X,Y,Z):
        """Calculate local geometry (r,theta,phi) Coordinates corresponding to the given Cartesian coordinates in lab frame 
        :param X: Radial coordinate in Cartesian coordinates, chosen as same as R in Major cylindrical coordinates
        :type X: numpy array of float
        :param Y: Vertical coordinate in Cartesian coordinates
        :type Y: numpy array of float, same shape as *X*
        :param Z: Horizontal coordinate in Cartesian coordinates, perpendicular to both X and Y
        :type Z: numpy array of float, same shape as *X*
        
        :return rtp: coordinates in local geometry coordinates
        :rtype rtp: tuple, (r,theta,phi), each is a numpy array of float, same shape as *X*
        """
        pass
    
    
class Slab(Geometry):
    """Slab Geometry
    Slab Geometry is a local cartesian coordinates where curvature effects are ignored.
    It is commonly used in local analysis, where extent of the grid is much smaller in every dimension compared to the Tokamak's size. 

    The coordinates convention here is:
    x: radial-like coordinate
    y: horizontal coordinate
    z: vertical coordinate
    x-y-z constructs a right-handed coordinates, corresponding to R-Phi-Z in major cylindrical coordinates.
    
    """
    
    def __init__(self, R0,Z0=0,Phi0=0, radial_zoom_ratio = 1, poloidal_zoom_ratio = 1, toroidal_zoom_ratio = 1):
        """To initialize a slab geometry, one should specify the origin of the local slab cooridnates in the Tokamak's major cylindrical coordinates, and the ratio between local coordinates and the global ones
        :param float R0: Major Radius coordinate of local frame's origin
        :param float Z0: Vertical coordinate of local frame's origin, default to be 0, meaning the midplane of the machine
        :param float Phi0: Toroidal cooridnate of local frame's origin, default to be 0. For machine with toroidal symmetry, this parameter should not matter much, as long as user define relative toroidal coordinate accordingly.
        :param float r_zoom_ratio: Zoom ratio in radial direction. For a displacement in local coordinates, :math:`\Delta r`, the actual distance in global coordinates is :math:`\Delta R=\Delta x \cdot r_zoom_ratio`
        :param float theta_zoom_ratio: Zoom ratio in vertical direction
        :param float phi_zoom_ratio: Zoom ratio in horizontal direction          
        """
        super(Slab,self).__init__('Slab Geometry')
        self._R0 = R0
        self._Z0 = Z0
        self._Phi0 = Phi0
        self._rzr = radial_zoom_ratio
        self._pzr = poloidal_zoom_ratio
        self._tzr = toroidal_zoom_ratio

    def __str__(self):
        
        return '{0}:\n Origin at: \n (R0,Z0,Phi0) = ({1},{2},{3}) \n Zoom ratios in 3 directions: ({4},{5},{6})'.format(self._name, self._R0, self._Z0, self._Phi0, self._rzr, self._tzr, self._pzr)
        
    def to_major_cylindrical(self, x, y, z):
        """converting slab cooridnates to major cylindrical coordinates
        :param x: radial coordinate
        :type x: numpy array of float        
        :param y: horizontal coordinate
        :type y: numpy array of float
        :param z: vertical coordinate
        :type z: numpy array of float
        
        :return RPhiZ: Major Cylindrical coordinates
        :rtype RPHiZ: tuple, (R,Phi,Z), each is numpy array of float
        """
        Z = z*self._pzr + self._Z0
        
        X = (self._R0 + x*self._rzr)
        Y = y*self._tzr

        R = np.sqrt(X*X + Y*Y)
        Phi = self._Phi0 + np.arcsin(Y/R)        
        
        return (R,Phi,Z)
        
    def to_cartesian(self,x,y,z):
        """converting slab coordinate to Cartesian coordinate in lab frame
        :param x: radial coordinate
        :type x: numpy array of float        
        :param y: horizontal coordinate
        :type y: numpy array of float
        :param z: vertical coordinate
        :type z: numpy array of float
        
        :return XYZ: Cartesian coordinates, convention is X:Radial, Y:Vertical, Z:Horizontal, X-Y-Z constructs right-handed system
        :rtype XYZ: tuple, (X,Y,Z), each is numpy array of float
        """
            
        #First, we calculate the (X,Y,Z) values in the frame that Phi0=0
        X0 = self._R0
        Y0 = self._Z0
        Z0 = 0
        
        X = X0+ x*self._rzr
        Y = Y0 + z*self._pzr
        
        #Note that in our convention, Z is in opposite direction of Phi and y
        Z = Z0 - y*self._tzr
        
        #Now we rotate the whole grid by Phi0 to get the coordinate values in lab frame, note that the lab frame is obtained by rotating the old frame by positive Phi0 
        cosphi = np.cos(self._Phi0)
        sinphi = np.sin(self._Phi0)        
        X_lab = X*cosphi + Z*sinphi
        Z_lab = -X*sinphi + Z*cosphi
        Y_lab = Y

        return (X_lab,Y_lab,Z_lab)

    def from_major_cylindrical(self,R,Phi,Z):
        """converting major cylindrical coordinates to slab cooridnates
        :param R: major radial coordinate
        :type R: numpy array of float        
        :param Phi: toroidal angle coordinate
        :type Phi: numpy array of float
        :param Z: vertical coordinate
        :type Z: numpy array of float
        
        :return xyz: slab coordinates
        :rtype xyz: tuple, (x,y,z), each is numpy array of float
        """
        
        dPhi = Phi-self._Phi0
        
        inverse_rzr = 1./self._rzr
        inverse_pzr = 1./self._pzr
        inverse_tzr = 1./self._tzr
        
        z = (Z-self._Z0)*inverse_pzr
        y = R*np.sin(dPhi)
        x = R*np.cos(dPhi) - self._R0
        
        return (x,y,z)
        
    def from_cartesian(self,X,Y,Z):
        """converting  Cartesian coordinate in lab frame to slab coordinate
        :param X: Radial coordinate
        :type x: numpy array of float        
        :param Y: vertical coordinate
        :type y: numpy array of float
        :param Z: horizontal coordinate
        :type z: numpy array of float
        
        :return xyz: slab coordinates
        :rtype xyz: tuple, (x,y,z), each is numpy array of float
        """
        inverse_rzr = 1./self._rzr
        inverse_pzr = 1./self._pzr
        inverse_tzr = 1./self._tzr

        cosphi = np.cos(-self._Phi0)
        sinphi = np.sin(-self._Phi0)

        X_rot = X*cosphi + Z*sinphi
        Z_rot = -X*sinphi + Z*cosphi

        z = (Y-self._Z0)*inverse_pzr        
        
        x = (X_rot-self._R0)*inverse_rzr
        
        y = -Z_rot*inverse_tzr
        
        return (x,y,z)
