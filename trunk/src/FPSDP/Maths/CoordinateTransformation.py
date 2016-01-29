# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:51:12 2016

@author: lei

Common coordinate transformations
"""
from math import sin, cos

import numpy as np


def rotate(axis, angle, coords):
    r""" Rotate the frame (x,y,z) along *axis* by *angle*.
    
    :param string axis: 'x' or 'y' or 'z'
    :param float angle: rotation angle in radian
    :param coords: points' coordinates in original frame
    :type coords: list of 3 arrays, containing (X, Y, Z) coordinates
    
    :return: points' new coordinates in rotated frame
    :rtype: list of 3 arrays, containing (X', Y', Z')
    
    rotating the frame by a angle of theta, is equivalent to rotating the 
    points by a angle of -theta, which is related to the rotational matrix
    :math:`\begin{pmatrix} \cos\;\theta & \sin\;\theta\\ -\sin\; \theta & 
    \cos\;\theta`.     
    """
    assert axis in ['x','y','z']   
    
    x = coords[0]
    y = coords[1]
    z = coords[2]
    
    if(axis == 'x'):
        yp = cos(angle)*y + sin(angle)*z
        zp = -sin(angle)*y + cos(angle)*z
        return (x, yp, zp)
    elif(axis == 'y'):
        zp = cos(angle)*z + sin(angle)*x
        xp = -sin(angle)*z + cos(angle)*x
        return (xp, y, zp)
    else:
        xp = cos(angle)*x + sin(angle)*y
        yp = -sin(angle)*x + cos(angle)*y
        return (xp, yp, z)
    
    
    
    

