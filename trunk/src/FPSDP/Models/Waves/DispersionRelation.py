# -*- coding: utf-8 -*-
r"""
Created on Thu Feb 11 19:01:22 2016

@author: lei

This module provides means to solve linear dispersion relation.

Formula
========

The linearized equation for electromagnetic waves in plasma is formally:

.. math::

    \bf{\Lambda}\cdot \bf{E} = 0
    
where

.. math::
    \bf{\lambda} \equiv 
    \begin{pmatrix}
        \epsilon_{xx}-n_\parallel^2 & \epsilon_{xy} & \epsilon_{xz}+
        n_\parallel n_\perp \\
        \epsilon_{yx} & \epsilon_{yy}-n^2 & \epsilon_{yz} \\
        \epsilon_{zx}+n_\parallel n_\perp & \epsilon_{zy} & 
        \epsilon_{zz}-n_\perp^2
    \end{\pmatrix}  ,
    
and :math:`\vec{n} \equiv c\vec{k}/\omega = n_\parallel \hat{x} + n_\perp 
\hat{z}`, :math:`\bf{\epsilon}` is the dielectric tensor. 

A non-trivial solution requires :math:`|\bf{\Lambda}|=0`, which essentially
gives function relation between the wave's frequency and wave vector. 
This is called the dispersion relation. 

Since in general :math:`\bf{\epsilon}` is a function of :math:`\omega` and 
:math:`\vec{k}`, solving the dispersion relation has to be done 
numerically.
"""

import numpy as np

from ...Plasma import DielectricTensor as dt
from ...GeneralSettings.UnitSystem import cgs  


def Lambda(omega, k_para, k_perp, dielectric, coordinates):
    r"""Evaluate the wave's dispersion matrix
    
    This function evaluates :math:`|\bf{\Lambda}(\omega, \vec{k})|` for given
    :math:`\omega` and :math:`\vec{k}`. 
     
    The root of this function gives allowed :math:`\omega` and :math:`\vec{k}`
    values for a wave.
    
    :param float omega: real frequency of the wave
    :param k_para: wave vectors parallel to magnetic field
    :type k_para: scalar or array of complex
    :param k_perp: wave vectors perpendicular to magnetic field
    :type k_perp: scalar or array of complex
    :param dielectric: dielectric tensor under consideration.
    :type dielectric: instance of 
                      :py:class:`FPSDP.Plasma.DielectricTensor.Dielectric`
    :param coordinates: spatial coordinates where dielectric tensor will be 
                        evaluated.
    :type coordinates: list of scalars or arrays. Should be the same length as 
                       dimension of the ``dielectric`` object.
    :return: Lambda function evaluated at each spatial location given by 
             ``coordinates``.
    :rtype: ndarray of float, same shape as one of the coordinates
    """
    omega = np.array(omega)
    k_para = np.array(k_para)
    k_perp = np.array(k_perp)
    coordinates = np.array(coordinates)    
    
    # Calculate the epsilon tensor
    eps = dielectric.epsilon(coordinates, omega, k_para, k_perp)
    
    # prepare all the other quantities into right shape
    frequency_shape = list(omega.shape)
    wave_vector_para_shape = list(k_para.shape)
    wave_vector_perp_shape = list(k_perp.shape)
    spatial_shape = list(coordinates[0].shape)
    f_dim = len(frequency_shape)
    sp_dim = len(spatial_shape)
    wv_para_dim = len(wave_vector_para_shape)
    wv_perp_dim = len(wave_vector_perp_shape)
    
    full_f_shape = frequency_shape
    full_f_shape.extend([1 for i in range(sp_dim + wv_para_dim + \
                                              wv_perp_dim)]) 
    omega = omega.reshape(full_f_shape)
    
    full_k_para_shape = []
    full_k_para_shape.extend([1 for i in range(f_dim)])
    full_k_para_shape.extend(wave_vector_para_shape)
    full_k_para_shape.extend([1 for i in range(wv_perp_dim)])
    full_k_para_shape.extend([1 for i in range(sp_dim)])
    k_para = k_para.reshape(full_k_para_shape)
    
    full_k_perp_shape = []
    full_k_perp_shape.extend([1 for i in range(f_dim)])
    full_k_perp_shape.extend([1 for i in range(wv_para_dim)])
    full_k_perp_shape.extend(wave_vector_perp_shape)
    full_k_perp_shape.extend([1 for i in range(sp_dim)])
    k_perp = k_perp.reshape(full_k_perp_shape)
    
    # now calculate the refractive index 
    n_para = cgs['c']*k_para/omega
    n_perp = cgs['c']*k_perp/omega  
    
    # Create the Lambda tensor
    Lambda = eps
    Lambda[0,0] += -n_para*n_para
    Lambda[0,2] += n_para*n_perp
    Lambda[1,1] += -n_para*n_para-n_perp*n_perp
    Lambda[2,0] += n_para*n_perp
    Lambda[2,2] += -n_perp*n_perp
        
    # Move the tensor axes to the last dimensions
    shape = range(2,eps.ndim)
    shape.extend([0,1])
    Lambda = np.transpose(Lambda, axes=shape)    
    
    return np.linalg.det(Lambda)
    
    
    


