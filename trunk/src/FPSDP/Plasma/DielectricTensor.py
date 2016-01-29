# -*- coding: utf-8 -*-
r"""
Created on Fri Jan 22 07:48:14 2016

@author: lei

This module is dedicated to evaluating plasma dielectric tensor 
:math:`\epsilon(\omega, \vec{k})`. 

Coordinate System
=================

Coordinate system is chosen to be such that :math:`\hat{z}` is in background
magnetic field :math:`\vec{B}` direction, and wave vector 
:math:`\vec{k} = k_x \hat{x} + k_z \hat{z}`. We will denote :math:`k_x` as 
:math:`k_\perp`, and :math:`k_z` as :math:`k_\parallel`. 

    **WARNING**: The expression of dielectric tensor is obtained for 
    **uniform** plasma only. For spatially varying plasma, local uniformity is 
    assumed! So the propagating wave is analyzed as locally Fourier transformed
    components with various wave vectors. 

General Form of Dielectric Tensor
=================================
    
Plasma dielectric tensor can be, in general, written as

.. math::

    \epsilon = \bf{I} + \sum\limits_s\chi_s
    
where :math:`\bf{I}` denotes the identity tensor, and *s* species of particles.

In particular, for high frequency waves, i.e. :math:`\omega > |\Omega_e|`,
ion susceptibility effects are negligible, only electron term is retained. 

Expression for Susceptibility Tensor
====================================

In different paramter regimes, different assumptions can be used to 
significantly simplify the calculation of susceptibility tensor.

The following regimes are implemented in this module:

1. Cold Plasma [1]_

    In cold limit, susceptibility tensor is obtained by solving two-fluid 
    equations with zero pressure.
    
    As a result, :math:`\chi_s` is a function of :math:`\omega` only, not 
    depend on wave vector :math:`\vec{k}`. 
    
    It has the following form:
    
    .. math::
        
        \chi_{s,xx} = \chi_{s,yy} = \frac{\chi^+ + \chi^-}{2} \; ,
        
        \chi_{s,xy} = -\chi_{s,yx} = \frac{\rm{i}(\chi^+ - \chi^-)}{2} \; ,
        
        \chi_{s,zz} = -\frac{\omega^2_{ps}}{\omega^2}\; ,
        
    where 
    
    .. math::
        
        \chi^\pm_s = - \frac{\omega^2_{ps}}{\omega(\omega\mp\Omega_s)} \;,
        
        \omega^2_{ps} \equiv \frac{4\pi n_s q_s^2}{m_s} \;.
        
    The components not given above are all zero.
        
2. Warm Electron Plasma [2]_

    Warm electron plasma susceptibility tensor is still obtained by two-fluid 
    equations, but with finite electron parallel pressure. 
    
    The zz-component of susceptibility tensor for electrons now changed to
    
    .. math::    
        
        \chi_{zz,e} = \frac{\omega^2_{pe}}{-\omega^2 + \gamma k^2_\parallel 
        T_{e,\parallel}/m_e},
        
    where :math:`\gamma = (n+2)/n` for adiabatic process, and :math:`\gamma=1`
    for isothermal process. The key parameter to distinguish these two process
    is :math:`k_\parallel v_{e, th}/\omega`, which characterizes the ratio of 
    thermal electron travel length in a wave period, :math:`v_{e,th}/\omega`, 
    over the parallel wave length, :math:`1/k_\parallel`. So, when this ratio
    is much less than unity, the electrons can be considered adiabatic during a 
    wave period. On the other hand, if it is much greater than unity, electrons
    are effectively isothermal.
    
3. Non-relativistic Hot Plasma [3]_

    Non-relativistic hot plasma susceptibility tensor is obtained through 
    solving Vlasov-Maxwell Equations. 
    
    This limit is valid when parallel Doppler shift effect is dominant over 
    cyclotron frequency shift due to relativistic mass increasement, i.e.
    :math:`k_\parallel v_{th,\parallel} \gg n\Omega(T_\perp+T_\parallel)/2c^2`.
    In isotropic plasma, this condition becomes :math:`k_\parallel c/\omega \gg
    v_{th,\parallel}/c`. 
    
    
    Here we use the expression for an 
    anisotropic Maxwellian distribution, :math:`T_\perp \ne T_\parallel`.
     
    .. math::
     
        \chi_{s,xx} = \frac{\omega_p^2}{\omega} 
        \sum\limits_{n=-\infty}^{\infty} 
        \mathrm{e}^{-\lambda}\frac{n^2 I_n}{\lambda}A_n, 
         
        \chi_{s,yy} = \frac{\omega_p^2}{\omega} 
        \sum\limits_{n=-\infty}^{\infty} 
        \mathrm{e}^{-\lambda} 
        \left( \frac{n^2}{\lambda}+2\lambda I_n - 2\lambda I'_n \right)A_n,
         
        \chi_{s,zz} = 
        \left[ \hat{e}_\parallel \hat{e}_\parallel \frac{2\omega_p^2}
        {\omega k_\parallel w^2_\perp}V + \frac{\omega_p^2}{\omega
        } \sum\limits_{n=-\infty}^{\infty} \mathrm{e}^{-\lambda} 
        \frac{2(\omega-n\Omega)}{k_\parallel w^2_\perp} I_n B_n
        \right],
         
        \chi_{s,xy} = -\chi_{s,yx} = 
        \frac{\omega_p^2}{\omega} 
        \sum\limits_{n=-\infty}^{\infty} 
        \mathrm{e}^{-\lambda} (-\mathrm{i}n(I_n - I'_n)A_n),
         
        \chi_{s,xz} = \chi_{s,zx} = 
        \frac{\omega_p^2}{\omega} 
        \sum\limits_{n=-\infty}^{\infty} 
        \mathrm{e}^{-\lambda} \frac{k_\perp}{\Omega} \frac{nI_n}{\lambda}B_n, 
         
        \chi_{s,yz} = -\chi_{s,zy} = 
        \frac{\omega_p^2}{\omega} 
        \sum\limits_{n=-\infty}^{\infty} 
        \mathrm{e}^{-\lambda} \frac{\mathrm{i}k_\perp}{\Omega} (I_n-I'_n) B_n,
    
    where :math:`\lambda\equiv\frac{k^2_\perp w^2_\perp}{2\Omega^2} measures 
    the Finite Larmor Radius(FLR) effects, :math:`w_\perp \equiv \sqrt{2T_\perp
    /m}` is the perpendicular thermal speed. 
    `:math:`I_n = I_n(\lambda)` is the modified Bessel function of the 
    first kind, with argument :math:`\lambda`, and :math:`I'_n = (\mathrm{d}/
    \mathrm{d} \lambda)I_n(\lambda)`. :math:`V` is the parallel equilibrium 
    flow velocity. 
    
    :math:`A_n` and :math:`B_n` are given as

    .. math::
    
        A_n = \frac{T_\perp - T_\parallel}{\omega T_\parallel} + 
        \frac{1}{k_\parallel w_\parallel}
        \frac{(\omega-k_\parallel V -n\Omega)+n\Omega T_\parallel}
        {\omega T_\parallel}  Z_0 \;,
        
        B_n = \frac{1}{k_\parallel} 
        \frac{(\omega-n\Omega)T_\perp - (k_\parallel V - n\Omega)T_\parallel}
        {\omega T_\parallel}
        +\frac{1}{k_\parallel}\frac{\omega-n\Omega}{k_\parallel w_\parallel}
        \frac{(\omega - k\parallel V - n\Omega)T_\perp + n\Omega T_\parallel}
        {\omega T_parallel} Z_0 \; ,
        
        
    where :math:`Z_0 = Z_0(\zeta_n)` is the Plasma Dispersion Function [4]_, 
    and :math:`\zeta_n =(\omega -k_\parallel V -n\Omega)/{k_\parallel 
    w_\parallel}` measures the distance from the n-th order cold resonance.
    
4. Weakly Relativistic Plasma [5]_ [6]_

    Weakly relativistic plasma susceptibility tensor is obtained through 
    solving Vlasov-Maxwell system, similarly to the Non-relativistic case. 
    However, in this case, we retain the relativistic effect of the mass 
    increasement. A detailed discussion of the validity of this limit can be 
    found in [5]_ and [6]_. 
    
    Here, we use the expressions given in [5]_, and the plasma is assumed 
    isotropic. (The notations are kept the same as in [5]_ so it's easy to 
    check with the reference.)
    
    We have:
    
    .. math::
    
        \chi_{xx} = -\frac{\mu \omega^2_p}{\omega^2}
        \sum\limits_{n=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} n^2 \lambda^{p+n-1} \mathcal{F}_{p+n+3/2}
        
        \chi_{yy} = -\frac{\mu \omega^2_p}{\omega^2}
        \sum\limits_{n=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} \left[ (p+n)^2 -\frac{p(p+2n)}{2n+2p-1} \right] 
        \lambda^{p+n-1} \mathcal{F}_{p+n+3/2}
        
        \chi_{xy} = -\chi_{yx} = \mathrm{i} \frac{\mu \omega^2_p}{\omega^2}
        \sum\limits_{N=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} n(p+n) \lambda^{p+n-1} \mathcal{F}_{p+n+3/2}
        
        \chi_{xz} = \chi_{zx} = 
        -\frac{\mu \omega^2_p k_\perp k_\parallel c^2}{\omega^3 \omega_c}
        \sum\limits_{N=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} n \lambda^{p+n-1} \mathcal{F}'_{p+n+5/2}
        
        \chi_{yz} = -\chi_{zy} = -\mathrm{i} 
        \frac{\mu \omega^2_p k_\perp k_\parallel c^2}{\omega^3 \omega_c}
        \sum\limits_{N=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} (p+n) \lambda^{p+n-1} \mathcal{F}'_{p+n+5/2}
        
        \chi_{zz} = -\frac{\mu \omega^2_p}{\omega^2}
        \sum\limits_{n=-\infty}^{\infty}\sum\limits_{p=0}^{\infty} a_{pn} 
        \lambda^{p+n} (\mathcal{F}_{p+n+5/2}+ 2\psi^2 \mathcal{F}''_{p+n+7/2} )
        
    where :math:`\mathcal{F}_q = \mathcal{F}_q(\phi, \psi)` is the weakly 
    relativistic plasma dispersion function defined in [5]_, and implemented in
    [4]_. :math:`\psi = k_\parallel c^2/\omega v_t \sqrt{2}`,  
    :math:`\phi^2 = \psi^2 - \mu \delta`, :math:`\mu \equiv c^2/v_t^2`, and 
    :math:`\delta = (\omega - n\omega_c)/\omega`. The sign of the real
    (imaginary) part of :math:`\phi` is defined to be +(-) [7]_. 
        
        
References
==========

.. [1] "Waves in Plasmas", Chapter 1-3, T.H.Stix, 1992, American Inst. of 
       Physics

.. [2] "Waves in Plasmas", Chapter 3-5, T.H.Stix, 1992, American Inst. of 
       Physics

.. [3] "Waves in Plasmas", Chapter 10-7, T.H.Stix, 1992, American Inst. of 
       Physics  

.. [4] :py:module:`..Maths.PlasmaDispersionFunction`  

.. [5] I.P.Shkarofsky, "New representations of dielectric tensor elements in 
       magnetized plasma", J. Plasma Physics(1986), vol. 35, part 2, pp. 
       319-331 
       
.. [6] M.Bornatici, R. Cano, et al., "Electron cyclotron emission and 
       absorption in fusion plasmas", Nucl. Fusion(1983), vol. 23, No.9, pp. 
       1153
       
.. [7] Weakly relativistic dielectric tensor and dispersion functions of a 
       Maxwellian plasma, V. Krivenski and A. Orefice, J. Plasma Physics 
       (1983), vol. 30, part 1, pp. 125-131
#TODO Finish the docstring. 
"""
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

import numpy as np

from ..Maths.PlasmaDispersionFunction import Fq, Fmq
from .PlasmaProfile import PlasmaProfile
from ..GeneralSettings.UnitSystem import UnitSystem, cgs


class ResonanceError(Exception):
    
    def __init__(self, s):
        self.message = s
        
    def __str__(self):
        return self.message


class Susceptilibity(object):
    r"""Abstract base class for susceptibility tensor classes
    
    Methods
    =======
    
    __call__(self, coordinates):
        Calculates susceptilibity tensor elements of the particular species 
        at given coordinates.
        
    __str__(self):
        returns a description of the model used.
            
    Attributes
    ==========
    
    _name:
        The name of the model used

    _model:
        The description of the model    
    
    plasma:
        :py:class:`.PlasmaProfile.PlasmaProfile` object
    
    species:
        either 'e' or 'i', denotes electron or ion
        
    species_id:
        if species is 'ion', this number indicates which ion species to use, 
        default to be 0, which means the first kind in *plasma.ni*
        
    
        
    """
    
    @abstractmethod
    def __call__(self, plasma, coordinates, species, species_id=0):
        pass
    
    @abstractmethod
    def __str__(self):
        return '{}:\n    {}'.format(self._name, self._model)
    
    
class SusceptCold(Susceptilibity):
    r"""Cold plasma susceptibility tensor
    
    Formula
    =======
    
    In cold limit, susceptibility tensor is obtained by solving two-fluid 
    equations with zero pressure.
    
    As a result, :math:`\chi_s` is a function of :math:`\omega` only, not 
    depend on wave vector :math:`\vec{k}`. 
    
    It has the following form:
    
    .. math::
        
        \chi_{s,xx} = \chi_{s,yy} = \frac{\chi^+ + \chi^-}{2} \; ,
        
        \chi_{s,xy} = -\chi_{s,yx} = \frac{\rm{i}(\chi^+ - \chi^-)}{2} \; ,
        
        \chi_{s,zz} = -\frac{\omega^2_{ps}}{\omega^2}\; ,
        
    where 
    
    .. math::
        
        \chi^\pm_s = - \frac{\omega^2_{ps}}{\omega(\omega\mp\Omega_s)} \;,
        
        \omega^2_{ps} \equiv \frac{4\pi n_s q_s^2}{m_s} \;.
        
    The components not given above are all zero.
    
    Methods
    =======
    
    __call__(self, coordinates):
        Calculates susceptilibity tensor elements of the particular species 
        at given coordinates.
        
    __str__(self):
        returns a description of the model used.
            
    Attributes
    ==========
    
    _name:
        The name of the model used

    _model:
        The description of the model    
    
    plasma:
        :py:class:`.PlasmaProfile.PlasmaProfile` object
    
    species:
        either 'e' or 'i', denotes electron or ion
        
    species_id:
        if species is 'ion', this number indicates which ion species to use, 
        should be the index in the density list ``plasma.ni0`` and 
        ``plasma.dni``, default to be 0, which means the first kind in 
        ``plasma.ni``.
        
    Initialization
    ==============
    
    :param plasma: Plasma infomation
    :type plasma: :py:class:`.PlasmaProfile.PlasmaProfile`
    :param string species: either 'e' or 'i', stands for electron and ion
    :param int species_id: Optional, default to be 0.
    
    Use
    ===
    
    After initialized, it can be called by giveing a set of frequencies, and a 
    set of spatial coordinates. The number of arrays in coordinates should be 
    equal to the dimension of grid used in ``plasma``. The result is then 
    stored in an array shaped (shape of frequency, shape of spatial coordinates
    , 3, 3)
    
    """
    
    def __init__(self, plasma, species, species_id=0):
        assert isinstance(plasma, PlasmaProfile)        
        assert species in ['e','i']
        if species == 'i':        
            assert isinstance(species_id, int)
            
        self._name = 'Cold Plasma Susceptibility Tensor'
        self._model = 'Cold fluid plasma model. No resonance allowed.'
                
        self.plasma = plasma
        self.species = species
        self.species_id = species_id
        
        
    def __call__(self, omega, coordinates, eq_only=True, time = 0, tol=1e-14):
        """Calculates cold susceptibility tensor at each coordinate given by 
        coordinates.
        
        :param omega: frequencies of the waves under study
        :type omega: array of float with shape (Nf, )
        :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                            for 3D or (Z,R) for 2D
        :type coordinates: list of array_like 
        :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                        be used. Default to be true.
        :type eq_only: bool
        :param int time: time step for perturbation loading
        :param float tol: the tolerance for determining a float is zero, used
                          to check if resonance is happening. 
                          Default to be 1e-14
        
        :return: susceptibility tensor at each point
        :rtype: ndarray of complex, shape (3, 3, frequency_shape,spatial_shape)
        """        
        assert len(coordinates) == self.plasma.grid.dimension
        
        
        omega = np.array(omega)
        coordinates = np.array(coordinates)
        result_shape = []
        result_shape.extend([3,3])
        frequency_shape = list(omega.shape)
        spatial_shape = list(coordinates[0].shape)
        result_shape.extend(frequency_shape)        
        result_shape.extend(spatial_shape)
        result = np.empty(result_shape, dtype='complex')
        
        # entend frequency array's dimension containing spatial dimensions, so 
        # it can be directly broadcasted with plasma profile quantities.
        
        full_shape = frequency_shape
        sp_dim = len(spatial_shape)
        # final dimension equals spatial dimension
        for i in range(sp_dim):
            full_shape.append(1) # add one new dimension
        omega = omega.reshape(full_shape)
        
        if(self.species == 'e'):
            # electron case
            # constants
            c = self.plasma.unit_system['c']
            q = -self.plasma.unit_system['e']
            m = self.plasma.unit_system['m_e']
            pi = np.pi
            
            # profile quantities
            if(eq_only == False):
                n = self.plasma.get_ne(coordinates, False, time=time)
                B = self.plasma.get_B(coordinates, False, time=time)
            else:
                n = self.plasma.get_ne(coordinates, True)
                B = self.plasma.get_B(coordinates, True)
        
        else:
            # ion case
        
            c = self.plasma.unit_system['c']
            q = self.plasma.unit_system['e'] * \
                self.plasma.ions[self.species_id].charge
            m = self.plasma.unit_system['m_p'] * \
                self.plasma.ions[self.species_id].mass
            
            pi = np.pi
            
            # profile quantities
            n = self.plasma.get_ni(coordinates, self.species_id, eq_only) 
            B = self.plasma.get_B(coordinates, eq_only)            
        
        # Now start calculating physical quantities
        
        # first, reshape
# WARNING: the following expressions are only valid for cgs unit.          
# TODO Find a way to convert non-cgs unit input into cgs unit
              
        omega_p2 = 4*pi*n*q*q/m
        Omega_c = q*B/(m*c)
        
        # check if cold cyclotron resonance is happening, if so, raise a 
        # ResonanceError
        if np.any(omega -Omega_c < tol) or np.any(omega +Omega_c < tol):
            raise ResonanceError('omega == Omega_c, susceptibility blow up!\n\
Plasma:{}\nSpecies:{}\nSpeciesID:{}'.format(self.plasma, self.species, 
                                            self.species_id))
        
        chi_plus = - omega_p2 / (omega * (omega - Omega_c))
        chi_minus = - omega_p2 / (omega * (omega + Omega_c))
        
        # construct the tensor
        # xx and yy components
        result[0,0, ...] = result[1, 1, ...] = (chi_plus + chi_minus)/2
        # xy and yx components
        xy = 1j*(chi_plus - chi_minus)/2
        result[0, 1, ... ] = xy
        result[1, 0, ...] = -xy
        # xz, yz, zx, zy components are 0
        result[:, 2, ...] = 0
        result[2, :, ...] = 0
        # zz component
        result[2, 2, ...] = -omega_p2/(omega*omega)
        
        return result
            
        
class SusceptWarm(Susceptilibity):
    r"""Warm plasma susceptibility tensor
    
    Formula
    ======= 
    
    Warm Plasma takes into account the parallel temperature only. So the effect
    only show up in the zz-component. The rest of the tensor is the same as in 
    cold plasma case.    
    
    It has the following form:
    
    .. math::
        
        \chi_{s,xx} = \chi_{s,yy} = \frac{\chi^+ + \chi^-}{2} \; ,
        
        \chi_{s,xy} = -\chi_{s,yx} = \frac{\rm{i}(\chi^+ - \chi^-)}{2} \; ,
        
        \chi_{zz,e} = \frac{\omega^2_{pe}}{-\omega^2 + \gamma k^2_\parallel 
        T_{e,\parallel}/m_e},
        
    where 
    
    .. math::
        
        \chi^\pm_s = - \frac{\omega^2_{ps}}{\omega(\omega\mp\Omega_s)} \;,
        
        \omega^2_{ps} \equiv \frac{4\pi n_s q_s^2}{m_s} \;,
    
    and :math:`\gamma = 3` in adiabatic process, and :math:`\gamma=1` in 
    isothermal process.
    
    The components not given above are all zero.
    
    It is worth noting that now the susceptibility depends on 
    :math:`k_parallel`.
    
    Methods
    =======
    
    :py:method:`__call__`(self, omega, k_para, coordinates):
        Calculates susceptilibity tensor elements of the particular species 
        at given coordinates.
        
    :py:method:`__str__`(self):
        returns a description of the model used.
            
    Attributes
    ==========
    
    _name:
        The name of the model used

    _model:
        The description of the model    
    
    plasma:
        :py:class:`.PlasmaProfile.PlasmaProfile` object
    
    species:
        either 'e' or 'i', denotes electron or ion
        
    species_id:
        if species is 'ion', this number indicates which ion species to use, 
        should be the index in the density list ``plasma.ni0`` and 
        ``plasma.dni``, default to be 0, which means the first kind in 
        ``plasma.ni``.
        
    Initialization
    ==============
    
    :param plasma: Plasma infomation
    :type plasma: :py:class:`.PlasmaProfile.PlasmaProfile`
    :param string species: either 'e' or 'i', stands for electron and ion
    :param int species_id: Optional, default to be 0.
    
    Use
    ===
    
    After initialized, it can be called by giveing a set of frequencies, a set 
    of parallel wave vectors, and a set of spatial coordinates. The number of 
    arrays in coordinates should be equal to the dimension of grid used in 
    ``plasma``. The result is then stored in an array shaped 
    (shape of frequency, shape of wave vector, shape of spatial coordinates, 3,
    3)
    
    """
    
    def __init__(self, plasma, species, species_id=0):
        assert isinstance(plasma, PlasmaProfile)        
        assert species in ['e','i']
        if species == 'i':        
            assert isinstance(species_id, int)
            
        self._name = 'Cold Plasma Susceptibility Tensor'
        self._model = 'Cold fluid plasma model. No resonance allowed.'
                
        self.plasma = plasma
        self.species = species
        self.species_id = species_id
        
        
    def __call__(self, omega, k_para, coordinates, gamma=3, eq_only=True, 
                 time = 0, tol=1e-14):
        """Calculates cold susceptibility tensor at each coordinate given by 
        coordinates.
        
        :param omega: frequencies of the waves under study
        :type omega: array of float with shape (Nf, )
        :param k_para: parallel wave vectors of the waves under study
        :type k_para: array of float with shape (Nk, )
        :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                            for 3D or (Z,R) for 2D
        :type coordinates: list of array_like
        :param float gamma: the parameter for equation of state, default to be
                            3, i.e. adiabatic process for 1 degree of freedom
        :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                        be used.
        :type eq_only: bool
        :param int time: time step for perturbation loading
        :param float tol: the tolerance for determining a float is zero, used
                          to check if resonance is happening. 
                          Default to be 1e-14
        
        :return: susceptibility tensor at each point
        :rtype: ndarray of complex, shape (3,3, frequency_shape, wave_vector_shape, 
                spatial_shape)
        """        
        assert len(coordinates) == self.plasma.grid.dimension
        
        # prepare the result array with the right shape        
        omega = np.array(omega)
        k_para = np.array(k_para)
        coordinates = np.array(coordinates)
        result_shape = []
        frequency_shape = list(omega.shape)
        wave_vector_shape = list(k_para.shape)
        spatial_shape = list(coordinates[0].shape)
        result_shape.extend([3,3])
        result_shape.extend(frequency_shape)
        result_shape.extend(wave_vector_shape)        
        result_shape.extend(spatial_shape)
        result = np.empty(result_shape, dtype='complex')
        
        # entend frequency array's dimension containing wave vector and spatial
        # dimensions, so it can be directly broadcasted with plasma profile 
        # quantities.
        
        f_dim = len(frequency_shape)
        sp_dim = len(spatial_shape)
        wv_dim = len(wave_vector_shape)
        # final dimension equals frequency + wave_vector + spatial dimension
        full_f_shape = frequency_shape
        full_f_shape.extend([1 for i in range(sp_dim + wv_dim)]) 
        omega = omega.reshape(full_f_shape)
        
        # same dimension expansion for wave vector array.         
        
        full_k_shape = []
        full_k_shape.extend([1 for i in range(f_dim)])
        full_k_shape.extend(wave_vector_shape)
        full_k_shape.extend([1 for i in range(sp_dim)])
        k_para = k_para.reshape(full_k_shape)
        
        if(self.species == 'e'):
            # electron case
            # constants
            c = self.plasma.unit_system['c']
            q = -self.plasma.unit_system['e']
            m = self.plasma.unit_system['m_e']
            pi = np.pi
            
            # profile quantities
            if(eq_only == False):
                n = self.plasma.get_ne(coordinates, False, time=time)
                B = self.plasma.get_B(coordinates, False, time=time)
                # need to use parallel Te perturbation here
                T = self.plasma.get_Te(coordinates, eq_only=False, 
                                        perpendicular=False, time=time)
            else:
                n = self.plasma.get_ne(coordinates, True)
                B = self.plasma.get_B(coordinates, True)
                T = self.plasma.get_Te0(coordinates)
        
        else:
            # ion case
            warnings.warn('Warm Susceptibility formula is used for ion species\
, this is usually not appropriate. Check your model to be sure this is what \
you wanted.')
        
            c = self.plasma.unit_system['c']
            q = self.plasma.unit_system['e'] * \
                self.plasma.ions[self.species_id].charge
            m = self.plasma.unit_system['m_p'] * \
                self.plasma.ions[self.species_id].mass
            
            pi = np.pi
            
            # profile quantities
# TODO finish get ion density and temperature methods in PlasmaProfile. 
            if(eq_only == False):
                n = self.plasma.get_ni(coordinates, False, time)
                B = self.plasma.get_B(coordinates, False, time)
                T = self.plasma.get_Ti(coordinates, eq_only=False, 
                                        perpendicular=True, time=time)
            else:
                n = self.plasma.get_ni(coordinates, True)
                B = self.plasma.get_B(coordinates, True)
                T = self.plasma.get_Ti0(coordinates)          
        
        # Now start calculating physical quantities
        
        # first, reshape
# WARNING: the following expressions are only valid for cgs unit.          
# TODO Find a way to convert non-cgs unit input into cgs unit. Modification in
# UnitSystem is needed to cover unit conversion. 
              
        omega_p2 = 4*pi*n*q*q/m
        Omega_c = q*B/(m*c)
        
        # check if cold cyclotron resonance is happening, if so, raise a 
        # ResonanceError
        if np.any(omega -Omega_c < tol) or np.any(omega +Omega_c < tol):
            raise ResonanceError('omega == Omega_c, susceptibility blow up!\n\
Plasma:{}\nSpecies:{}\nSpeciesID:{}'.format(self.plasma, self.species, 
                                            self.species_id))
        
        chi_plus = - omega_p2 / (omega * (omega - Omega_c))
        chi_minus = - omega_p2 / (omega * (omega + Omega_c))
        
        # construct the tensor
        # xx and yy components
        result[0,0, ...] = result[1, 1, ...] = (chi_plus + chi_minus)/2
        # xy and yx components
        xy = 1j*(chi_plus - chi_minus)/2
        result[0, 1, ... ] = xy
        result[1, 0, ...] = -xy
        # xz, yz, zx, zy components are 0
        result[:, 2, ...] = 0
        result[2, :, ...] = 0
        # zz component
        result[2,2, ...] = omega_p2/(-omega*omega + gamma*k_para*k_para*T/m)
        
        return result        
        
                
    
    
    
class Dielectric(object):
    """Abstract base class for Dielectric tensor classes
    
    Methods
    =======
    
    epsilon(self, omega, k_perp, k_para, coordinates): 
        The main evaluation method. Calculates dielectric tensor elements
        at given spatial locations.
        
    chi_e(self, omega, k_perp, k_para, coordinates,):
        Calculates electron susceptibility :math:`\chi_e` at given spatial
        locations.
        
    chi_i(self, omega, k_perp, k_para, coordinates, species_id=0):
        Calculates ions' susceptibility :math:`\chi_i` at given spatial 
        locations. If *n* ion species are given, *plasma* should have
        *ni* entry with the first dimension length *n*. 
    
    info(self):
        print out a description of the plasma and models used.
    
    __str__(self):
        returns a description of the plasma, and models used for electron and 
        ion species.
    
    Attributes
    ==========
    
    _name: The name of the models used
    
    Chi_e_model: :py:class:`Susceptibility` object for electron
    
    Chi_i_model: :py:class:`Susceptibility` object for ions
    
    plasma: :py:class:`PlasmaProfile` object containing plasma information.
    
    """
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def chi_e(self, omega, k_perp, k_para, coordinates):
        pass
    
    @abstractmethod
    def chi_i(self, omega, k_perp, k_para, coordinates, species_id=0):
        pass
    
    @abstractmethod
    def epsilon(self, coordinates):
        raise NotImplemented('Derived Classes of Dielectric must override \
Epsilon method!')

    @abstractmethod
    def info(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractproperty
    def dimension(self):
        pass
    
    @abstractproperty
    def plasma(self):
        pass
    
   
class ColdDielectric(Dielectric):
    r"""Class evaluating cold plasma dielectric tensor
    
    Plasma dielectric tensor , 
    
    Since cold plasma dielectric tensor does not depend on *k*, we can simply
    evaluate it at each spatial location with local plasma parameters only, 
    i.e. ne and B.  
    
    Initialization
    ==============
    :param plasma: plasma profile containing at least ne and B data
    :type plasma: :py:class:`PlasmaProfile` object
    """
    
    def __init__(self, plasma, ion_species=None):
        self._name = 'Cold Plasma Dielectric Tensor'
        self._description = 'Cold electrons and optional cold ions.'
        self._plasma = plasma
        self._Chi_e_model = SusceptCold(plasma,'e')
        if ion_species is not None:
            self.has_ion = True
            self.ion_species = ion_species
            self._Chi_i_model = []
            for s in ion_species:
                self._Chi_i_model.append(SusceptCold(plasma,'i', s))
        else:
            self.has_ion = False
    
    @property
    def dimension(self):
        return self._plasma.grid.dimension
        
    @property
    def plasma(self):
        return self._plasma
    
    def chi_e(self, omega, coordinates=None, eq_only=True, time=None):
        """Calculates electron susceptibility at given locations
        
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param coordinates: *optional*, Cartesian coordinates where 
                            :math:`\chi_e` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
                            
                            If not given, the mesh in *self.plasma* will be 
                            used.
        :type cooridnates: ndarrays of floats with shape ``(ndim, nc1, nc2, ...
                           , ncn)``
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: Chi_e
        :rtype: ndarray of shape ``[nt, nf, nc1, nc2, ..., ncn, 3, 3]``
        """
        if coordinates is None:
            coordinates = self._plasma.grid.get_ndmesh()
        return self._Chi_e_model(omega, coordinates, eq_only, time)
        
    def chi_i(self, omega, coordinates=None, eq_only=True, time=None):
        """Calculates ion susceptibility at given locations
        
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param coordinates: *optional*, Cartesian coordinates where 
                            :math:`\chi_e` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
                            
                            If not given, the mesh in *self.plasma* will be 
                            used.
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: Chi_i
        :rtype: ndarray of shape [nf, nc1, nc2, ..., ncn, 3, 3]
        """
        assert self.has_ion
        
        if coordinates is None:
            coordinates = self._plasma.grid.get_ndmesh()
        result = 0
        for i in range(len(self.ion_species)):
            result += self._Chi_i_model[i](omega,coordinates, eq_only, time)
        return result      
        
    def epsilon(self, omega, coordinates=None, eq_only=True, time=None):
        """Calculates the total dielectric tensor
        
        .. math::
            
            \epsilon = \bf(I) + \sum\limits_s \chi_s
            
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param coordinates: *optional*, Cartesian coordinates where 
                            :math:`\chi_e` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
                            
                            If not given, the mesh in *self.plasma* will be 
                            used.
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: epsilon
        :rtype: ndarray of shape [nf, nc1, nc2, ..., ncn, 3, 3]
        """
        
        
        if coordinates is None:
            coordinates = self._plasma.grid.get_ndmesh()              
        result = self.chi_e(omega, coordinates, eq_only, time)
        if self.has_ion:
            result += self.chi_i(omega, coordinates, eq_only, time)
        
        I = np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
        # I needs to be cast into result dimension so it can be broadcasted to 
        # result
        
        result_dim = result.ndim
        I_shape = [3,3]
        I_shape.extend([1 for i in range(result_dim-2)])
        I = I.reshape(I_shape)                      
        result += I
        
        return result
        
    
    def __str__(self):
        
        info = self._name + '\n'
        info += '    '+self._description+'\n'
        info += str(self.plasma)
        
        return info
        
    def info(self):
        print str(self)
                      
        
                

