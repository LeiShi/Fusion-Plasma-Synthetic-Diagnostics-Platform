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
    :math:`k_\parallel v_{th,\parallel}\gg n\Omega(T_\perp+T_\parallel)/2mc^2`.
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
        \left( \frac{n^2}{\lambda}I_n+2\lambda I_n - 2\lambda I'_n \right)A_n,
         
        \chi_{s,zz} = 
        \left[ \frac{2\omega_p^2}
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
    
    where :math:`\lambda\equiv\frac{k^2_\perp w^2_\perp}{2\Omega^2}` measures 
    the Finite Larmor Radius(FLR) effects, :math:`w_\perp \equiv \sqrt{2T_\perp
    /m}` is the perpendicular thermal speed. 
    :math:`I_n = I_n(\lambda)` is the modified Bessel function of the 
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
        \frac{(\omega - k_\parallel V - n\Omega)T_\perp + n\Omega T_\parallel}
        {\omega T_\parallel} Z_0 \; ,
        
        
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
        a_{pn} N(p+n) \lambda^{p+n-1} \mathcal{F}_{p+n+3/2}
        
        \chi_{xz} = \chi_{zx} = 
        -\frac{\mu \omega^2_p k_\perp k_\parallel c^2}{\omega^3 \omega_c}
        \sum\limits_{N=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} N \lambda^{p+n-1} \mathcal{F}'_{p+n+5/2}
        
        \chi_{yz} = -\chi_{zy} = -\mathrm{i} 
        \frac{\mu \omega^2_p k_\perp k_\parallel c^2}{\omega^3 \omega_c}
        \sum\limits_{N=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} (p+n) \lambda^{p+n-1} \mathcal{F}'_{p+n+5/2}
        
        \chi_{zz} = -\frac{\mu \omega^2_p}{\omega^2}
        \sum\limits_{n=-\infty}^{\infty}\sum\limits_{p=0}^{\infty} a_{pn} 
        \lambda^{p+n} (\mathcal{F}_{p+n+5/2}+ 2\psi^2 \mathcal{F}''_{p+n+7/2} )
        
    where :math:`n\equiv |N|, :math:`\lambda \equiv (k_\perp v_t/\omega_c)^2`,
    `:math:`\mathcal{F}_q = \mathcal{F}_q(\phi, \psi)` is the weakly 
    relativistic plasma dispersion function defined in [5]_, and implemented in
    [4]_. :math:`\psi = k_\parallel c^2/\omega v_t \sqrt{2}`,  
    :math:`\phi^2 = \psi^2 - \mu \delta`, :math:`\mu \equiv c^2/v_t^2`, and 
    :math:`\delta = (\omega - N\omega_c)/\omega`, :math:`v_t \equiv \sqrt{T/m}` 
    , :math:`\omega_c \equiv \left|\frac{qB}{mc}\right|`.The sign of the real
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
 
"""
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from scipy.special import iv, ivp

from ..Maths.PlasmaDispersionFunction import Fq_list, F1q_list, F2q_list
from ..Maths.PlasmaDispersionFunction import Z, a_pn
from .PlasmaProfile import PlasmaProfile
from ..GeneralSettings.UnitSystem import UnitSystem, cgs



class ResonanceError(Exception):
    
    def __init__(self, s):
        self.message = s
        
    def __str__(self):
        return self.message
        
class ModelInvalidError(Exception):
    
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
    
    Initialization
    ==============
    
    :param plasma: Plasma infomation
    :type plasma: :py:class:`.PlasmaProfile.PlasmaProfile`
    :param string species: either 'e' or 'i', stands for electron and ion
    :param int species_id: Optional, default to be 0.    
    
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
        
        
    def __call__(self, coordinates, omega, k_para=None, k_perp=None, 
                 eq_only=True, time = 0, tol=1e-14):
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
        
        
        # prepare the result array with the right shape        
        omega = np.array(omega)
        k_para = np.array(k_para)
        k_perp = np.array(k_perp)
        coordinates = np.array(coordinates)
        result_shape = []
        frequency_shape = list(omega.shape)
        wave_vector_para_shape = list(k_para.shape)
        wave_vector_perp_shape = list(k_perp.shape)
        spatial_shape = list(coordinates[0].shape)
        result_shape.extend([3,3])
        result_shape.extend(frequency_shape)
        result_shape.extend(wave_vector_para_shape)
        result_shape.extend(wave_vector_perp_shape)        
        result_shape.extend(spatial_shape)
        result = np.empty(result_shape, dtype='complex')
        
        # entend frequency array's dimension containing wave vector and spatial
        # dimensions, so it can be directly broadcasted with plasma profile 
        # quantities.
        
        sp_dim = len(spatial_shape)
        wv_para_dim = len(wave_vector_para_shape)
        wv_perp_dim = len(wave_vector_perp_shape)
        # final dimension equals frequency + wave_vector + spatial dimension
        full_f_shape = frequency_shape
        full_f_shape.extend([1 for i in range(sp_dim + wv_para_dim + \
                                              wv_perp_dim)]) 
        omega = omega.reshape(full_f_shape)
        
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
    
    Initialization
    ==============
    
    :param plasma: Plasma infomation
    :type plasma: :py:class:`.PlasmaProfile.PlasmaProfile`
    :param string species: either 'e' or 'i', stands for electron and ion
    :param int species_id: Optional, default to be 0.
    :param float gamma: Optional, default to be 3
    
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
    
    gamma:
        The ratio of specific heat for chosen process. gamma = 3 for 1D 
        adiabatic process, and gamma = 1 for isothermal.
        
    
    
    Use
    ===
    
    After initialized, it can be called by giveing a set of frequencies, a set 
    of parallel wave vectors, and a set of spatial coordinates. The number of 
    arrays in coordinates should be equal to the dimension of grid used in 
    ``plasma``. The result is then stored in an array shaped 
    (shape of frequency, shape of wave vector, shape of spatial coordinates, 3,
    3)
    
    Calling Signiture
    =================
    
    __call__(self, coordinates, omega, k_para, k_perp=None, eq_only=True, 
             time = 0, tol=1e-14):
    
    Calculates warm susceptibility tensor at each coordinate given by 
    coordinates.
    
    :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                        for 3D or (Z,R) for 2D
    :type coordinates: list of array_like
    :param float gamma: the parameter for equation of state, default to be
                        3, i.e. adiabatic process for 1 degree of freedom
    :param omega: frequencies of the waves under study
    :type omega: array of float with shape (Nf, )
    :param k_para: parallel wave vectors of the waves under study
    :type k_para: array of float with shape (Nk_para, )
    :param k_perp: perpendicular wave vectors. NOT USED IN WARM FORMULA,
                   default to be None.
    :type k_perp: None or array of float with shape (Nk_perp, )        
    :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                    be used.
    :type eq_only: bool
    :param int time: time step for perturbation loading
    :param float tol: the tolerance for determining a float is zero, used
                      to check if resonance is happening. 
                      Default to be 1e-14
    
    :return: susceptibility tensor at each point
    :rtype: ndarray of complex, shape (3,3, Nf, Nk_para, Nk_perp 
            spatial_shape)
    
    """
    
    def __init__(self, plasma, species, species_id=0, gamma=3):
        assert isinstance(plasma, PlasmaProfile)        
        assert species in ['e','i']
        if species == 'i':        
            assert isinstance(species_id, int)
            
        self._name = 'Warm Plasma Susceptibility Tensor'
        self._model = 'Fluid plasma model with finite parallel temperature. \
No resonance allowed.'
                
        self.plasma = plasma
        self.species = species
        self.species_id = species_id
        self.gamma = gamma
        
        
    def __call__(self, coordinates, omega, k_para, k_perp=None, 
                 eq_only=True, time = 0, tol=1e-14):
        """Calculates warm susceptibility tensor at each coordinate given by 
        coordinates.
        
        :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                            for 3D or (Z,R) for 2D
        :type coordinates: list of array_like
        :param float gamma: the parameter for equation of state, default to be
                            3, i.e. adiabatic process for 1 degree of freedom
        :param omega: frequencies of the waves under study
        :type omega: array of float with shape (Nf, )
        :param k_para: parallel wave vectors of the waves under study
        :type k_para: array of float with shape (Nk_para, )
        :param k_perp: perpendicular wave vectors. NOT USED IN WARM FORMULA,
                       default to be None.
        :type k_perp: None or array of float with shape (Nk_perp, )
        :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                        be used.
        :type eq_only: bool
        :param int time: time step for perturbation loading
        :param float tol: the tolerance for determining a float is zero, used
                          to check if resonance is happening. 
                          Default to be 1e-14
        
        :return: susceptibility tensor at each point
        :rtype: ndarray of complex, shape (3,3, Nf, Nk_para, Nk_perp 
                spatial_shape)
        """        
        assert len(coordinates) == self.plasma.grid.dimension
        
        # prepare the result array with the right shape        
        omega = np.array(omega)
        k_para = np.array(k_para)
        k_perp = np.array(k_perp)
        coordinates = np.array(coordinates)
        result_shape = []
        frequency_shape = list(omega.shape)
        wave_vector_para_shape = list(k_para.shape)
        wave_vector_perp_shape = list(k_perp.shape)
        spatial_shape = list(coordinates[0].shape)
        result_shape.extend([3,3])
        result_shape.extend(frequency_shape)
        result_shape.extend(wave_vector_para_shape)
        result_shape.extend(wave_vector_perp_shape)        
        result_shape.extend(spatial_shape)
        result = np.empty(result_shape, dtype='complex')
        
        # entend frequency array's dimension containing wave vector and spatial
        # dimensions, so it can be directly broadcasted with plasma profile 
        # quantities.
        
        f_dim = len(frequency_shape)
        sp_dim = len(spatial_shape)
        wv_para_dim = len(wave_vector_para_shape)
        wv_perp_dim = len(wave_vector_perp_shape)
        # final dimension equals frequency + wave_vector + spatial dimension
        full_f_shape = frequency_shape
        full_f_shape.extend([1 for i in range(sp_dim + wv_para_dim + \
                                              wv_perp_dim)]) 
        omega = omega.reshape(full_f_shape)
        
        # same dimension expansion for wave vector array.         
        
        full_k_para_shape = []
        full_k_para_shape.extend([1 for i in range(f_dim)])
        full_k_para_shape.extend(wave_vector_para_shape)
        full_k_para_shape.extend([1 for i in range(wv_perp_dim)])
        full_k_para_shape.extend([1 for i in range(sp_dim)])
        k_para = k_para.reshape(full_k_para_shape)
        
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
        gamma = self.gamma
        
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
        

class SusceptNonrelativistic(Susceptilibity):
    r""" Susceptibility tensor using non-relativistic kinetic formula
    
    Initialization
    ==============
    
    :param plasma: Plasma infomation
    :type plasma: :py:class:`.PlasmaProfile.PlasmaProfile`
    :param string species: either 'e' or 'i', stands for electron and ion
    :param int species_id: Optional, default to be 0.
    :param int max_harmonic: Optional, default to be 4. The highest order 
                             cyclotron harmonic contribution to be included. It
                             is refered as :math:`n` in the formula below.
    
    Formula
    ========
    
    Non-relativistic hot plasma susceptibility tensor is obtained through 
    solving Vlasov-Maxwell Equations. [1]_ 
    
    This limit is valid when parallel Doppler shift effect is dominant over 
    cyclotron frequency shift due to relativistic mass increasement, i.e.
    :math:`k_\parallel v_{th,\parallel}\gg n\Omega(T_\perp+T_\parallel)/2mc^2`.
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
        \left( \frac{n^2}{\lambda}I_n+2\lambda I_n - 2\lambda I'_n \right)A_n,
         
        \chi_{s,zz} = 
        \left[ \frac{2\omega_p^2}
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
    
    where :math:`\lambda\equiv\frac{k^2_\perp w^2_\perp}{2\Omega^2}` measures 
    the Finite Larmor Radius(FLR) effects, :math:`w_\perp \equiv \sqrt{2T_\perp
    /m}` is the perpendicular thermal speed. 
    :math:`I_n = I_n(\lambda)` is the modified Bessel function of the 
    first kind, with argument :math:`\lambda`, and :math:`I'_n = (\mathrm{d}/
    \mathrm{d} \lambda)I_n(\lambda)`. :math:`V` is the parallel equilibrium 
    flow velocity. 
    
    :math:`A_n` and :math:`B_n` are given as

    .. math::
    
        A_n = \frac{T_\perp - T_\parallel}{\omega T_\parallel} + 
        \frac{1}{k_\parallel w_\parallel}
        \frac{(\omega-k_\parallel V -n\Omega)T_\perp + n\Omega T_\parallel}
        {\omega T_\parallel}  Z_0 \;,
        
        B_n = \frac{1}{k_\parallel} 
        \frac{(\omega-n\Omega)T_\perp - (k_\parallel V - n\Omega)T_\parallel}
        {\omega T_\parallel}
        +\frac{1}{k_\parallel}\frac{\omega-n\Omega}{k_\parallel w_\parallel}
        \frac{(\omega - k_\parallel V - n\Omega)T_\perp + n\Omega T_\parallel}
        {\omega T_\parallel} Z_0 \; ,
        
        
    where :math:`Z_0 = Z_0(\zeta_n)` is the Plasma Dispersion Function [2]_, 
    and :math:`\zeta_n =(\omega -k_\parallel V -n\Omega)/{k_\parallel 
    w_\parallel}` measures the distance from the n-th order cold resonance.
    
    Calling Signiture
    =================
    
    __call__(self, coordinates, omega, k_para, k_perp, eq_only=True, 
             time = 0, tol=1e-14):
    
    Calculates non-relativistic susceptibility tensor at each coordinate given 
    by coordinates.
    
    :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                        for 3D or (Z,R) for 2D
    :type coordinates: list of array_like
    :param omega: frequencies of the waves under study
    :type omega: array of float with shape (Nf, )
    :param k_para: parallel wave vectors of the waves under study
    :type k_para: array of float with shape (Nk_para, )
    :param k_perp: perpendicular wave vectors of the waves under study
    :type k_perp: array of float with shape (Nk_perp, )
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
            
    :raise ModelInvalidError: if :math:`k_\parallel v_{th,\parallel}` is 
                              too small. 
            
    References
    ==========
    
    .. [1] "Waves in Plasmas", Chapter 10-7, T.H.Stix, 1992, American Inst. of 
           Physics 
    
    .. [2] :py:module:`..Maths.PlasmaDispersionFunction`
               
    """                
    
    def __init__(self, plasma, species, species_id=0, max_harmonic=4):
        assert isinstance(plasma, PlasmaProfile)        
        assert species in ['e','i']
        if species == 'i':        
            assert isinstance(species_id, int)
            
        self._name = 'Non-relativistic Plasma Susceptibility Tensor'
        self._model = 'Kinetic plasma model with finite temperature, but in \
non-relativistic limit. Resonance allowed. Max_harmonic = {}'.format(\
                                                                  max_harmonic)
                
        self.plasma = plasma
        self.species = species
        self.species_id = species_id
        self.max_harmonic = max_harmonic
        
    def __call__(self, coordinates, omega, k_para, k_perp=None, 
                 eq_only=True, time = 0, tol=1e-14):
        """Calculates non-relativistic susceptibility tensor at each coordinate
        given by coordinates.
        
        :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                            for 3D or (Z,R) for 2D
        :type coordinates: list of array_like
        :param float gamma: the parameter for equation of state, default to be
                            3, i.e. adiabatic process for 1 degree of freedom
        :param omega: frequencies of the waves under study
        :type omega: array of float with shape (Nf, )
        :param k_para: parallel wave vectors of the waves under study
        :type k_para: array of float with shape (Nk_para, )
        :param k_perp: perpendicular wave vectors. NOT USED IN WARM FORMULA,
                       default to be None.
        :type k_perp: None or array of float with shape (Nk_perp, )
        
        
        :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                        be used.
        :type eq_only: bool
        :param int time: time step for perturbation loading
        :param float tol: the tolerance for determining a float is zero, used
                          to check if resonance is happening. 
                          Default to be 1e-14
        
        :return: susceptibility tensor at each point
        :rtype: ndarray of complex, shape (3,3, Nf, Nk_para, Nk_perp 
                spatial_shape)
                
        :raise ModelInvalidError: if :math:`k_\parallel v_{th,\parallel}` is 
                                  too small. 
        """        
        assert len(coordinates) == self.plasma.grid.dimension
        
        # prepare the result array with the right shape        
        omega = np.array(omega)
        k_para = np.array(k_para)
        k_perp = np.array(k_perp)
        coordinates = np.array(coordinates)
        result_shape = []
        frequency_shape = list(omega.shape)
        wave_vector_para_shape = list(k_para.shape)
        wave_vector_perp_shape = list(k_perp.shape)
        spatial_shape = list(coordinates[0].shape)
        result_shape.extend([3,3])
        result_shape.extend(frequency_shape)
        result_shape.extend(wave_vector_para_shape)
        result_shape.extend(wave_vector_perp_shape)        
        result_shape.extend(spatial_shape)
        result = np.zeros(result_shape, dtype='complex')
        
        # entend frequency array's dimension containing wave vector and spatial
        # dimensions, so it can be directly broadcasted with plasma profile 
        # quantities.
        
        f_dim = len(frequency_shape)
        sp_dim = len(spatial_shape)
        wv_para_dim = len(wave_vector_para_shape)
        wv_perp_dim = len(wave_vector_perp_shape)
        # final dimension equals frequency + wave_vector + spatial dimension
        full_f_shape = frequency_shape
        full_f_shape.extend([1 for i in range(sp_dim + wv_para_dim + \
                                              wv_perp_dim)]) 
        omega = omega.reshape(full_f_shape)
        
        # same dimension expansion for wave vector array.         
        
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
        
        # now we calculate the tensor
        # first, get all particle quantities
        
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
                T_para = self.plasma.get_Te(coordinates, eq_only=False, 
                                            perpendicular=False, time=time)
                T_perp = self.plasma.get_Te(coordinates, eq_only=False,
                                            perpendicular=True, time=time)
                try:
                    V = self.plasma.get_Ve(coordinates, eq_only=False,
                                           time=time)
                except AttributeError:
                    V = 0
            else:
                n = self.plasma.get_ne(coordinates, True)
                B = self.plasma.get_B(coordinates, True)
                T_para = self.plasma.get_Te0(coordinates)
                T_perp = T_para
                try:
                    V = self.plasma.get_Ve(coordinates, eq_only=True)
                except AttributeError:
                    V = 0
        
        else:
            # ion case
            warnings.warn('Hot non-relativistic Susceptibility formula is \
used for ion species, this is usually not appropriate. Check your model to be \
sure this is what you wanted.')
        
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
                T_para = self.plasma.get_Ti(coordinates, eq_only=False, 
                                            perpendicular=False, time=time)
                T_perp = self.plams.get_Ti(coordinates, eq_only=False,
                                           perpendicular=True, time=time)
                try:
                    V = self.plasma.get_Vi(coordinates, eq_only=False,
                                           time=time)
                except AttributeError:
                    V = 0
            else:
                n = self.plasma.get_ni(coordinates, True)
                B = self.plasma.get_B(coordinates, True)
                T_para = self.plasma.get_Ti0(coordinates)
                T_perp = T_para
                try:
                    V = self.plasma.get_Vi(coordinates, eq_only=True)
                except AttributeError:
                    V = 0
                
        # Calculate physical quantities used in the formula
        
        w_perp2 = 2*T_perp/m
        w_para2 = 2*T_para/m
        w_para = np.sqrt(w_para2)
        
        res_width = k_para*w_para
        # check if k_para*w_para is too small for non-relativistic model to be
        # good, also eliminates the potential zero denominator problems
    
        if np.any(res_width < omega*(T_para+T_perp)/(2*m*c*c)) or \
        np.any(res_width < tol):
            raise  ModelInvalidError('k_para*w_para is too small. \
Non-relativistic model may not be valid. Try relativistic models instead.')    
        
        Omega = q*B/(m*c)
        lambd = k_perp*k_perp * w_perp2 /(2*Omega*Omega)
        
        # Now, calculate susceptibility tensor elements order by order, note 
        # that we treat positive and negative n together as the same order.
        
        # We also leave the constant coefficients outside the summation, and 
        # multiply them afterwards
        for i in range(self.max_harmonic+1):
            # note that I_n = I_-n, so no need to calculate -n terms           
            I = iv(i,lambd)
            I_p = ivp(i,lambd, 1) 
            
            # first calculate positive i part
            res = (omega - k_para*V - i*Omega)
                
            zeta = res / res_width            
            Ai = ((T_perp - T_para) + 1/res_width * (res*T_perp +\
                       i*Omega*T_para) * Z(zeta)) / (omega*T_para)
                       
            Bi = ((omega-i*Omega)*T_perp - (k_para*V - i*Omega)*T_para +\
                1/res_width*(omega-i*Omega)*(res*T_perp + i*Omega*T_para)\
                *Z(zeta)) / (k_para * omega * T_para)
                
            result[0,0] += i*i*I*Ai
            result[1,1] += ((i*i/lambd + 2*lambd)*I - 2*lambd*I_p)*Ai
            result[2,2] += 2*(omega-i*Omega)*I*Bi
            result[0,1] += -1j*i*(I-I_p)*Ai
            result[0,2] += i*I*Bi
            result[1,2] += 1j*(I-I_p)*Bi
            
            if (i != 0):
                # now, negative i part
                i = -i
            
                res = (omega - k_para*V - i*Omega)
                    
                zeta = res / res_width            
                Ai = ((T_perp - T_para) + 1/res_width * (res*T_perp +\
                           i*Omega*T_para) * Z(zeta)) / (omega*T_para)
                           
                Bi = ((omega-i*Omega)*T_perp - (k_para*V - i*Omega)*T_para +\
                    1/res_width*(omega-i*Omega)*(res*T_perp + i*Omega*T_para)\
                    *Z(zeta)) / (k_para * omega * T_para)
                    
                result[0,0] += i*i*I*Ai
                result[1,1] += ((i*i/lambd + 2*lambd)*I - 2*lambd*I_p)*Ai
                result[2,2] += 2*(omega-i*Omega)*I*Bi
                result[0,1] += -1j*i*(I-I_p)*Ai
                result[0,2] += i*I*Bi
                result[1,2] += 1j*(I-I_p)*Bi
        
        # now, multiply with each common factors            
        result[0,0] *= 1/lambd
        result[2,2] *= 1/(k_para*w_perp2)
        result[0,2] *= k_perp/(Omega*lambd)
        result[1,2] *= k_perp/lambd
        
        # multiply with the all common factor omega_pe^2/omega * exp(-lambd)
        result *= 4*pi*n*q*q/(m*omega) * np.exp(-lambd)
        
        # finally, add the flow term in zz component
        result[2,2] += 8*pi*n*q*q/(m*omega*k_para*w_perp2) * V
        
        return result
      
      
class SusceptRelativistic(Susceptilibity):
    r""" Susceptibility tensor using weakly-relativistic kinetic formula
    
    Initialization
    ==============
    
    :param plasma: Plasma infomation
    :type plasma: :py:class:`.PlasmaProfile.PlasmaProfile`
    :param string species: either 'e' or 'i', stands for electron and ion
    :param int species_id: Optional, default to be 0.
    :param int max_harmonic: Optional, default to be 4. The highest order 
                             cyclotron harmonic contribution to be included. It
                             is refered as :math:`n` in the formula below.
    :param int max_power: Optional, default to be 4. The highest order of 
                          :math:`\lambda` kept in formula. It is related to the
                          :math:`p+n` term in the formula below.
    
    Formula
    ========
    
    Weakly relativistic plasma susceptibility tensor is obtained through 
    solving Vlasov-Maxwell system, similarly to the Non-relativistic case. 
    However, in this case, we retain the relativistic effect of the mass 
    increasement. A detailed discussion of the validity of this limit can be 
    found in [2]_ and [3]_. 
    
    Here, we use the expressions given in [2]_, and the plasma is assumed 
    isotropic. (The notations are kept the same as in [2]_ so it's easy to 
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
        a_{pn} N(p+n) \lambda^{p+n-1} \mathcal{F}_{p+n+3/2}
        
        \chi_{xz} = \chi_{zx} = 
        -\frac{\mu \omega^2_p k_\perp k_\parallel c^2}{\omega^3 \omega_c}
        \sum\limits_{N=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} N \lambda^{p+n-1} \mathcal{F}'_{p+n+5/2}
        
        \chi_{yz} = -\chi_{zy} = -\mathrm{i} 
        \frac{\mu \omega^2_p k_\perp k_\parallel c^2}{\omega^3 \omega_c}
        \sum\limits_{N=-\infty}^{\infty}\sum\limits_{p=0}^{\infty}
        a_{pn} (p+n) \lambda^{p+n-1} \mathcal{F}'_{p+n+5/2}
        
        \chi_{zz} = -\frac{\mu \omega^2_p}{\omega^2}
        \sum\limits_{n=-\infty}^{\infty}\sum\limits_{p=0}^{\infty} a_{pn} 
        \lambda^{p+n} (\mathcal{F}_{p+n+5/2}+ 2\psi^2 \mathcal{F}''_{p+n+7/2} )
        
    where :math:`n \equiv |N|`, :math:`\lambda \equiv (k_\perp v_t/\omega_c)^2`
    , :math:`\mathcal{F}_q = \mathcal{F}_q(\phi, \psi)` is the weakly 
    relativistic plasma dispersion function defined in [2]_, and implemented in
    [1]_. :math:`\psi = k_\parallel c^2/\omega v_t \sqrt{2}`,  
    :math:`\phi^2 = \psi^2 - \mu \delta`, :math:`\mu \equiv c^2/v_t^2`, and 
    :math:`\delta = (\omega - N\omega_c)/\omega`, :math:`v_t \equiv \sqrt{T/m}`
    , :math:`\omega_c \equiv \left|\frac{qB}{mc}\right|`. The sign of the real
    (imaginary) part of :math:`\phi` is defined to be +(-) [4]_. 
    
    Calling Signiture
    =================
    
    __call__(self, coordinates, omega, k_para, k_perp, eq_only=True, 
             time = 0, tol=1e-14):
    
    Calculates relativistic susceptibility tensor at each coordinate given by 
    coordinates.
    
    :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                        for 3D or (Z,R) for 2D or (X,) for 1D
    :type coordinates: list of array_like
    :param omega: frequencies of the waves under study
    :type omega: array of float with shape (Nf, )
    :param k_para: parallel wave vectors of the waves under study
    :type k_para: array of float with shape (Nk_para, )
    :param k_perp: perpendicular wave vectors of the waves under study
    :type k_perp: array of float with shape (Nk_perp, )
    :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                    be used.
    :type eq_only: bool
    :param int time: time step for perturbation loading
    :param float tol: the tolerance for determining a float is zero, used
                      to check if resonance is happening. 
                      Default to be 1e-14
    
    :return: susceptibility tensor at each point
    :rtype: ndarray of complex, shape (3,3, Nf, Nk_para, Nk_perp,spatial_shape)
            
    References
    ==========
    
    .. [1] :py:module:`..Maths.PlasmaDispersionFunction`  

    .. [2] I.P.Shkarofsky, "New representations of dielectric tensor elements 
           in magnetized plasma", J. Plasma Physics(1986), vol. 35, part 2, pp. 
           319-331 
           
    .. [3] M.Bornatici, R. Cano, et al., "Electron cyclotron emission and 
           absorption in fusion plasmas", Nucl. Fusion(1983), vol.23, No.9, pp. 
           1153
           
    .. [4] Weakly relativistic dielectric tensor and dispersion functions of a 
           Maxwellian plasma, V. Krivenski and A. Orefice, J. Plasma Physics 
           (1983), vol. 30, part 1, pp. 125-131 
               
    """
    
    def __init__(self, plasma, species, species_id=0, max_harmonic=4, 
                 max_power=4):
        assert isinstance(plasma, PlasmaProfile)        
        assert species in ['e','i']
        if species == 'i':        
            assert isinstance(species_id, int)
            
        self._name = 'Weakly-relativistic Plasma Susceptibility Tensor'
        self._model = 'Kinetic plasma model with finite temperature, in \
weakly-relativistic limit. Resonance allowed. Max_harmonic = {}'.format(\
                                                                  max_harmonic)
                
        self.plasma = plasma
        self.species = species
        self.species_id = species_id
        self.max_harmonic = max_harmonic
        self.max_power = max_power
        
        # self.max_harmonic has higher priority, if self.max_power is not high
        # enough to have max_harmonic contribution, it will be set to 
        # max_harmonic.
        
        if(self.max_power < self.max_harmonic):
            self.max_power = self.max_harmonic
            
        
    def __call__(self, coordinates, omega, k_para, k_perp, 
                 eq_only=True, time = 0, tol=1e-14):
        r"""Calculates weakly-relativistic susceptibility tensor at each 
        coordinate given by coordinates.
        
        :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                            for 3D or (Z,R) for 2D or (X,) for 1D
        :type coordinates: list of array_like
        :param float gamma: the parameter for equation of state, default to be
                            3, i.e. adiabatic process for 1 degree of freedom
        :param omega: frequencies of the waves under study
        :type omega: array of float with shape (Nf, )
        :param k_para: parallel wave vectors of the waves under study
        :type k_para: array of float with shape (Nk_para, )
        :param k_perp: perpendicular wave vectors.
        :type k_perp: None or array of float with shape (Nk_perp, )
        
        
        :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                        be used.
        :type eq_only: bool
        :param int time: time step for perturbation loading
        :param float tol: the tolerance for determining a float is zero, used
                          to check if resonance is happening. 
                          Default to be 1e-14
        
        :return: susceptibility tensor at each point
        :rtype: ndarray of complex, shape (3,3, Nf, Nk_para, Nk_perp 
                spatial_shape)
                
         
        """        
        assert len(coordinates) == self.plasma.grid.dimension
        
        # prepare the result array with the right shape        
        omega = np.array(omega)
        k_para = np.array(k_para)
        k_perp = np.array(k_perp)
        coordinates = np.array(coordinates)
        result_shape = []
        frequency_shape = list(omega.shape)
        wave_vector_para_shape = list(k_para.shape)
        wave_vector_perp_shape = list(k_perp.shape)
        spatial_shape = list(coordinates[0].shape)
        result_shape.extend([3,3])
        result_shape.extend(frequency_shape)
        result_shape.extend(wave_vector_para_shape)
        result_shape.extend(wave_vector_perp_shape)        
        result_shape.extend(spatial_shape)
        result = np.zeros(result_shape, dtype='complex')
        
        # entend frequency array's dimension containing wave vector and spatial
        # dimensions, so it can be directly broadcasted with plasma profile 
        # quantities.
        
        f_dim = len(frequency_shape)
        sp_dim = len(spatial_shape)
        wv_para_dim = len(wave_vector_para_shape)
        wv_perp_dim = len(wave_vector_perp_shape)
        # final dimension equals frequency + wave_vector + spatial dimension
        full_f_shape = frequency_shape
        full_f_shape.extend([1 for i in range(sp_dim + wv_para_dim + \
                                              wv_perp_dim)]) 
        omega = omega.reshape(full_f_shape)
        
        # same dimension expansion for wave vector array.         
        
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
        
        # now we calculate the tensor
        # first, get all particle quantities
        
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
                                            perpendicular=True, time=time)
                
            else:
                n = self.plasma.get_ne(coordinates, True)
                B = self.plasma.get_B(coordinates, True)
                T= self.plasma.get_Te0(coordinates)
        
        else:
            # ion case
            warnings.warn('Hot non-relativistic Susceptibility formula is \
used for ion species, this is usually not appropriate. Check your model to be \
sure this is what you wanted.')
        
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
                T = self.plams.get_Ti(coordinates, eq_only=False,
                                           perpendicular=True, time=time)
                
            else:
                n = self.plasma.get_ni(coordinates, True)
                B = self.plasma.get_B(coordinates, True)
                T = self.plasma.get_Ti0(coordinates)
                
        # Now we calculate the tensor elements
                
        # First, calculate some useful quantities
        vt = np.sqrt(T/m)
        c2 = c * c
        mu = c2*m/T
        omega_c = np.abs(q)*B/(m*c)
        lambd = k_perp * k_perp * T / (m* omega_c * omega_c)
        psi = k_para*c2/(omega*vt*np.sqrt(2))
        

        # Now we calculate harmonic by harmonic
        for i in range(self.max_harmonic+1):
            i_mod = i
            delta = (omega - i*omega_c)/omega
            phi = np.lib.scimath.sqrt(psi*psi - mu*delta)
            
            p_max = self.max_power - i_mod + 1
            
            # We relabel p'=p+1 in zz component formula, thus it sums over p'
            # starting from 1, but lambda power and F functions have similar
            # order as other elements
            
            # So, we first add p=0 terms for other elements
            
            # two F functions are used more than once
                        
            lambd_pn1 = lambd**(i_mod-1)
            
            if (i_mod != 0): 
                a0n = a_pn(0,i_mod)
                Fn32 = Fq_list[2*i_mod+3](phi,psi)
                Fpn52 = F1q_list[2*i_mod+5](phi, psi)
                result[0,0] += a0n*i*i*lambd_pn1*Fn32
                result[1,1] += a0n*(i*i)*lambd_pn1*Fn32
                result[0,1] += a0n*i*i_mod*lambd_pn1*Fn32
                result[0,2] += a0n*i*lambd_pn1*Fpn52
                result[1,2] += a0n*i_mod*lambd_pn1*Fpn52
                
            # Now sum over p, starting from p=1 up to p=p_max
            for p in range(1, p_max+1):
                Fn32 = Fq_list[2*(i_mod+p)+3](phi,psi)
                Fpn52 = F1q_list[2*(i_mod+p)+5](phi, psi)
                apn = a_pn(p,i_mod)
                lambd_pn1 *= lambd
                result[0,0] += apn * i*i * lambd_pn1 * Fn32
                result[1,1] += apn*( (p+i_mod)**2 - p*(p+2*i_mod)/ \
                                     (2*(i_mod+p)-1) ) * lambd_pn1 * Fn32
                result[0,1] += apn * i*(p+i_mod) * lambd_pn1 * Fn32
                result[0,2] += apn*i*lambd_pn1*Fpn52
                result[1,2] += apn*(p+i_mod)*lambd_pn1*Fpn52
                result[2,2] += a_pn(p-1,i_mod) * lambd_pn1 * \
                       (Fn32 + 2*psi*psi*F2q_list[2*(i_mod+p)+5](phi,psi))
                               
            if (i != 0):
                # i>0 case, we need to add -i terms as well
                i = -i
                # the rest will be exactly the same as before
                delta = (omega - i*omega_c)/omega
                phi = np.lib.scimath.sqrt(psi*psi - mu*delta)
                
                p_max = self.max_power - i_mod + 1
                
            # We relabel p'=p+1 in zz component formula, thus it sums over p'
            # starting from 1, but lambda power and F functions have similar
            # order as other elements
                
                # So, we first add p=0 terms for other elements
                
                a0n = a_pn(0,i_mod)
                Fn32 = Fq_list[2*i_mod+3](phi,psi)
                Fpn52 = F1q_list[2*i_mod+5](phi, psi)
                result[0,0] += a0n*i*i*lambd_pn1*Fn32
                result[1,1] += a0n*(i*i)*lambd_pn1*Fn32
                result[0,1] += a0n*i*i_mod*lambd_pn1*Fn32
                result[0,2] += a0n*i*lambd_pn1*Fpn52
                result[1,2] += a0n*i_mod*lambd_pn1*Fpn52
                
            # Now sum over p, starting from p=1 up to p=p_max
                for p in range(1, p_max+1):
                    Fn32 = Fq_list[2*(i_mod+p)+3](phi,psi)
                    Fpn52 = F1q_list[2*(i_mod+p)+5](phi, psi)
                    apn = a_pn(p,i_mod)
                    lambd_pn1 *= lambd
                    result[0,0] += apn * i*i * lambd_pn1 * Fn32
                    result[1,1] += apn*( (p+i_mod)**2 - p*(p+2*i_mod)/ \
                                         (2*(i_mod+p)-1) ) * lambd_pn1 * Fn32
                    result[0,1] += apn * i*(p+i_mod) * lambd_pn1 * Fn32
                    result[0,2] += apn*i*lambd_pn1*Fpn52
                    result[1,2] += apn*(p+i_mod)*lambd_pn1*Fpn52
                    result[2,2] += a_pn(p-1,i_mod) * lambd_pn1 * \
                       (Fn32 + 2*psi*psi*F2q_list[2*(i_mod+p)+5](phi,psi))
                                
        # Now multiply the coeffecients in front of the summation
        mu_omegap2_over_omega2 = mu*4*pi*n*q*q/(m*omega*omega)
        
        result[0,0] *= -mu_omegap2_over_omega2
        result[1,1] *= -mu_omegap2_over_omega2
        result[0,1] *= 1j*mu_omegap2_over_omega2
        result[0,2] *= -mu_omegap2_over_omega2 * k_perp * k_para *c2 / \
                       (omega*omega_c)
        result[1,2] *= -1j*mu_omegap2_over_omega2 * k_perp * k_para *c2 / \
                       (omega*omega_c)
        result[2,2] *= -mu_omegap2_over_omega2
        
        # Fill up the other off-diagonal components
        result[1,0] = -result[0,1]
        result[2,0] = result[0,2]
        result[2,1] = -result[1,2]
        
        return result
            
            
            
            
            
        

                   
###############################################################################
#                    Divider for Dielectric Classes                           #
###############################################################################


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
    
    _Chi_e_model: :py:class:`Susceptibility` object for electron
    
    _Chi_i_model: :py:class:`Susceptibility` object for ions
    
    dimension: int, the dimension of configuration space
    
    plasma: :py:class:`PlasmaProfile` object containing plasma information.
    
    """
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def chi_e(self, coordinates, omega, k_para, k_perp, eq_only=True, time=0):
        pass
    
    @abstractmethod
    def chi_i(self, coordinates, omega, k_para, k_perp, eq_only=True, time=0, 
              species_id=None):
        pass
    
    @abstractmethod
    def epsilon(self, coordinates, omega, k_para, k_perp, eq_only=True,time=0):
        raise NotImplemented('Derived Classes of Dielectric must override \
Epsilon method!')

   
    def info(self):
        print str(self)
    
    
    def __str__(self):
        info = self._name + '\n'
        info += '    '+self._description+'\n'
        info += str(self.plasma)
        
        return info
    
    @property
    def dimension(self):
        return self._plasma.grid.dimension
        
    @property
    def plasma(self):
        return self._plasma
    
class ColdDielectric(Dielectric):
    r""" Base class for dielectric tensors consists of all cold fluids
    
    Provide methods::
    
        chi_e(self, coordinates, omega, k_para=None, k_perp=None, eq_only=True,
              time=0)
              
        chi_i(self, coordinates, omega, k_para=None, k_perp=None, eq_only=True,
              time=0, species_id=None)
              
        epsilon(self, coordinates, omega, k_para=None, k_perp=None,
                eq_only=True, time=None)
    """
    
    __metaclass__ = ABCMeta
    
    def chi_e(self, coordinates, omega, k_para=None, k_perp=None, eq_only=True,
              time=0):
        """Calculates electron susceptibility at given locations
        
        :param coordinates: Cartesian coordinates where :math:`\chi_e` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape ``(ndim, nc1, nc2, ...
                           , ncn)``
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: optional, parallel wave vectors. NOT USED IN COLD MODEL.
        :type k_para: None, or float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors. NOT USED IN COLD 
                       MODEL.
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: Chi_e
        :rtype: ndarray of shape ``[ 3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, 
                ..., ncn]``
        """
        return self._Chi_e_model(coordinates, omega, k_para, k_perp, eq_only, 
                                 time)
                                 
    def chi_i(self, coordinates, omega, k_para=None, k_perp=None, eq_only=True,
              time=0, species_id=None):
        """Calculates ion susceptibility at given locations
        
        :param coordinates: Cartesian coordinates where :math:`\chi_i` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: optional, parallel wave vectors. NOT USED IN COLD MODEL.
        :type k_para: None, or float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors. NOT USED IN COLD 
                       MODEL.
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        :param species_id: Chosen ion species to contribute to Chi_i. Optional,
                           if not given, all ion species available are added. 
        :type species_id: None, or int, or list of int.
        
        :return: Chi_i
        :rtype: ndarray of shape [3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, ...
                , ncn]
        """
        assert self.has_ion
        
        if species_id is None:
            species_id = range(len(self.ion_species))
        result = 0
        for i in species_id:
            result += self._Chi_i_model[i](coordinates, omega, k_para, k_perp, 
                                           eq_only, time)
        return result
        
    def epsilon(self, coordinates, omega, k_para=None, k_perp=None,
                eq_only=True, time=None):
        """Calculates the total dielectric tensor
        
        .. math::
            
            \epsilon = \bf(I) + \sum\limits_s \chi_s
            
        :param coordinates: Cartesian coordinates where :math:`\epsilon` will 
                            be evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)    
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: optional, parallel wave vectors. NOT USED IN COLD MODEL.
        :type k_para: None, or float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors. NOT USED IN COLD 
                       MODEL.
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: epsilon
        :rtype: ndarray of shape [ 3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, 
                ..., ncn]
        """
        
        result = self.chi_e(coordinates, omega, k_para, k_perp, eq_only, time)
        if self.has_ion:
            result += self.chi_i(coordinates, omega, k_para, k_perp, eq_only, 
                                 time)
        
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

   
class ColdElectronColdIon(ColdDielectric):
    r"""Concrete class for dielectric tensor of cold electron and cold ions.
    
    Plasma dielectric tensor , 
    
    Since cold plasma dielectric tensor does not depend on *k*, we can simply
    evaluate it at each spatial location with local plasma parameters only, 
    i.e. ne and B.  
    
    Initialization
    ==============
    :param plasma: plasma profile containing at least ne and B data
    :type plasma: :py:class:`PlasmaProfile` object
    :param ion_species: Optional, default is None. If given, chosen ion species
                        will contribute to dielectric tensor. If None, no ion
                        contribution.
    :type ion_species: None, or list of int. 
    """
    
    def __init__(self, plasma, ion_species=None):
        self._name = 'Cold Electrons and Ions Plasma Dielectric Tensor'
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

class WarmDielectric(Dielectric):
    r""" Base class for dielectric tensors consists of at least one warm fluids
    
    The key difference between warm and cold dielectric is that in warm ones, 
    k_para is a required argument.    
    
    Provide methods::
    
        chi_e(self, coordinates, omega, k_para, k_perp=None, eq_only=True,
              time=0)
              
        chi_i(self, coordinates, omega, k_para, k_perp=None, eq_only=True,
              time=0, species_id=None)
              
        epsilon(self, coordinates, omega, k_para, k_perp=None,
                eq_only=True, time=None)
    """
    
    __metaclass__ = ABCMeta
    
    def chi_e(self, coordinates, omega, k_para, k_perp=None, eq_only=True,
              time=0):
        """Calculates electron susceptibility at given locations
        
        :param coordinates: Cartesian coordinates where :math:`\chi_e` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape ``(ndim, nc1, nc2, ...
                           , ncn)``
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: optional, parallel wave vectors. 
        :type k_para: None, or float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors. NOT USED IN WARM 
                       MODEL.
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: Chi_e
        :rtype: ndarray of shape ``[ 3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, 
                ..., ncn]``
        """
        return self._Chi_e_model(coordinates, omega, k_para, k_perp, eq_only, 
                                 time)
                                 
    def chi_i(self, coordinates, omega, k_para, k_perp=None, eq_only=True,
              time=0, species_id=None):
        """Calculates ion susceptibility at given locations
        
        :param coordinates: Cartesian coordinates where :math:`\chi_i` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: optional, parallel wave vectors. 
        :type k_para: None, or float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors. NOT USED IN WARM
                       MODEL.
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        :param species_id: Chosen ion species to contribute to Chi_i. Optional,
                           if not given, all ion species available are added. 
        :type species_id: None, or int, or list of int.
        
        :return: Chi_i
        :rtype: ndarray of shape [3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, ...
                , ncn]
        """
        assert self.has_ion
        
        if species_id is None:
            species_id = range(len(self.ion_species))
        result = 0
        for i in species_id:
            result += self._Chi_i_model[i](coordinates, omega, k_para, k_perp, 
                                           eq_only, time)
        return result
        
    def epsilon(self, coordinates, omega, k_para, k_perp=None,
                eq_only=True, time=None):
        """Calculates the total dielectric tensor
        
        .. math::
            
            \epsilon = \bf(I) + \sum\limits_s \chi_s
            
        :param coordinates: Cartesian coordinates where :math:`\epsilon` will 
                            be evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)    
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: optional, parallel wave vectors. 
        :type k_para: None, or float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors. NOT USED IN WARM 
                       MODEL.
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: epsilon
        :rtype: ndarray of shape [ 3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, 
                ..., ncn]
        """
        
        result = self.chi_e(coordinates, omega, k_para, k_perp, eq_only, time)
        if self.has_ion:
            result += self.chi_i(coordinates, omega, k_para, k_perp, eq_only, 
                                 time)
        
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
                          

class WarmElectronColdIon(WarmDielectric):        
    r"""Concrete Class evaluating warm electron + cold ion plasma dielectric 
    tensor 
    
    Warm electron susceptibility tensor depends on *k_para*, we need to use 
    *k_para* information
    
    Initialization
    ==============
    :param plasma: plasma profile containing at least ne and B data
    :type plasma: :py:class:`PlasmaProfile` object
    :ion_species: Ion species ids used in calculating dielectric tensor. If 
                  None, no ion is added, only electrons contribute.
    :type ion_species: None, or int, or list of int. 
                       Optional, default to be None.
    :param float gamma: ratio of specific heat for chosen electron equation of 
                        state. gamma=3 for 1D adiabatic process, gamma=1 for
                        isothermal.
    """
    
    def __init__(self, plasma, ion_species=None, gamma=3):
        self._name = 'Warm Electron + Cold Ion Plasma Dielectric Tensor'
        self._description = 'Warm electrons and optional cold ions.'
        self._plasma = plasma
        self._Chi_e_model = SusceptWarm(plasma,'e', gamma)
        if ion_species is not None:
            self.has_ion = True
            self.ion_species = ion_species
            self._Chi_i_model = []
            for s in ion_species:
                self._Chi_i_model.append(SusceptCold(plasma,'i', s))
        else:
            self.has_ion = False
    


class HotDielectric(Dielectric):
    r""" Base class for dielectric tensors containing at least one hot(kinetic)
    component.
    
    The key difference between hot(kinetic) and fluid dielectric is that in hot
    ones, k_para and k_perp are both required.    
    
    Provide methods::
    
        chi_e(self, coordinates, omega, k_para, k_perp, eq_only=True,
              time=0)
              
        chi_i(self, coordinates, omega, k_para, k_perp, eq_only=True,
              time=0, species_id=None)
              
        epsilon(self, coordinates, omega, k_para, k_perp, eq_only=True, 
                time=None)
    """
    __metaclass__ = ABCMeta
    
    def chi_e(self, coordinates, omega, k_para, k_perp, eq_only=True,
              time=0):
        """Calculates electron susceptibility at given locations
        
        :param coordinates: Cartesian coordinates where :math:`\chi_e` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape ``(ndim, nc1, nc2, ...
                           , ncn)``
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: parallel wave vectors.
        :type k_para: float, or list of float, shape(nk_para)
        :param k_perp: perpendicular wave vectors. 
        :type k_perp: float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: Chi_e
        :rtype: ndarray of shape ``[ 3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, 
                ..., ncn]``
        """
        return self._Chi_e_model(coordinates, omega, k_para, k_perp, eq_only, 
                                 time)
        
    def chi_i(self, coordinates, omega, k_para, k_perp, eq_only=True,
              time=0, species_id=None):
        """Calculates ion susceptibility at given locations
        
        :param coordinates: Cartesian coordinates where :math:`\chi_i` will be
                            evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: optional, parallel wave vectors. 
        :type k_para: None, or float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors. 
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        :param species_id: Chosen ion species to contribute to Chi_i. Optional,
                           if not given, all ion species available are added. 
        :type species_id: None, or int, or list of int.
        
        :return: Chi_i
        :rtype: ndarray of shape [3, 3, nt, nf, nk_para, nk_perp, nc1, nc2, ...
                , ncn]
        """
        assert self.has_ion
        
        if species_id is None:
            species_id = range(len(self.ion_species))
        result = 0
        for i in species_id:
            result += self._Chi_i_model[i](coordinates, omega, k_para, k_perp, 
                                           eq_only, time)
        return result      
        
    def epsilon(self, coordinates, omega, k_para, k_perp,
                eq_only=True, time=None):
        """Calculates the total dielectric tensor
        
        .. math::
            
            \epsilon = \bf(I) + \sum\limits_s \chi_s
            
        :param coordinates: Cartesian coordinates where :math:`\epsilon` will 
                            be evaluated. The number of arrays should equal to 
                            *self.plasma.grid.dimension*.  
        :type cooridnates: ndarrays of floats with shape (ndim, nc1, nc2, ..., 
                           ncn)    
        :param omega: frequencies with which the susceptibility tensor is 
                      calculating.
        :type: 1d array of floats with shape (nf, )
        :param k_para: parallel wave vectors.
        :type k_para: float, or list of float, shape(nk_para)
        :param k_perp: optional, perpendicular wave vectors.
        :type k_perp: None, or float, or list of float, shape(nk_perp)
        :param bool eq_only: if True, only equilibrium data is used.
        :param time: time steps chosen for perturbations
        :type time: list or scalar of int, with length of ``nt``. If scalar, 
                    nt dimension is supressed in the returned array.
        
        :return: epsilon
        :rtype: ndarray of shape [nf, nc1, nc2, ..., ncn, 3, 3]
        """
        
        result = self.chi_e(coordinates, omega, k_para, k_perp, eq_only, time)
        if self.has_ion:
            result += self.chi_i(coordinates, omega, k_para, k_perp, eq_only, 
                                 time)
        
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
    

class HotElectronColdIon(HotDielectric):        
    r"""Class evaluating hot(non-relativistic) electron + cold ion plasma 
    dielectric tensor 
    
    Non-relativistic electron susceptibility tensor depends on *k_para* and 
    *k_perp*, we need to use their information.
    
    Initialization
    ==============
    :param plasma: plasma profile containing at least ne and B data
    :type plasma: :py:class:`PlasmaProfile` object
    :param ion_species: Ion species ids used in calculating dielectric tensor. 
                        If None, no ion is added, only electrons contribute.
    :type ion_species: None, or int, or list of int. 
                       Optional, default to be None.
    :param int max_harmonic: Optional, default is 4. The highest order of 
                             harmonic to keep.
    :param int max_power: NOT USED IN THIS MODEL
    """
    
    def __init__(self, plasma, ion_species=None, max_harmonic=4, 
                 max_power=None):
        self._name = 'Warm Electron + Cold Ion Plasma Dielectric Tensor'
        self._description = 'Hot(non-relativistic) electrons and optional cold\
 ions. Maximum harmonic = {}'.format(max_harmonic)
        self._plasma = plasma
        self._Chi_e_model = SusceptNonrelativistic(plasma,'e', 
                                                   max_harmonic=max_harmonic)
        if ion_species is not None:
            self.has_ion = True
            self.ion_species = ion_species
            self._Chi_i_model = []
            for s in ion_species:
                self._Chi_i_model.append(SusceptCold(plasma,'i', s))
        else:
            self.has_ion = False
                  

class RelElectronColdIon(HotDielectric):        
    r"""Class evaluating Relativistic electron + cold ion plasma 
    dielectric tensor 
    
    Relativistic electron susceptibility tensor depends on *k_para* and 
    *k_perp*, we need to use their information.
    
    Initialization
    ==============
    :param plasma: plasma profile containing at least ne and B data
    :type plasma: :py:class:`PlasmaProfile` object
    :param ion_species: Ion species ids used in calculating dielectric tensor. 
                        If None, no ion is added, only electrons contribute.
    :type ion_species: None, or int, or list of int. 
                       Optional, default to be None.
    :param int max_harmonic: Optional, default is 4. The highest order of 
                             harmonic to keep.
    :param int max_power: Optional, default is 4. The highest power of lambda
                          to keep in the formula. If it's smaller than 
                          *max_harmonic*, then it's changed to *max_harmonic*.
    """
    
    def __init__(self, plasma, ion_species=None, max_harmonic=4, max_power=4):
        self._name ='Relativistic Electron + Cold Ion Plasma Dielectric Tensor'
        self._description = 'Relativistic) electrons and optional cold ions. \
Maximum harmonic = {}, maximum power = {}'.format(max_harmonic,max_power)
        self._plasma = plasma
        self._Chi_e_model = SusceptRelativistic(plasma,'e', 
                                                   max_harmonic=max_harmonic,
                                                   max_power=max_power)
        if ion_species is not None:
            self.has_ion = True
            self.ion_species = ion_species
            self._Chi_i_model = []
            for s in ion_species:
                self._Chi_i_model.append(SusceptCold(plasma,'i', s))
        else:
            self.has_ion = False
    
    