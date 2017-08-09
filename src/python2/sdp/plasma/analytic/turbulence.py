# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 16:11:48 2015

@author: lei

Module for random data generation

Mainly for the use of creating testing plasma turbulence, given statistical properties such as auto-correlation function, or power spectrum, in either space or time.
"""
import numpy as np
from scipy.integrate import quadrature, quad
from numpy.random import random
from abc import MetaABC, abstractmethod, abstractproperty

from .fluctuation import Fluctuation

class Turbulence_Generation_Error(Exception):
    def __init__(self,msg):
        self.message = msg
    def __str__(self):
        return str(self.message)

class Auto_Correlation_Function(object):
    """base class for user-defined auto-correlation functions.
    Derived classes shoud substantiate following methods:
        __call__(tau): evaluate the auto-correlation function at given tau values. tau: ndarray of float. returns: ndarray as the same shape of tau.
        power_spectrum(): returns the corresponding power-sectrum function. returns: Power_Spectrum object
        get_func(): returns the underline function. returns: function callable

    auto-correlation function of a signal :math:`I(t)` in time is defined as:
    ..math::
        \psi(\tau) \equiv \lim_{T \to \infty} \frac{1}{T} \int\limits_0^T I(t)I(t+\tau) \rm{d}t

    It is one of the most important statistical properties that characterize the random signal `I(t)`.
    It is related to the power spectrum of the signal by:
    ..math::

        w(f) =  4\; \int\limits_0^\infty \psi(\tau)\; \cos \;2\pi f \tau \; \rm{d} \tau

        \psi(\tau) = \int\limits_0^\infty w(f) \; \cos \; 2\pi f \tau \; \rm{d} f

    where `w(f)` is defined as in :py:class:`Power_Spectrum`
    Furthur information on correlation functions can be found in [1].

    Note: It is the user's responsibility to check the dimension and unit system of the function provided to this class.

    [1]. Rice, S. O. "Mathematical Analysis of Random Noise", in Selected Papers on Noise and Stochastic Processes, Dover Publications, inc. 1954.


    """
    def __init__(self):
        self._func_set = False
    def set_func(self, func):
        """ setting the correlation function. A valid function must be even, and real valued. The function should be a Python callable, and expect a 1-d array of floats as input.
        """
        self._func = func
        self._func_set = True

    def get_func(self):
        if (self._func_set):
            return self._func
        else:
            raise Turbulence_Generation_Error('Auto_correlation function not set.')

    def __call__(self,tau):
        """ Although the _func expects only 1-d float array. The wrapper for this class can take any shape of input, and return the same shaped array.
        """
        if(self._func_set):
            tau = np.array(tau)
            shape = tau.shape
            res = self._func(tau.flatten())
            return res.reshape(shape)
        else:
            raise Turbulence_Generation_Error('Auto-correlation function not set. A callable function needs to be assigned to an Auto_Correlation_Function object before calling it.')

    def power_spectrum(self):
        """ power_spectrum for a general correlation function needs to be evaluated using the integration method
        """
        ps = Power_Spectrum()
        def ps_func(f):
            assert len(f.shape) == 1 #only take 1-d array as input
            res = np.empty_like(f)
            f *= 2*np.pi
            for i in range(len(f)):
                res[i] = 4 * quad(self._func, 0, np.inf, weight= 'cos', wvar = f[i])[0]
            return res
        ps.set_func(ps_func)
        return ps

class Power_Spectrum(object):
    """base class for user-defined power spectrum functions

    The power spectrum of a signal :math:`I(t)` is defined as:

    ..math::

        w(f) \equiv \lim_{T \to \infty} \frac{2|S(f)|^2}{T}

    where :math:`S(f)\equiv\int_0^T I(t) \exp(-2\pi \rm{i} f t)\rm{d}t` is the the Fourier Spectrum of `I(t)` within time 0 to T.
    The relation between *power spectrum* and *auto-correlation function* can be found in :py:class:`Auto_Correlation_Function`.

    Derived classes should substantiate the following methods:
        __call__(self,f): evaluate the power spectrum function at given frequencies, f can be either float or array of floats.
        auto_correlation_function(self): return the corresponding Auto_Correlation_Function object.
        get_func(self): return the Python callable function.
    """
    def __init__(self):
        self._func_set = False

    def set_func(self, func):
        self._func = func
        self._func_set = True

    def get_func(self):
        if(self._func_set):
            return self.func
        else:
            raise Turbulence_Generation_Error('Power_Spectrum function not set.')
    def __call__(self, f):
        if(self._func_set):
            f = np.array(f)
            shape = f.shape
            res = self._func(f.flatten())
            return res.reshape(shape)
        else:
            raise Turbulence_Generation_Error('Power_Spectrum function not set. A callable function needs to be assigned to an Power_Spectrum object before calling it.')

    def auto_correlation_function(self):
        """ auto_correlation_function corresponding to a general power_spectrum function needs to be evaluated using the integration method
        """
        acf = Auto_Correlation_Function()
        def ac_func(tau):
            assert len(tau.shape) == 1 #only take 1-d array as input
            tau *= 2*np.pi
            res = np.empty_like(tau)
            for i in range(len(tau)):
                res[i] = quad(self._func, 0, np.inf, weight= 'cos', wvar = tau[i])[0]
            return res
        acf.set_func(ac_func)
        return acf

class Constant_Corr(Auto_Correlation_Function):
    """Constant auto-correlation function.
    This special kind of correlation function is used to treat fluctuations not changing along certain direction
    """
    def __init__(self, A):
        super(Constant_Corr,self).__init__()
        self.A = A
        self.set_func(lambda t: self.A)

    def power_spectrum(self):
        return Delta_Spec(self.A)

class Delta_Spec(Power_Spectrum):
    """Dirac Delta function like power spectrum
    Mathematically, a Dirac Delta function is the fourier transform of the uniform function, and has following properties:
    1) :math:`\delta(x)=0` at `x \ne 0`, and `\delta(x)=+\infty` at `x = 0`
    2) :math:`\int_a^b \delta(x) dx = 1` if `a< 0 <b`, and `= 0` if `0<a` or `0>b`, specially, if either a or b = 0, the integral gives 0.5
    3) nth derivative of delta-function is odd(even) if n is odd(even).

    Numerically, delta-functions can only be approximated by a finite pulse. The pulse's height and width should
    """
class Cauchy_Spec(Power_Spectrum):
    """Cauchy Power Spectrum
    A power spectrum that has a Cauchy distribution shape. Cauchy distribution is (note that power spectrums are always even functions in f):
    ..math::

        p(f) = \frac{1}{\pi \gamma} \frac{\gamma^2}{f^2 + \gamma^2}

    and Cauchy power spectrum is:
    ..math::

        w(f) = A \frac{\gamma^2}{f^2 + \gamma^2} = \frac{\rho}{\pi \gamma} \frac{\gamma^2}{f^2 + \gamma^2}

    where `A=\rho/(\pi \gamma)` is the peak power density, with `\rho` the total power.

    """
    def __init__(self, gamma=None, A=None, rho=None):
        gamma_given = False
        A_given = False
        rho_given = False
        arg_count = 0
        if (gamma is not None):
            self.gamma = gamma
            gamma_given = True
            arg_count += 1
        if (A is not None):
            self.A = A
            A_given = True
            arg_count += 1
        if (rho is not None):
            self.rho = rho
            rho_given = True
            arg_count +=1

        assert arg_count == 2 # Among gamma, A, and rho, only 2 should be specified. The third should be calculated from the other two.
        if (not gamma_given):
            self.gamma = self.rho / (np.pi * self.A)
        if (not A_given):
            self.A = self.rho / (np.pi * self.gamma)
        if (not rho_given):
            self.rho = np.pi * self.A * self.gamma

        super(Cauchy_Spec,self).__init__()
        gamma2 = self.gamma*self.gamma
        w = lambda f: self.A * gamma2 / ( f*f + gamma2 )
        self.set_func(w)

    def auto_correlation_function(self):
        """ The Cauchy shape power spectrum corresponds to exponential decay auto-correlation function
        """
        return Exponential_Corr(self.rho/2, 1/(2*np.pi*self.gamma))


class Exponential_Corr(Auto_Correlation_Function):
    """Exponential decay correlation function
    A correlation function in the shape of exponential decay. i.e.
    ..math::

        \psi(\tau) = A \rm{e}^{- |\tau|/\tau_0}

    where `A` is the auto-correlation at no time delay, and `\tau_0` the typical decorrelation time.

    """

    def __init__(self, A, tau_0):
        super(Exponential_Corr, self).__init__()
        func = lambda t: A*np.exp(-np.abs(t)/tau_0)
        self.set_func(func)
        self.A = A
        self.tau_0 = tau_0

    def power_spectrum(self):
        """ The Exponential correlation function corresponds to Cauchy power spectrum
        """
        return Cauchy_Spec(rho=2*self.A, gamma = 1/(2*np.pi*self.tau_0))


class Gaussian_Corr(Auto_Correlation_Function):
    """Gaussian decay correlation function
    ..math::

        \psi(\tau) = A \rm{e}^{- \tau^2/\tau_0^2}
    """

    def __init__(self,A, tau_0):
        super(Gaussian_Corr, self).__init__()
        func = lambda t: A*np.exp(-t**2/tau_0**2)
        self.set_func(func)
        self.A = A
        self.tau_0 = tau_0

    def power_spectrum(self):
        """ Gaussian correlation function corresponds to Gaussian power spectrum
        ..math::

            w(f) = 4\int_0^\infty \psi(\tau) \cos \; 2\pi f \tau \; \rm{d}\tau = 2 \sqrt{\pi} \tau_0 A \exp (-\pi^2 \tau_0^2 f^2)


        """
        return Gaussian_Spec(A = 2*np.sqrt(np.pi)*self.tau_0 * self.A, f0 = 1/(np.pi*self.tau_0))

class Gaussian_Spec(Power_Spectrum):
    """ Gaussian shaped Power Spectrum
    ..math::

        w(f) = A \rm{e}^{-f^2/f_0^2}
    """
    def __init__(self,A,f0):
        super(Gaussian_Spec,self).__init__()
        func = lambda f: A*np.exp(-f**2/f0**2)
        self.set_func(func)
        self.A = A
        self.f0 = f0

    def auto_correlation_function(self):
        """ Gaussian power spectrum corrsponds to Gaussian shaped auto-correlation function

        ..math::

            \psi(\tau) = \int_0^\infty w(f) \cos \; 2\pi f \tau \rm{d}f = \frac{\sqrt{\pi}f_0 A}{2} \exp(-\pi^2 f_0^2 \tau^2)
        """
        return Gaussian_Corr(A = np.sqrt(np.pi)*self.f0*self.A/2 , tau_0 = 1/(np.pi*self.f0))



#Now we have defined some common Auto-Correlation functions and Power Spectra, we are ready to produce fluctuations



class Turbulence(Fluctuation):
    """Random Turbulence generator, generates turbulent fluctuations on a specific grid based on given spatial correlation or spectral power functions. Realizations are independent to each other. Derived from the abstract class :py:class:`Fluctuation`.
    Initialize with the the following parameters:
    :param acf_radial: Auto correlation function in radial coordinate
    :type acf_radial: :py:class:`Auto_Correlation_Function`
    :param acf_pol: Auto correlation function in poloidal coordinate
    :type acf_pol: :py:class:`Auto_Correlation_Function`
    :param acf_tor: Auto correlation function in toroidal coordinate
    :type acf_tor: :py:class:`Auto_Correlation_Function`
    :param grid: Optional, default grid for generating turbulence
    :type grid: :py:class:`AnalyticGrid`
    :param random_seed: Optional, set the random seed for :py:module`random` module, default to be 1
    :type random_seed: int

    Substantiate the following methods:
    realize: generate one random realization for given auto-correlation functions on given grid


    """
    @property
    def acf_radial(self):
        return self._acf_radial

    @acf_radial.setter
    def acf_radial(self,acf):
        assert isinstance(acf, Auto_Correlation_Function)
        self._acf_radial = acf

    @acf_radial.deleter
    def acf_radial(self):
        del self._acf_radial

    @property
    def acf_pol(self):
        return self._acf_pol

    @acf_pol.setter
    def acf_pol(self,acf):
        assert isinstance(acf, Auto_Correlation_Function)
        self._acf_pol = acf

    @acf_pol.deleter
    def acf_pol(self):
        del self._acf_pol

    @property
    def acf_tor(self):
        return self._acf_tor

    @acf_tor.setter
    def acf_tor(self,acf):
        assert isinstance(acf, Auto_Correlation_Function)
        self._acf_tor = acf

    @acf_tor.deleter
    def acf_tor(self):
        del self._acf_tor

    def __init__(self, acf_radial, acf_pol, acf_tor, grid = None, random_seed = 1):
        """initialization. Set auto-correlation functions on each direction, and corresponding power spectra
        """
        self.acf_radial = acf_radial
        self.acf_pol = acf_pol
        self.acf_tor = acf_tor
        self._rseed = random_seed
        if(grid is not None):
            assert isinstance(grid,AnalyticGrid)
        self.grid = grid

        self.ps_radial = self.acf_radial.power_spectrum()
        self.ps_pol = self.acf_pol.power_spectrum()
        self.ps_tor = self.acf_tor.power_spectrum()

    def realize(self, grid = None):
        if(grid is None):
            grid = self.grid
        if(grid is None):
            raise Turbulence_Generation_Error('Default grid not set. Please set a default grid before generating turbulence without a specific grid.')
        rgrid = self.grid.r1d
