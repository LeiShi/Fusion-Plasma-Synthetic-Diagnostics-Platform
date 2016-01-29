# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:20:15 2016

@author: lei

Propagators for electromagnetic waves propagating in plasma
"""

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy.integrate import cumtrapz

from ...Plasma.DielectricTensor import ColdDielectric, Dielectric, \
                                       ResonanceError
from ...Plasma.PlasmaProfile import PlasmaProfile
from ...GeneralSettings.UnitSystem import cgs


class Propagator(object):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def propagate(self, E_start, x_start, x_end, dx):
        pass
    

class ParaxialPerpendicularPropagator1D(Propagator):
    r""" The paraxial propagator for perpendicular propagation of 1D waves.    

    Geometry
    ========
    
    The usual coordinates system is used.
    
    z direction:
        The background magnetic field direction. Magnetic field is assumed no 
        shear.
        
    x direction:
        The direction of the wave's main propagation. In Tokamak diagnostic 
        case, it's usually very close to major radius direction. For near mid-
        plane diagnostics, it's also along the gradiant of density and 
        temperature profile.
        
    y direction:
        The 3rd direction which perpendicular to both x and z. (x,y,z) should 
        form a right-handed system
        
    
    Approximations
    ==============
    
    Paraxial approximation:
        wave propagates mainly along x direction. Refraction and diffraction 
        effects are weak with in the region of interest
        
    1D approximation:
        wave field is generally uniform perpendicular to the propagation 
        direction. Diffraction is again very weak in this assumption.
        
    Method
    ======
    
    Electromagnetic wave equation in plasma is solved under above 
    approximations. WKB kind of solution is assumed, and it's phase and 
    amplitude obtained by solving 0th and 1st order equations.
    
    The original equation is 
    
    .. math::
        
        \nabla \times \nabla \times E + \frac{\omega^2}{c^2} \epsilon\cdot E =0
        
    Using Paraxial approximation and the WKB solution [1]_:
    
    .. math:: 
        E = E_0(x) \exp\left( \mathrm{i} \int\limits^x k(x')\mathrm{d}x'\right)
        :label: WKB_solution
        
    The 0th order equation is then:
    
    .. math::
        (\epsilon - n^2 {\bf I} + n^2\hat{x}\hat{x}) \cdot
        E = 0
    
    where :math:`n \equiv ck/\omega` is the refractive index.
    
    Non-zero solution requires zero determinant of the tensor in front of E, 
    this gives us the usual dispersion relation. There are two solutions of 
    :math:`n`:
    
    .. math::
        n^2 = \epsilon_{zz} 
        \quad \text{(for O-mode)}
        
        n^2 = \frac{\epsilon_{yy}\epsilon_{xx}-\epsilon_{xy}\epsilon_{yx}}
              {\epsilon_{xx}} \quad \text{(for X-mode)}
    
    The corresponding eigen-vectors are: (without normalization)
    
    .. math::
        e_O = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} \;, \quad 
        e_X =\frac{1}{\sqrt{|\epsilon_{xy}|^2+|\epsilon_{xx}|^2}}
             \begin{pmatrix} -\epsilon_{xy} \\ \epsilon_{xx} \\ 0 
             \end{pmatrix}
        
    The 1st order equation is:
    
    .. math::
    
        2\mathrm{i}(kE_0' + \frac{1}{2}k'E_0) + \frac{\omega^2}{c^2}\delta 
        \epsilon \cdot e_{O/X} E_0 = 0.
        
    Letting :math:`F \equiv k^{1/2}E_0`, we have
    
    .. math::
    
        2\mathrm{i}k F' + \frac{\omega^2}{c^2}\delta 
        \epsilon \cdot e_{O/X} F = 0.
        
    A Formal solution to this equation is
    
    .. math::
        F(x) =\exp\left( \mathrm{i} \int\limits_0^x \frac{\omega^2}{2k(x') c^2}
               \delta \epsilon (x') \mathrm{d}x'\right) F(0)
        :label: 1st_order_solution
     
    Equation :eq:`WKB_solution` and :eq:`1st_order_solution` gives us the whole
    solution up to the first order.

    Numerical Scheme
    ====================
    
    We need to numerically evaluate the phase advance for electric field at 
    each y/z location. In 1-D approximation, they are naturally non-coupled, so
    we can calculate them separately.
    
    Phase term includes two parts:
        
        1. main phase :math:`k_0`. 
        
           This part is from 0-th order equation, 
           and the solution is the normal dispersion relation:
        
           .. math::
        
               \frac{c^2k_0^2}{\omega^2} = \frac{\epsilon_{yy} \epsilon_{xx} - 
                     \epsilon_{xy}*\epsilon_{yx}}{\epsilon_{xx}}
        
        2. fluctuated phase.             
           
           This part is in first order solution, and will be retained by 
           solving for :math:`F(x)` using :eq:`1st_order_solution`.
           
    So, two integrations over ``x`` will be evaluated numerically. Trapezoidal 
    integration is used to have 2nd order accurancy in ``dx``. 
    
    Initialization
    ==============
    
    :param plasma: plasma profile under study
    :type plasma: :py:class:`...Plasma.PlasmaProfile.PlasmaProfile` object
    :param dielectric_class: dielectric tensor model used for calculating full
                             epsilon
    :type dielectric_class: derived class from 
                            :py:class:`...Plasma.DielectricTensor.Dielectric`
    :param polarization: specific polarization for cold plasma perpendicular 
                         waves. Either 'X' or 'O'.
    :type polarization: string, either 'X' or 'O'
    :param direction: propagation direction. 1 means propagating along positive
                      x direction, -1 along negative x direction.
    :type direction: int, either 1 or -1.
    :param float tol: the tolerance for testing zero components and determining
                      resonance and cutoff. Default to be 1e-14
    
    :raise AssertionError: if parameters passed in are not as expected.    
    
    References
    ==========
    
    .. [1] WKB Approximation on Wikipedia. 
           https://en.wikipedia.org/wiki/WKB_approximation       
    """
    

    def __init__(self, plasma, dielectric_class, polarization, 
                 direction, unitsystem=cgs, tol=1e-14):
        assert isinstance(plasma, PlasmaProfile) 
        assert issubclass(dielectric_class, Dielectric)
        assert polarization in ['X','O']
        assert direction in [1, -1]
        
        self.ray_dielectric = ColdDielectric(plasma)
        self.dielectric = dielectric_class(plasma)
        self.polarization = polarization
        self.direction = direction
        self.tol = tol
        self.unit_system = unitsystem
        
        
    def _generate_epsilon(self):
        r"""Generate main dielectric :math:`\epsilon_0` along the ray and 
        fluctuating dielectric :math:`\delta \epsilon` on the full mesh
        
        The main ray is assumed along x direction
        
        :param float omega: angular frequency of the wave
        :param x_coords: x coordinates of the main ray
        :type x_coords: 1D array of float
        :param float ray_y: y coordinate of the main ray
        :param ray_z: z coordinate of the main ray, if plasma is 3D
        :type ray_z: None if plasma is 2D, float if 3D
        :param coords: coordinates for full mesh
        :type coords: list of ndarrays of float. length should equal the 
                      dimension of plasma.
        """
        omega = self.omega
        coords = self.coords
        x_coords = self.x_coords
        ray_y = self.ray_y
        ray_z = self.ray_z
        time = self.time
        
        if ray_z is None:
            # 2D plasma space is expected
            assert (self.dielectric.dimension == 2)                     
            self.eps0 = \
              self.ray_dielectric.epsilon(omega, [np.ones_like(x_coords)*ray_y, 
                                              x_coords], True)
                                    
            self.deps = \
              self.dielectric.epsilon(omega, coords, False, time) - \
              self.eps0[:, :, np.newaxis, :]
              
        else:
            # 3D plasma is expected
            assert (self.dielectric.dimension == 3)
            self.eps0 = \
              self.ray_dielectric.epsilon(omega, [np.ones_like(x_coords)*ray_z,
                                              np.ones_like(x_coords)*ray_y,
                                              x_coords], True)
            self.deps = \
              self.dielectric.epsilon(omega, coords, False, time) - \
              self.eps0[:, :, np.newaxis, np.newaxis, :]
              
    
    def _generate_eOX(self):
        """Create unit polarization vectors along the ray
        """
        if self.polarization == 'O':
            self.e_x = 0
            self.e_y = 0
            self.e_z = 1
            
        else:
            exx = self.eps0[0, 0, :]
            # eyy = self.eps0[1, 1, :]
            exy = self.eps0[0, 1, :]
            # eyx = self.eps0[1, 0, :]
            exy_mod = np.abs(exy)
            exx_mod = np.abs(exx)
            norm = 1/np.sqrt(exy_mod*exy_mod + exx_mod*exx_mod) 
            self.e_x = -exy * norm
            self.e_y = exx * norm
            self.e_z = 0
       
    def _generate_k(self):
        """Calculate k_0 along the reference ray path
        """
        
        omega = self.omega
        
        eps0 = self.eps0
        
        if self.polarization == 'O':
            P = np.real(eps0[2,2,:])
            if np.any(P < self.tol):
                raise ResonanceError('Cutoff of O mode occurs. Paraxial \
propagator is not appropriate in this case. Use full wave solver instead.')
            self.k_0 = omega/self.unit_system['c'] * np.sqrt(P)
            
        else:
            S = np.real(eps0[0,0,:])
            D = np.imag(eps0[1,0,:])
            numerator = S*S - D*D
            if np.any(S < self.tol):
                raise ResonanceError('Cold Resonance of X mode occurs. Change \
to full wave solver with Relativistic Dielectric Tensor to overcome this.')
            if np.any(numerator < self.tol):
                raise ResonanceError('Cutoff of X mode occrus. Use full wave \
solver instead of paraxial solver.')
            self.k_0 = omega/self.uni_system['c'] * np.sqrt(numerator/S)
            
        
    def _generate_F(self):
        """integrate the phase term to get F.
        
        Note: F=k^(1/2) E
        """
        
        dexx = self.deps[0, 0, ...]
        dexy = self.deps[0, 1, ...]
        deyx = self.deps[1, 0, ...]
        deyy = self.deps[1, 1, ...]
        # dexz = self.deps[0, 2, ...]
        # dezx = self.deps[2, 0, ...]
        # deyz = self.deps[1, 2, ...]
        # dezy = self.deps[2, 1, ...]
        dezz = self.deps[2, 2, ...]
        
        omega2 = self.omega*self.omega
        c = self.unit_system['c']
        c2 = c*c        
        
        if self.polarization == 'O':
            de_O = dezz
            F_0 = self.E_start * np.sqrt(self.k_0[0])
            self.delta_phase = self.direction* omega2/c2 * \
                                  cumtrapz(de_O/self.k_0, x=self.x_coords, 
                                           initial=0)
            self.E_0 = np.exp(1j*self.delta_phase)*F_0[..., np.newaxis] /\
                       np.sqrt(self.k_0)
                     
        else:
            ex = self.e_x
            ey = self.e_y
            de_X = ex*dexx*ex + ex*dexy*ey + ey*deyx*ex + ey*deyy*ey
            F_0 = self.E_start * np.sqrt(self.k_0[0])
            self.delta_phase = self.direction* omega2/c2 * \
                                  cumtrapz(de_X/self.k_0, x=self.x_coords, 
                                           initial=0)
            self.E_0 = np.exp(1j*self.delta_phase)*F_0[..., np.newaxis] / \
                       np.sqrt(self.k_0)
                     
    def _generate_E(self):
        """Calculate the total E including the main phase advance
        """
        self.main_phase = self.direction*cumtrapz(self.k_0, x=self.x_coords,
                                                  initial=0)
        self.E = self.E_0 * np.exp(1j*self.main_phase)
            
       
    def propagate(self, time, omega, E_start, x_start, x_end, nx, y_coords, 
                  ray_y, z_coords=None, ray_z=None, x_coords=None):
        r"""Propagate electric field from x_start to x_end
        
        The propagation is assumed mainly in x direction. The (ray_y,ray_z) is 
        the (y,z) coordinates of the reference ray path, on which the 
        equilibrium dielectric tensor is taken to be :math:`\epsilon_0`. 
        
        See :py:class:`ParaxialPerpendicularPropagator1D` for detailed 
        description of the method and assumptions.
        
        :param int time: chosen time step of perturbation in plasma.
        :param float omega: angular frequency of the wave, omega must be 
                            positive.
        :param E_start: complex amplitude of the electric field at x_start
        :type E_start: ndarray of complex with shape ([nz,] ny)
        :param float x_start: starting point for propagation
        :param float x_end: end point for propagation
        :param int nx: number of intermediate steps to use for propagation
        :param y_coords: y coordinates of the E_start mesh
        :type y_coords: 1D array of float, y coordinates of E_start data
        :param float ray_y: y coordinate of the main ray
        :param z_coords: z coordinates of E_start mesh, if plasma is 3D
        :type z_coords: None if plasma is 2D, 1D array of float if 3D
        :param ray_z: z coordinate of the main ray, if plasma is 3D
        :type ray_z: None if plasma is 2D, float if 3D
        :param x_coords: *Optional*, x coordinates to use for propagation, if
                         given, *x_start*, *x_end*, and *nx* are ignored.
        :type x_coords: 1d array of float. Must be monotonic.
        
        """ 
        assert omega > 0
        assert E_start.shape == y_coords.shape
        self.time = time        
        self.omega = omega
        self.E_start = E_start
        if (x_coords is None):
            self.x_coords = np.linspace(x_start, x_end, nx+1)
        else:
            self.x_coords = x_coords
        self.y_coords = np.copy(y_coords)
        self.ray_y = ray_y
        if z_coords is not None:
            assert E_start.shape == z_coords.shape
            self.z_coords = np.copy(z_coords)
        else:
            self.z_coords = None
        self.ray_z = ray_z
        
        coord_x = np.zeros_like(y_coords)[..., np.newaxis] + self.x_coords
        coord_y = np.zeros_like(self.x_coords) + self.y_coords[..., np.newaxis]
        self.coords = [coord_y, coord_x]
        if self.ray_z is not None:
            coord_z = np.zeros_like(self.x_coords) + self.z_coords[..., 
                                                                   np.newaxis]
            self.coords = [coord_z, coord_y, coord_x]
            
        self._generate_epsilon()
        self._generate_eOX()
        self._generate_k()
        self._generate_F()
        self._generate_E()
        
        return self.E
        
        

class ParaxialPerpendicularPropagator(Propagator):

    def __init__(self, plasma, electron_model='cold', mode='X', direction = 1):
        self.dielectric = ColdDielectric(plasma)
        self.mode = mode
        self.direction = direction

    

    def propagate(self, omega, E_start, x_start, x_end, dx):
        pass
    
    