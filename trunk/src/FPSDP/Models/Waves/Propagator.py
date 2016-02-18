# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:20:15 2016

@author: lei

Propagators for electromagnetic waves propagating in plasma
"""

from __future__ import print_function
import sys
from time import clock
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import cumtrapz, quadrature

from ...Plasma.DielectricTensor import HotDielectric, Dielectric, \
                                       ColdElectronColdIon, ResonanceError 
from ...Plasma.PlasmaProfile import PlasmaProfile
from ...GeneralSettings.UnitSystem import cgs


class Propagator(object):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def propagate(self, E_start, x_start, x_end, dx):
        pass
    

class ParaxialPerpendicularPropagator1D(Propagator):
    r""" The paraxial propagator for perpendicular propagation of 1D waves.    

    Initialization
    ==============
    
    ParaxialPerpendicularPropagator1D(self, plasma, dielectric_class, 
                                      polarization, direction, unitsystem=cgs, 
                                      tol=1e-14)
    
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
    :param unitsystem: unit system used 
    :type unitsystem: :py:class:`...GeneralSettings.UnitSystem` object
    :param float tol: the tolerance for testing zero components and determining
                      resonance and cutoff. Default to be 1e-14
    :param int max_harmonic: highest order harmonic to keep. 
                             Only used in hot electron models.
    :param int max_power: highest power in lambda to keep.
                          Only used in hot electron models.
    
    :raise AssertionError: if parameters passed in are not as expected.

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
        plasma perturbation is uniform in both y and z directions. Wave 
        amplitude can be Fourier transformed along both of these directions.
        
    Method
    ======
    
    Electromagnetic wave equation in plasma is solved under above 
    approximations. WKB kind of solution is assumed, and it's phase and 
    amplitude obtained by solving 0th and 1st order equations.
    
    The original equation is 
    
    .. math::
        
        -\nabla \times \nabla \times E + \frac{\omega^2}{c^2} \epsilon\cdot E=0
        
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
    
    The sign of :math:`k` is then determined by direction of propagation. In 
    our convention, :math:`k>0` means propagation along positive x, :math:`k<0`
    along negative x.    
    
    The corresponding eigen-vectors are:
    
    .. math::
        e_O = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} \;, \quad 
        e_X =\frac{1}{\sqrt{|\epsilon_{xy}|^2+|\epsilon_{xx}|^2}}
             \begin{pmatrix} -\epsilon_{xy} \\ \epsilon_{xx} \\ 0 
             \end{pmatrix}
        
    The 2nd order equation is:
    
    O-mode:    
    
    .. math::
    
        2\mathrm{i}(kE_0' + \frac{1}{2}k'E_0) + \left( \frac{\partial^2}
        {\partial y^2}+ P \frac{\partial^2}{\partial z^2}\right) E_0 + 
        \frac{\omega^2}{c^2}\delta \epsilon_{OO} E_0 = 0,

    where :math:`\delta \epsilon_{OO} \equiv e^*_{O} \cdot 
    \delta \epsilon \cdot e_{O}` is the perturbed dielectric tensor element
    projected by O mode eigen vector. Since :math:`\delta\epsilon` does
    not depend on y and z, we can Fourier transform along y and z direction, 
    and obtain the equation for :math:`\hat{E}_0(x, k_y, k_z)`:
    
    .. math::
        2\mathrm{i}(k\hat{E}_0' + \frac{1}{2}k'\hat{E}_0) - 
        \left( k_y^2 + P k_z^2 \right) \hat{E}_0 + 
        \frac{\omega^2}{c^2}\delta \epsilon_{OO} \hat{E}_0 = 0,
    
    X-mode:
    
    .. math::
        2\mathrm{i}(kE_0' + \frac{1}{2}k'E_0) + 
        \left[ \frac{\partial^2}{\partial y^2} + 
        \left( \frac{S^2+D^2}{S^2}- \frac{(S^2-D^2)D^2}{(S-P)S^2}\right)
        \frac{\partial^2}{\partial z^2}\right] E_0 + 
        \frac{S^2+D^2}{S^2} \frac{\omega^2}{c^2}\delta \epsilon_{XX} E_0 = 0,
    
    Fourier transform along y, z directions, we have equation for 
    :math:`\hat{E}_0`.
    
    .. math::
        2\mathrm{i}(k\hat{E}_0' + \frac{1}{2}k'\hat{E}_0) - 
        \left[ k_y^2 + \left( \frac{S^2+D^2}{S^2}- 
        \frac{(S^2-D^2)D^2}{(S-P)S^2}\right) k_z^2 \right] \hat{E}_0 + 
        \frac{S^2+D^2}{S^2} \frac{\omega^2}{c^2}\delta \epsilon_{XX} \hat{E}_0 
        = 0,
    
    
    Letting :math:`F \equiv |k|^{1/2}\hat{E}_0` we have
    
    .. math::
    
        2\mathrm{i}k F'(x, k_y, k_z) -  \left[k_y^2 +  
        C(x) k_z^2\right] F(x, k_y, k_z) + 
        A(x) \frac{\omega^2}{c^2}\delta \epsilon_{OO/XX} 
        F(x, k_y, k_z)= 0.
        
    A Formal solution to this equation is
    
    .. math::
        F(x, k_y, k_z) =\exp\left( \mathrm{i} \int\limits_0^x 
        \frac{1}{2k(x')}\left(A(x')\frac{\omega^2}{c^2}\delta \epsilon (x') 
        - k_y^2 - C(x') k_z^2 \right) \mathrm{d}x'\right) F(0)
        :label: 2nd_order_solution
        
    where :math:`A(x')=1, C(x') = P` for O-mode, 
    :math:`A(x')=\frac{S^2+D^2}{S^2}, C(x')=\frac{S^2+D^2}{S^2}- 
    \frac{(S^2-D^2)D^2}{(S-P)S^2}` for X-mode.
     
    Equation :eq:`WKB_solution` and :eq:`2nd_order_solution` gives us the whole
    solution up to the 2nd order.

    Numerical Scheme
    ====================
    
    We need to numerically evaluate the phase advance for electric field with 
    each k_y,k_z value, then we inverse Fourier transform it back to y,z space.
    
    Phase term includes two parts:
        
        1. main phase :math:`k_0`. 
        
           This part is from 0-th order equation, 
           and the solution is the normal dispersion relation:
           
           O-Mode:

           .. math::
               \frac{c^2k_0^2}{\omega^2} = \epsilon_{zz}
           
           X-Mode:            
           
           .. math::
        
               \frac{c^2k_0^2}{\omega^2} = \frac{\epsilon_{yy} \epsilon_{xx} - 
                     \epsilon_{xy}*\epsilon_{yx}}{\epsilon_{xx}}
        
        2. 2nd order phase correction.             
           
           This part is in 2nd order solution, and will be retained by 
           solving for :math:`F(x)` using :eq:`2nd_order_solution`.
           
    So, two integrations over ``x`` will be evaluated numerically. Trapezoidal 
    integration is used to have 2nd order accurancy in ``dx``. 
    
        
    
    References
    ==========
    
    .. [1] WKB Approximation on Wikipedia. 
           https://en.wikipedia.org/wiki/WKB_approximation       
    """
    

    def __init__(self, plasma, dielectric_class, polarization, 
                 direction, unitsystem=cgs, tol=1e-14, max_harmonic=4, 
                 max_power=4):
        assert isinstance(plasma, PlasmaProfile) 
        assert issubclass(dielectric_class, Dielectric)
        assert polarization in ['X','O']
        assert direction in [1, -1]
        
        self.main_dielectric = ColdElectronColdIon(plasma)
        if issubclass(dielectric_class, HotDielectric):
            self.fluc_dielectric = dielectric_class(plasma, 
                                                    max_harmonic=max_harmonic,
                                                    max_power=max_power)
        else:
            self.fluc_dielectric = dielectric_class(plasma)
        self.polarization = polarization
        self.direction = direction
        self.tol = tol
        self.unit_system = unitsystem
        
        # Prepare the cold plasma 
        x_fine = self.plasma.grid.X1D
        eps0_fine = self.main_dielectric.epsilon([x_fine], omega, True)
        S = np.real(eps0[0,0])
        D = np.imag(eps0[1,0])
        P = np.real(eps0[2,2])
        self._S =

        print('Propagator 1D initialized.', file=sys.stdout)
        
    def _k0(self, x):
        """ evaluate main wave vector at specified x locations
        
        This function is mainly used to carry out the main phase integral with
        increased accuracy.
        """
        if self.polarization == 'O':
            n
        
        
    def _generate_epsilon0(self, mute=True):
        r"""Generate main dielectric :math:`\epsilon_0` along the ray 
        
        The main ray is assumed along x direction.
        Main dielectric tensor uses Cold Electron Cold Ion model
        
        Needs Attribute:
            
            self.omega
            
            self.x_coords
            
            self.main_dielectric
            
        Create Attribute:
            
            self.eps0
        """
        tstart = clock()
        
        omega = self.omega
        x_coords = self.x_coords
        self.eps0 = self.main_dielectric.epsilon([x_coords], omega, True)
        
        tend = clock()

        if not mute:        
            print('Epsilon0 generated. Time used: {:.3}s'.format(tend-tstart), 
                  file=sys.stdout)
        
    
    def _generate_k(self, mask_order=3, mute=True):
        """Calculate k_0 along the reference ray path
        
        :param mask_order: the decay order where kz will be cut off. 
                           If |E_k| peaks at k0, then we pick the range (k0-dk,
                           k0+dk) to use in calculating delta_epsilon. dk is
                           determined by the decay length of |E_k| times the
                           mask_order. i.e. the masked out part have |E_k| less
                           than exp(-mask_order)*|E_k,max|.
        """
        
        tstart = clock()        
        
        omega = self.omega
        c=self.unit_system['c']
        
        eps0 = self.eps0
        
        if self.polarization == 'O':
            P = np.real(eps0[2,2,:])
            if np.any(P < self.tol):
                raise ResonanceError('Cutoff of O mode occurs. Paraxial \
propagator is not appropriate in this case. Use full wave solver instead.')
            self.k_0 = self.direction*omega/c * np.sqrt(P)
            
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
            self.k_0 = self.direction*omega/c * np.sqrt(numerator/S)
        # generate wave vector arrays
            
        # Fourier transform E along y and z
        self.E_k_start = np.fft.fft2(self.E_start)
        
        self.nz = len(self.z_coords)
        self.dz = self.z_coords[1] - self.z_coords[0]
        self.kz = 2*np.pi*np.fft.fftfreq(self.nz, self.dz)
        
        self.ny = len(self.y_coords)
        self.dy = self.y_coords[1] - self.y_coords[0]
        self.ky = 2*np.pi*np.fft.fftfreq(self.ny, self.dy)
        
        # we need to mask kz in order to avoid non-physical zero k_parallel 
        # components 
        
        # find out the peak location
        marg = np.argmax(np.abs(self.E_k_start))
        # find the y index of the peak
        myarg = marg % self.ny
        Ekmax = np.max(np.abs(self.E_k_start))
        E_margin = Ekmax*np.exp(-mask_order)
        # create the mask for components smaller than our marginal E
        mask = np.abs(self.E_k_start[:,myarg]) < E_margin
        # choose the largest kz in not masked part as the marginal kz
        kz_margin = np.max(np.abs(self.kz[~mask]))
        # fill all outside kz with the marginal kz
        self.masked_kz = np.copy(self.kz)
        self.masked_kz[mask] = kz_margin
        #save central kz id
        self.central_kz = marg // self.ny
        self.margin_kz = np.argmin(mask)
        
        tend = clock()
        
        if not mute:
            print('k0, ky, kz generated. Time used: {:.3}s'.format\
                  (tend-tstart), file=sys.stdout)
        
        
        
    
    def _generate_delta_epsilon(self, mute=True):
        r"""Generate fluctuated dielectric :math:`\delta\epsilon` on full mesh 
        
        Fluctuated dielectric tensor may use any dielectric model.
        
        Needs Attribute::
            
            self.omega
            
            self.x_coords
            
            self.k_0
            
            self.kz
            
            self.eps0
            
        Create Attribute::
            
            self.deps
        """
        tstart = clock()
        
        omega = self.omega
        x_coords = self.x_coords
        k_perp = self.k_0
        k_para = self.masked_kz
        time = self.time
        self.deps = np.empty((3,3,len(k_para),len(x_coords)),dtype='complex')
        for i,x in enumerate(x_coords):                              
            self.deps[... ,i] = self.fluc_dielectric.epsilon([x],omega, k_para, 
                                                      k_perp[i], self.eq_only, 
                                                      time)-\
                                self.eps0[:,:,np.newaxis,i]
        # add one dimension for ky, between kz, and spatial coordinates.
        self.deps = self.deps[..., np.newaxis, :]
        
        tend = clock()        
        
        if not mute:
            print('Delta_epsilon generated.Time used: {:.3}s'.format(tend-\
                   tstart), file=sys.stdout)
              
    
    def _generate_eOX(self, mute=True):
        """Create unit polarization vectors along the ray
        """
        tstart = clock()
        
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
        
        tend = clock()
        if not mute:
            print('Polarization eigen-vector generated. Time used: {:.3}s'.\
                   format(tend-tstart), file=sys.stdout)
            
        
    def _generate_F(self, mute=True):
        """integrate the phase term to get F.
        
        Note: F=k^(1/2) E
        """  

        tstart = clock()        
        
        ny=self.ny
        nz=self.nz
        
        ky = self.ky[np.newaxis, :, np.newaxis]
        kz = self.kz[:, np.newaxis, np.newaxis]
        
        omega2 = self.omega*self.omega
        c = self.unit_system['c']
        c2 = c*c 
        
        S = np.real(self.eps0[0,0])
        D = np.imag(self.eps0[1,0])
        P = np.real(self.eps0[2,2])
        
        if self.polarization == 'O':
            de_O = self.deps[2, 2, ... ]
            # de_kO = np.fft.fft2(de_O, axes=(0,1))
            F_k0 = self.E_k_start * np.sqrt(np.abs(self.k_0[0]))
            self.delta_phase = cumtrapz((omega2/c2*de_O-ky*ky- \
                                         P*kz*kz)/(2*self.k_0), 
                                           x=self.x_coords, initial=0)
            self.E_k0 = np.exp(1j*self.delta_phase)*F_k0[..., np.newaxis] /\
                       np.sqrt(np.abs(self.k_0))
                     
        else:

            dexx = self.deps[0, 0, ...]
            dexy = self.deps[0, 1, ...]
            deyx = self.deps[1, 0, ...]
            deyy = self.deps[1, 1, ...]
            S2 = S*S
            D2 = D*D
            # vacuum case needs special attention. C coefficient has a 0/0 part
            # the limit gives C=1, which is correct for vacuum.
            vacuum_idx = np.abs(D) < self.tol
            non_vacuum = np.logical_not(vacuum_idx)            
            C = np.empty_like(self.x_coords)
            C[vacuum_idx] = 1
            C[non_vacuum] = (S2+D2)/S2 - (S2-D2)*D2/(S2*(S-P))
            ex = self.e_x
            ey = self.e_y
            ex_conj = np.conj(ex)
            ey_conj = np.conj(ey)
            de_X = ex_conj*dexx*ex + ex_conj*dexy*ey + ey_conj*deyx*ex + \
                   ey_conj*deyy*ey
            de_X = de_X * np.ones((nz,ny,1))
            # de_kX = np.fft.fft2(de_X, axes=(0,1))
            F_k0 =self.E_k_start * np.sqrt(np.abs(self.k_0[0]))

            if(self._debug):
                self.pe = (S2+D2)/S2* omega2/c2 *de_X /\
                                         (2*self.k_0)
                self.dphi_eps = cumtrapz(self.pe, 
                                         x=self.x_coords, initial=0)
                self.dphi_ky = cumtrapz(-ky*ky/(2*self.k_0), 
                                        x=self.x_coords, initial=0)
                self.dphi_kz = cumtrapz(-C*kz*kz/(2*self.k_0), 
                                        x=self.x_coords, initial=0)
                self.delta_phase = self.dphi_eps + self.dphi_ky + self.dphi_kz
            
            else:
                self.delta_phase = cumtrapz(((S2+D2)/S2* omega2/c2 *de_X -\
                                            ky*ky-C*kz*kz)/(2*self.k_0), 
                                            x=self.x_coords, initial=0)
            self.E_k0 = np.exp(1j*self.delta_phase)*F_k0[..., np.newaxis] / \
                       np.sqrt(np.abs(self.k_0))
                       
        tend = clock()
        
        if not mute:               
            print('F function calculated. Time used: {:.3}s'.format\
                  (tend-tstart), file=sys.stdout)
                     
    def _generate_E(self, mute=True):
        """Calculate the total E including the main phase advance
        """

        tstart = clock()        
        
        #self.main_phase = cumtrapz(self.k_0, x=self.x_coords, initial=0)
        #self.E_k = self.E_k0 * np.exp(1j*self.main_phase)
        self.E_k = self.E_k0        
        self.E = np.fft.ifft2(self.E_k, axes=(0,1))
        
        tend = clock()
        
        if not mute:
            print('E field calculated. Time used: {:.3}s'.format(tend-tstart),
                  file=sys.stdout)
            
       
    def propagate(self, omega, x_start, x_end, nx, E_start, y_E, 
                  z_E, x_coords=None, time=None, mute=True, debug_mode=False):
        r"""propagate(self, omega, x_start, x_end, nx, E_start, y_E, 
                  z_E, x_coords=None, regular_E_mesh=True, time=None)
        
        Propagate electric field from x_start to x_end
        
        The propagation is assumed mainly in x direction. The (ray_y,ray_z) is 
        the (y,z) coordinates of the reference ray path, on which the 
        equilibrium dielectric tensor is taken to be :math:`\epsilon_0`. 
        
        See :py:class:`ParaxialPerpendicularPropagator1D` for detailed 
        description of the method and assumptions.
        
        :param float omega: angular frequency of the wave, omega must be 
                            positive.
        :param E_start: complex amplitude of the electric field at x_start,
        :type E_start: ndarray of complex with shape (nz, ny),
        :param float x_start: starting point for propagation
        :param float x_end: end point for propagation
        :param int nx: number of intermediate steps to use for propagation
        :param y_E: y coordinates of the E_start mesh, uniformly placed
        :type y_E: 1D array of float
        :param z_E: z coordinates of E_start mesh, uniformly placed
        :type z_E: 1D array of float 
        :param x_coords: *Optional*, x coordinates to use for propagation, if
                         given, *x_start*, *x_end*, and *nx* are ignored.
        :type x_coords: 1d array of float. Must be monotonic.
        :param int time: chosen time step of perturbation in plasma. If None, 
                         only equilibrium plasma is used. 
        :param bool mute: if True, no intermediate outputs for progress.
        :param bool debug_mode: if True, additional detailed information will 
                                be saved for later inspection.                           
        
        """ 

        tstart = clock()        
        
        assert omega > 0
        assert E_start.shape[1] == y_E.shape[0]
        assert E_start.shape[0] == z_E.shape[0]
        
        if time is None:
            self.eq_only = True
            self.time=None
        else:
            self.eq_only = False
            self.time = time
        self._debug = debug_mode
        
        self.omega = omega
        
        self.E_start = E_start            
        self.y_coords = np.copy(y_E)
        self.z_coords = np.copy(z_E)
         
        if (x_coords is None):
            self.x_coords = np.linspace(x_start, x_end, nx+1)
        else:
            self.x_coords = x_coords        
        
        self._generate_epsilon0(mute=mute)
        self._generate_k(mute=mute)
        self._generate_delta_epsilon(mute=mute)
        self._generate_eOX(mute=mute)
        self._generate_F(mute=mute)
        self._generate_E(mute=mute)
        
        tend = clock()
        
        if not mute:
            print('1D Propogation Finish! Check the returned E field. More \
infomation is available in Propagator object.\nTotal Time used: {:.3}s\n'.\
                  format(tend-tstart), file=sys.stdout)
        
        return self.E
        
        
class ParaxialPerpendicularPropagator2D(Propagator):
    r""" The paraxial propagator for perpendicular propagation of 2D waves.    

    1. Initialization
       
    
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
                      
    :param int max_harmonic: highest order harmonic to keep. 
                             Only used in hot electron models.
    :param int max_power: highest power in lambda to keep.
                          Only used in hot electron models.
    
    :raise AssertionError: if parameters passed in are not as expected.    

    2. Geometry
    
    
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
        
    
    3. Approximations
    
    
    Paraxial approximation:
        wave propagates mainly along x direction. Refraction and diffraction 
        effects are weak with in the region of interest
        
    2D approximation:
        Plasma perturbations are assumed uniform along magnetic field lines, so
        the perturbed dielectric tensor is not a function of z. So we can 
        Fourier transform the wave amplitude in z direction and analyze each 
        k_parallel component separately.
        
    4. Ordering
    
    
    We assume the length scales in the problem obey the following ordering:
    
    .. math::
        \frac{\lambda}{E}\frac{\partial E}{\partial y} \sim \delta
    
    .. math::
        \frac{\delta\epsilon}{\epsilon_0} \sim \delta^2
        
    where :math:`\epsilon_0` is chosen to be the equilibrium dielectric 
    tensor along main light path, normally use Cold or Warm formulation, and
    :math:`\delta\epsilon` the deviation of full dielectric tensor from 
    :math:`\epsilon_0` due to fluctuations, away from main light path, and/or
    relativstic kinetic effects. 
        
    5. Method
    
    
    Electromagnetic wave equation in plasma is solved under above 
    approximations. WKB kind of solution is assumed, and it's phase and 
    amplitude obtained by solving 0th and 2nd order equations.
    
    The original equation is 
    
    .. math::
        
        -\nabla \times \nabla \times E + \frac{\omega^2}{c^2} \epsilon\cdot E=0
        
    Using Paraxial approximation and the WKB solution [1]_:
    
    .. math:: 
        E = E_0(x,y,z) \exp\left( \mathrm{i} \int\limits^x k(x')\mathrm{d}x'
        \right)
        :label: WKB_solution
        
    a. The 0th order equation
    
    
    .. math::
        (\epsilon_0 - n^2 {\bf I} + n^2\hat{x}\hat{x}) \cdot
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
    
    The corresponding eigen-vectors are: 
    
    .. math::
        e_O = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} \;, \quad 
        e_X =\frac{1}{\sqrt{|\epsilon_{xy}|^2+|\epsilon_{xx}|^2}}
             \begin{pmatrix} -\epsilon_{xy} \\ \epsilon_{xx} \\ 0 
             \end{pmatrix}
        
    *The 1st order equation is natually satisfied.*
    
    
    b. The 2nd order equation
    
    
    2nd order equations are different for O-mode and X-mode

    (i) O-mode    
    
    
    .. math::
    
        2\mathrm{i}(kE_0' + \frac{1}{2}k'E_0) +
        \frac{\partial^2 E_0}{\partial y^2} + 
        P\frac{\partial^2 E_0}{\partial z^2} +
        \frac{\omega^2}{c^2}e_O^* \cdot \delta\epsilon \cdot e_O E_0 = 0.
        
    Letting :math:`F \equiv k^{1/2}E_0`, we have
    
    .. math::
        2\mathrm{i}k \frac{\partial F(x,y,k_z)}{\partial x} + 
        \frac{\partial^2}{\partial y^2} F(x,y,k_z) - P k_z^2 F(x,y,k_z)
        +\frac{\omega^2}{c^2}\delta \epsilon_{OO} F(x,y,k_z) = 0,
    
    where :math:`\delta\epsilon_{OO} \equiv e_O^* \cdot\delta\epsilon\cdot e_O 
    = \delta \epsilon_{zz}`, and :math:`P \equiv \epsilon_{0,zz}`.
    
    (ii) X-mode
    
    
    .. math::
        2\mathrm{i}(kE_0' + \frac{1}{2}k'E_0) + 
        \left[ \frac{\partial^2}{\partial y^2} + 
        \left( \frac{S^2+D^2}{S^2}- \frac{(S^2-D^2)D^2}{(S-P)S^2}\right)
        \frac{\partial^2}{\partial z^2}\right] E_0 + \frac{S^2+D^2}{S^2}
        \frac{\omega^2}{c^2}\delta \epsilon_{XX} E_0 = 0,
    
    Letting :math:`F \equiv k^{1/2}E_0`, and Fourier transform along z 
    direction, we have
    
    .. math::
    
        2\mathrm{i}k F'(x, y, k_z) + \frac{\partial^2}{\partial y^2}F(x,y,k_z)
        -\left( \frac{S^2+D^2}{S^2}- \frac{(S^2-D^2)D^2}{(S-P)S^2}\right) 
        k_z^2  F(x, y, k_z) + \frac{S^2+D^2}{S^2}
        \frac{\omega^2}{c^2}\delta \epsilon_{XX} F(x, y, k_z)= 0.
        
    where :math:`S \equiv \epsilon_{0,xx}` and :math:`D \equiv \mathrm{i}
    \epsilon_{0,xy}` are notations adopted from Cold Plasma Dielectric tensor,
    and :math:`\delta \epsilon_{XX} \equiv e_X^* \cdot \delta \epsilon \cdot
    e_X` is tensor element projected on X-mode eigen-vector.
    
    The O-mod and X-mode equations need to be solved numerically because they 
    contain partial derivatives respect to y, and dielectric tensor depends on 
    y.
    
    The scheme is described in the next section.

    6. Numerical Scheme
    
    The full solution includes a main phase part and an amplitude part.
    
    a. Main phase  
        
        As in 1D case, the main phase is integration of :math:`k_0` over x.
       
        :math:`k_0` is obtained through dispersion relation which is the 
        solvability condition for 0th order equation.
       
        O-mode:       
       
        .. math::
           k_0^2 = \frac{\omega^2}{c^2} \epsilon_{0,zz}
       
        X-mode:
       
        .. math::    
            k_0^2 = \frac{\omega^2}{c^2}\frac{\epsilon_{0,yy} \epsilon_{0,xx} - 
                 \epsilon_{0,xy}\epsilon_{0,yx}}{\epsilon_{0,xx}}

        The sign of :math:`k_0` is determined by direction of the propagation.
        
    b. Amplitude
           
        The amplitude equation is more complicated than that in 1D, because now
        perturbed dielectric tensor depends on y, we can no longer Fourier 
        transform in y direction. 
        
        The equation now has a general form of
        
        .. math::
            2\mathrm{i}k \frac{\partial F}{\partial x} + 
            \frac{\partial^2 F}{\partial y^2} + C(y) F = 0,
            
        We notice that :math:`B\equiv \partial^2/\partial y^2` operator does 
        not commute with :math:`C(y)`, so there is not a single eigen state 
        :math:`F` for both operators. A numerical technique to solve this 
        equation is that we propagate F along x with very small steps. Within
        each step, we propagate operator :math:`B` and :math:`C` separately, so
        we can use their own eigen state in their substeps. The scheme is like
        
        .. math::
            F(x+\delta x, y, k_z) = 
            \exp\left( \frac{\mathrm{i}}{2k} \frac{C\delta x}{2} \right)
            \cdot \exp \left(\frac{\mathrm{i}}{2k} B \delta x\right) 
            \cdot \exp \left( \frac{\mathrm{i}}{2k} \frac{C\delta x}{2} \right)
            F(x),
            
        We can show that this scheme evolves the phase with an accuracy of 
        :math:`o(\delta x^2)`. 
        
        Since original equation is an order one differential equation in x, 
        Magnus expansion theorum [2]_ tells us the exact solution to the 
        equation goes like

        .. math::
            F(x') = \exp(\Omega_1 + \Omega_2 + ...)F(x).
        
        where 
        
        .. math::
            \Omega_1 = \int\limits_x^{x'} A(x_1) dx_1
        
        .. math::
            \Omega_2 = \int\limits_x^{x'}\int\limits_{x}^{x_1} [A(x_1),A(x_2)] 
                       dx_1 dx_2 
        and 
    
        .. math::
            A = \frac{i}{2k(x)} (B+C(x)) 

        .. math::
            [A(x_1), A(x_2)] &= A(x_1)A(x_2) - A(x_2)A(x_1) \\
                             &= -\frac{1}{4k^2} ([B, C(x_2)]-[B, C(x_1)])
              
        if we only propagate x for a small step :math:`\delta x`, we can see 
        that :math:`\Omega_1 \sim \delta x`, but :math:`\Omega_2 \sim \delta 
        x^3`. We write
        
        .. math::
            F(x+\delta x) &= \exp( A(x_1) \delta x + o(\delta x^3)) F(x) \\
                          &= \exp\left( \frac{i\delta x}{2k}(B+C) + 
                          o(\delta x^3)\right) F(x).  
        
        Then using Baker-Campbell-Housdorff formula [3]_, we can show:
        
        .. math::
            \exp\left( \frac{\mathrm{i}}{2k} \frac{C\delta x}{2} \right)
            \cdot \exp \left(\frac{\mathrm{i}}{2k} B \delta x\right) 
            \cdot \exp \left( \frac{\mathrm{i}}{2k} \frac{C\delta x}{2} \right)
            = \exp\left( \frac{i\delta x}{2k}(B+C) + o(\delta x^3)\right)
            
        So, finally, we show that our scheme gives a :math:`F(x+\delta x)` with
        a phase error of :math:`o(\delta x^3)`. Since the total step goes as 
        :math:`1/\delta x`, we finally get a :math:`F(x)` with phase error 
        :math:`\sim o(\delta x^2)`.
        
    
    7. References
    
    .. [1] WKB Approximation on Wikipedia. 
           https://en.wikipedia.org/wiki/WKB_approximation  
    
    .. [2] https://en.wikipedia.org/wiki/Magnus_expansion
       
    .. [3] https://en.wikipedia.org/wiki/
           Baker-Campbell-Hausdorff_formula
    """
    
    def __init__(self, plasma, dielectric_class, polarization, 
                 direction, ray_y, unitsystem=cgs, tol=1e-14, 
                 max_harmonic=4, max_power=4):
        assert isinstance(plasma, PlasmaProfile) 
        assert issubclass(dielectric_class, Dielectric)
        assert polarization in ['X','O']
        assert direction in [1, -1]
        
        self.main_dielectric = ColdElectronColdIon(plasma)
        self.ray_y = ray_y
        if issubclass(dielectric_class, HotDielectric):
            self.fluc_dielectric = dielectric_class(plasma, 
                                                    max_harmonic=max_harmonic,
                                                    max_power=max_power)
        else:
            self.fluc_dielectric = dielectric_class(plasma)
        self.polarization = polarization
        self.direction = direction
        self.tol = tol
        self.unit_system = unitsystem
        
        print('Propagator 2D initialized.', file=sys.stdout)
        
        
    def _generate_epsilon(self, mute=True):
        r"""Generate main dielectric :math:`\epsilon_0` along the ray 
        
        The main ray is assumed along x direction.
        Main dielectric tensor uses Cold Electron Cold Ion model
        
        Needs Attribute:
            
            self.omega
            
            self.x_coords
            
            self.main_dielectric
            
        Create Attribute:
            
            self.eps0
        """

        tstart = clock()        
        
        omega = self.omega
        # x_coords needs to be enlarged twice since we need to split each step
        # into two steps to evolve the two operators
        self.nx_calc = len(self.x_coords)*2-1
        self.calc_x_coords = np.empty((self.nx_calc))
        self.calc_x_coords[::2] = self.x_coords
        self.calc_x_coords[1::2] = (self.x_coords[:-1]+self.x_coords[1:])/2.
        
        self.eps0 = self.main_dielectric.epsilon\
                         ([np.ones_like(self.calc_x_coords)*self.ray_y,
                           self.calc_x_coords], omega, True)
                           
        tend = clock()
        
        if not mute:                   
            print('Epsilon0 generated. Time used: {:.3}'.format(tend-tstart), 
                  file=sys.stdout)
 
                                
    def _generate_k(self, mute=True, mask_order=3):
        """Calculate k_0 along the reference ray path
        
        Need Attributes:
        
            self.omega
            
            self.eps0
            
            self.polarization
            
            self.tol
            
            self.direction
            
            self.y_coords
            
            self.ny
            
            self.z_coords

            self.nz
            
            self.E_start

        Create Attributes:
        
            self.k_0

            self.ky

            self.kz 
            
            self.dy
            
            self.dz
            
            self.masked_kz
            
            self.E_k_start
        """
        
        tstart = clock()        
        
        omega = self.omega
        c=self.unit_system['c']
        
        eps0 = self.eps0
        
        if self.polarization == 'O':
            P = np.real(eps0[2,2,:])
            if np.any(P < self.tol):
                raise ResonanceError('Cutoff of O mode occurs. Paraxial \
propagator is not appropriate in this case. Use full wave solver instead.')
            self.k_0 = self.direction*omega/c * np.sqrt(P)
            
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
            self.k_0 = self.direction*omega/c * np.sqrt(numerator/S)
        
        # Fourier transform E along z
        self.E_k_start = np.fft.fft(self.E_start, axis=0)
        
        self.nz = len(self.z_coords)
        self.dz = self.z_coords[1] - self.z_coords[0]
        self.kz = 2*np.pi*np.fft.fftfreq(self.nz, self.dz)[:, np.newaxis, 
                                                           np.newaxis]
        
        self.ny = len(self.y_coords)
        self.dy = self.y_coords[1] - self.y_coords[0]
        self.ky = 2*np.pi*np.fft.fftfreq(self.ny, self.dy)[np.newaxis, :, 
                                                           np.newaxis]
        
        # we need to mask kz in order to avoid non-physical zero k_parallel 
        # components 
        
        # find out the peak location
        marg = np.argmax(np.abs(self.E_k_start))
        # find the y index of the peak
        myarg = marg % self.ny
        Ekmax = np.max(np.abs(self.E_k_start))
        E_margin = Ekmax*np.exp(-mask_order)
        # create the mask for components smaller than our marginal E
        mask = np.abs(self.E_k_start[:,myarg]) < E_margin
        # choose the largest kz in not masked part as the marginal kz
        kz_margin = np.max(np.abs(self.kz[~mask]))
        # fill all outside kz with the marginal kz
        self.masked_kz = np.copy(self.kz)
        self.masked_kz[mask] = kz_margin
        
        self.central_kz = marg // self.ny
        self.margin_kz = np.argmin(mask)
        
        tend = clock()
        
        if not mute:
            print('k0, kz generated. Time used: {:.3}'.format(tend-tstart), 
                  file=sys.stdout)
        
                                 
    def _generate_delta_epsilon(self, mute=True):
        r"""Generate fluctuated dielectric :math:`\delta\epsilon` on full mesh 
        
        Fluctuated dielectric tensor may use any dielectric model.
        
        Needs Attribute::
            
            self.omega
            
            self.x_coords
            
            self.y_coords
            
            self.k_0
            
            self.kz
            
            self.eps0
            
            self.time
            
        Create Attribute::
            
            self.deps
        """

        tstart = clock()        
        
        omega = self.omega  
        time = self.time
        k_perp = self.k_0        
        k_para = self.masked_kz[:,0,0]
        y1d = self.y_coords
        self.deps = np.empty((3,3,self.nz, self.ny, self.nx_calc), 
                             dtype='complex')
        for i,x in enumerate(self.calc_x_coords):        
            x1d = np.zeros_like(y1d) + x 
            self.deps[..., i] = self.fluc_dielectric.epsilon([y1d, x1d], omega, 
                                               k_para, k_perp[i], self.eq_only,
                                                             time)-\
                                self.eps0[:,:,np.newaxis,np.newaxis,i]
        
        tend = clock()        
        
        if not mute:                        
            print('Delta epsilon generated. Time used: {:.3}'.\
                   format(tend-tstart), file=sys.stdout)
                                                          
    
    def _generate_eOX(self, mute=True):
        """Create unit polarization vectors along the ray
        
        Need Attributes::
            
            self.polarization
            
            self.eps0
        
        Create Attributes::
        
            self.e_x
            
            self.e_y
            
            self.e_z
        """

        tstart = clock()        
        
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
        
        tend = clock()        
        
        if not mute:
            print('Polarization eigen-vector generated. Time used: {:.3}'.\
                  format(tend-tstart), file=sys.stdout)
       
    
            
    
    def _generate_C(self, mute=True):
        """prepare C operator for refraction propagation
        
        C = omega^2 / c^2 * deps[2,2] for O mode

        C = 
        omega^2/c^2 (D^2 deps[0,0] + iDS (deps[1,0]-deps[0,1]) + S^2 deps[1,1])
        /S^2   for X mode

        Need Attributes::
        
            self.omega
            
            self.unit_system
            
            self.nx
            
            self.ny
            
            self.deps
            
            self.eps0
            
        Create Attributes::
        
            self.C
            
        """

        tstart = clock()        
        
        omega = self.omega
        c = self.unit_system['c']        
        self.C = np.empty((self.ny, self.nx), dtype='complex')
        
        if self.polarization == 'O':
            self.C = omega*omega/(c*c) * self.deps[2,2]
            
        else:
            S = np.real(self.eps0[0,0])
            D = np.imag(self.eps0[1,0])
            S2 = S*S
            D2 = D*D
            self.C = omega*omega/(c*c) * ( D2*self.deps[0,0] + \
            1j*D*S*(self.deps[1,0]-self.deps[0,1]) + S2*self.deps[1,1] ) / S2
        
        tend = clock()        
        
        if not mute:
            print('Operator C generated. Time used: {:.3}'.format(tend-tstart),
                  file=sys.stdout)
    
    
    def _generate_F(self, mute=True):
        """Prepare F0(x0,y,kz).
        
        Note: F=k^(1/2) E
        
        In order to increase efficiency, we change the axis order into [X,Y,Z]
        for solving F. Afterwards, we'll change back to [Z, Y, X].
        
        Need Attributes::
        
            self.E_k_start
            
            self.k_0
            
            self.nz, self.ny, self.nx_calc
            
        Create Attributes::
            
            self.F_k_start
            
            self.Fk
        """
        
        tstart = clock()

        # F = sqrt(k)*E
        self.F_k_start = np.sqrt(np.abs(self.k_0[0]))*self.E_k_start
        self.Fk = np.empty((self.nz, self.ny, self.nx_calc), dtype='complex')
        self.Fk[:,:,0] = self.F_k_start
        
        # Now we integrate over x using our scheme, taking care of B,C operator
        self._generate_C()
        
        if self._debug:
            # in debug mode, we want to store the phase advances due to 
            # diffraction and refractions.
            self.dphi_eps = np.empty_like(self.C[..., ::2])
            self.dphi_ky = np.empty_like(self.C[..., ::2])
            self.dphi_eps[0] = 0
            self.dphi_ky[0] = 0
            self._counter = 1
        
        i=0
        while(i < self.nx_calc-1):
            F = self.Fk[:,:,i]
            self.Fk[:,:,i+1] = self._refraction(F, i, forward=True)
                        
            i = i + 1
            F = self.Fk[:,:,i]
            
            self.Fk[:,:,i+1] = self._diffraction_y(F, i)
            
            i = i + 1
            F = self.Fk[:,:,i]
            self.Fk[:,:,i] = self._refraction(F, i, forward=False)
        
        tend = clock()
        if not mute:
            print('F field calculated. Time used: {:.3}'.format(tend-tstart),
                  file=sys.stdout)
            
            

    
    
    def _refraction(self, F, i, forward=True):
        """ propagate the phase step with operator C
        
        advance F with dx using dielectric data at self.calc_x_coords[i]
        if forward==True, dx = calc_x_coords[i+1]-calc_x_coords[i]
        otherwise, dx = calc_x_coords[i]-calc_x_coords[i-1]
        
        refraction propagation always happens at knots.
        
        Need Attributes::
        
            self.calc_x_coords
            
            self.k_0
            
            self.C
        
        Create Attributes::
        
            None
        """
                
        
        if forward:
            dx = self.calc_x_coords[i+1]-self.calc_x_coords[i]
        else:
            dx = self.calc_x_coords[i]-self.calc_x_coords[i-1]
        
        C = self.C[...,i]
        phase = dx* C/(2*self.k_0[i])
        
        if self._debug:
            if forward:
                self._temp_dphi_eps = phase
            else:
                self.dphi_eps[..., self._counter] = \
                                         self.dphi_eps[..., self._counter-1]+\
                                         self._temp_dphi_eps + phase
                self._counter += 1
        
        return np.exp(1j*phase)*F
        
    def _diffraction_y(self, F, i):
        """propagate the phase step with operator B
        
        advance F with dx = calc_x_coords[i+1] - calc_x_coords[i-1]
        
        Fourier transform along y, and the operator B becomes:

        B(ky) = -ky^2        
        
        diffraction propagation always happens at center between two knots
        
        Need Attributes::
        
            self.calc_x_coords
            
            self.ky
            
            self.k_0
            
        Create Attributes::
        
            None
        
        """
        
        dx = self.calc_x_coords[i+1]-self.calc_x_coords[i-1]
        ky = self.ky[0,:,0]
        B = -ky*ky
        phase = B*dx/(2*self.k_0[i])
        if self._debug:
            self.dphi_ky[..., self._counter] = \
                                   self.dphi_ky[..., self._counter-1] + phase
            
        Fk = np.exp(1j * phase) * fft(F)
        return ifft(Fk)
        
    def _generate_phase_kz(self, mute=True):
        """ Propagate the phase due to kz^2
        
        a direct integration can be used
        
        Need Attributes::
        
            self.polarization
            
            self.eps0
            
            self.kz
            
            self.calc_x_coords
            
            self.tol
            
        Create Attributes::
        
            self.phase_kz
        """

        tstart = clock()        
        
        if self.polarization == 'O':
            P = np.real(self.eps0[2,2])
            self.phase_kz = cumtrapz(-P*self.kz*self.kz/(2*self.k_0), 
                                x=self.calc_x_coords, initial=0)            
        else:
            S = np.real(self.eps0[0,0])
            D = np.imag(self.eps0[1,0])
            P = np.real(self.eps0[2,2])
            S2 = S*S
            D2 = D*D
            # vacuum case needs special attention. C coefficient has a 0/0 part
            # the limit gives C=1, which is correct for vacuum.
            vacuum_idx = np.abs(D) < self.tol
            non_vacuum = np.logical_not(vacuum_idx)            
            C = np.empty_like(self.calc_x_coords)
            C[vacuum_idx] = 1
            C[non_vacuum] = (S2+D2)/S2 - (S2-D2)*D2/(S2*(S-P))
            
            self.phase_kz = cumtrapz(- C*self.kz*self.kz / (2*self.k_0), 
                                           x=self.calc_x_coords, initial=0)
        
        tend = clock()        
        if not mute:                                   
            print('Phase related to kz generated. Time used: {:.3}'.\
                  format(tend-tstart), file=sys.stdout)
                                           
        
                     
    def _generate_E(self, mute=True):
        """Calculate the total E including the main phase advance
        
        Need Attributes:
        
            self.k_0
            
            self.calc_x_coords
            
            self.Fk
            
            self.phase_kz
            
            self.k_0
            
        Create Attributes::
        
            self.main_phase
            
            self.F
            
            self.E
        """
        tstart = clock()        
        
        self._generate_phase_kz()
        #self.main_phase = cumtrapz(self.k_0, x=self.calc_x_coords, initial=0)
        #self.Fk = self.Fk * np.exp(1j * self.main_phase)
        self.Fk = self.Fk * np.exp(1j * self.phase_kz)
        self.F = np.fft.ifft(self.Fk, axis=0)
        self.E = self.F / np.sqrt(np.abs(self.k_0))
        
        tend = clock()
        if not mute:
            print('E field calculated. Time used: {:.3}'.format(tend-tstart), 
                  file=sys.stdout)
            
       
    def propagate(self, omega, x_start, x_end, nx, E_start, y_E, 
                  z_E, x_coords=None, regular_E_mesh=True, time=None, 
                  mute=True, debug_mode=False):
        r"""propagate(self, time, omega, x_start, x_end, nx, E_start, y_E, 
                  z_E, x_coords=None)
        
        Propagate electric field from x_start to x_end
        
        The propagation is assumed mainly in x direction. The (ray_y,ray_z) is 
        the (y,z) coordinates of the reference ray path, on which the 
        equilibrium dielectric tensor is taken to be :math:`\epsilon_0`. 
        
        See :py:class:`ParaxialPerpendicularPropagator1D` for detailed 
        description of the method and assumptions.
        
        :param int time: chosen time step of perturbation in plasma.
        :param float omega: angular frequency of the wave, omega must be 
                            positive.
        :param E_start: complex amplitude of the electric field at x_start,
        :type E_start: ndarray of complex with shape (nz, ny),
        :param float x_start: starting point for propagation
        :param float x_end: end point for propagation
        :param int nx: number of intermediate steps to use for propagation
        :param y_E: y coordinates of the E_start mesh, uniformly placed
        :type y_E: 1D array of float
        :param z_E: z coordinates of E_start mesh, uniformly placed
        :type z_E: 1D array of float 
        :param x_coords: *Optional*, x coordinates to use for propagation, if
                         given, *x_start*, *x_end*, and *nx* are ignored.
        :type x_coords: 1d array of float. Must be monotonic.
                                    
        
        """ 
        
        tstart = clock()
        
        assert omega > 0
        assert E_start.shape[1] == y_E.shape[0]
        assert E_start.shape[0] == z_E.shape[0]
        
        if time is None:
            self.eq_only = True
            self.time = None
        else:
            self.eq_only = False            
            self.time = time  
        self._debug = debug_mode            
            
        self.omega = omega
        
        self.E_start = E_start            
        self.y_coords = np.copy(y_E)
        self.ny = len(self.y_coords)
        self.z_coords = np.copy(z_E)
        self.nz = len(self.z_coords)
         
        if (x_coords is None):
            self.x_coords = np.linspace(x_start, x_end, nx+1)
        else:
            self.x_coords = x_coords
        self.nx = len(self.x_coords)
        
        self._generate_epsilon(mute=mute)
        self._generate_k(mute=mute)
        self._generate_delta_epsilon(mute=mute)
        self._generate_eOX(mute=mute)        
        self._generate_F(mute=mute)
        self._generate_E(mute=mute)
        
        tend = clock()
        
        if not mute:
            print('2D Propagation Finish! Check the returned E field. More \
infomation is available in Propagator object. Total time used: {:.3}'.\
                   format(tend-tstart), file=sys.stdout)
        
        return self.E[...,::2]
    