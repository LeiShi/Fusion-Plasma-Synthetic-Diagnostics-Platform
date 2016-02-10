r"""
This module provides ways to evaluate the Plasma Dispersion Function [1]_, 
:math:`Z(x)`, and other related functions, specifically, the 
:math:`\mathcal{F}_q(\phi,\psi)` Function [2]_. 

Faddeeva function 
=====================
.. math::
   
  w(z) \equiv \exp(-z^2) \; {\mathrm {erfc}}(-{\mathrm i}z)
    
is used, where :math:`{\mathrm {erfc}}(z)` is the complementary error function. 
It is evaluated using the python wrapper of Steven G. Johnson's routine, 
provided by scipy, see :py:func:`scipy.spetial.wofz` for more details.


Plasma Dispersion Function(PDF)
====================================
 
The PDF is related to Faddeeva function as

.. math::
   
  Z(z) = {\mathrm i}\sqrt{\pi} \; w(z) \; .

PDF has the following property [1]_:

.. math::
   Z'(z) = -2(1+zZ(z)) \; ,

and it's easy to show the following recurrence relation 

.. math::
   Z^m(z) = -2[(m-1)Z^{m-2}(z) + zZ^{m-1}(z)] \quad \mathrm{for}\; m>2 \; .
    

Weakly Relativistic Plasma Dispersion Function
===============================================

:math:`\mathcal{F}_q` function is related to PDF as [2]_:

.. math::    
  \mathcal{F}_{\frac{1}{2}}(\phi,\psi) = 
   -\frac{1}{2\phi}[Z(\psi-\phi)+Z(-\psi-\phi)] \; ,

.. math::   
  \mathcal{F}_{\frac{3}{2}}(\phi,\psi) = 
   -\frac{1}{2\psi}[Z(\psi-\phi)-Z(-\psi-\phi)] \; ,

.. math::
  \mathcal{F}_{q+2}(\phi,\psi) = 
   (1+\phi^2\mathcal{F}_q-q\mathcal{F}_{q+1})/\psi^2 \; .
   
The derivatives of :math:`\mathcal{F}_q` respect to :math:`\phi^2` can be 
evaluated as:

.. math::    
  \mathcal{F}_q^m \equiv \frac{\partial^m \mathcal{F}_q}{\partial(\phi^2)^m} 
  = \mathcal{F}_{q-1}^{m-1} - \mathcal{F}_q^{m-1} \; ,

.. math::    
  \mathcal{F}_{q+2}^m =
  (\phi^2\mathcal{F}_q^m - q\mathcal{F}_{q+1}^m + m\mathcal{F}_q^{m-1})/\psi^2.

However, as pointed out in [2]_, evaluating derivatives using the first formula 
may suffer from the cancellation of two large numbers. A more reliable way is 
to express the derivatives of :math:`\mathcal{F}_{1/2}` and 
:math:`\mathcal{F}_{3/2}` in terms of derivatives of the PDF, and then use the 
second formula to evaluate larger q's.

Fianlly, for special case, :math:`\psi=0`, L'Hopital rule needs to be used to
evaluate the "0/0" kind expressions. More details in Appendix part of [2]_.

Fast Evaluators
================

In order to evaluate ``Fq`` and ``Fmq`` functions faster, two new classes 
``FqFastEvaluator`` and ``FmqFastEvaluator`` are provided. In initialization, 
these classes calculate ``Fq`` or ``Fmq`` of given order on given parameter 
meshes, and generate linear interpolators on these mesh. Then, when called with
given parameters, they use the existing interpolator to evaluate an 
approximated value for ``Fq`` or ``Fmq``. The parameter mesh is on 
:math:`\mu\delta = \psi^2 - \phi^2` and :math:`\psi` space, because they are 
both real and the function value is essentially center peaked at the origin. 
The calling signiture is almost the same as ``Fq`` and ``Fmq`` functions, 
except that the order of the function are now determined at the time of 
initialization. Two suggested mesh, ``mudelta_mesh`` and ``psi_mesh`` are 
provided for default use.


.. [1] https://farside.ph.utexas.edu/teaching/plasma/lectures1/node87.html

.. [2] I.P.Shkarofsky, "New representations of dielectric tensor elements in 
       magnetized plasma", J. Plasma Physics(1986), vol. 35, part 2, pp. 
       319-331

"""


import warnings

from numpy.lib.scimath import sqrt
import numpy as np
from scipy.special import wofz, gamma
from scipy.interpolate import RegularGridInterpolator

from ..Geometry.Grid import cubicspace


def Z(z):
    r"""Plasma Dispersion Function. See the module's documentation for details:
    :py:mod:`.PlasmaDispersionFunction`
    
    The Plasma Dispersion Function(PDF) is related to Faddeeva function as
    
    .. math::
       
      Z(z) = {\mathrm i}\sqrt{\pi} \; w(z) \; .
    """
    return 1j*sqrt(np.pi)*wofz(z)


def Z_1(z):
    """First derivative of Z
    See :py:mod:`.PlasmaDispersionFunction` for details
    """
    return -2*(1+z*Z(z))

def Z_2(z):
    """Shorthand for Z_m(z,2) function
    """
    return -2*(z*Z_1(z) + Z(z))

def Z_m(z, m):
    r"""m'th derivative of Z
    
    Recurrence relation is used to evaluate this function.
    See :py:mod:`.PlasmaDispersionFunction` for details:
    The recurrence relation is
    
    .. math::
        
        Z_m = -2zZ_{m-1} - 2(m-1)Z_{m-2}
        
    and the starting points are Z_0 and Z_1 evaluated by :py:func:`Z` and 
    :py:func:`Z_1` respectively.
    """
    
    assert (m >= 0)
    assert isinstance(m, int)
    if m == 0:
        return Z(z)
    elif m == 1:
        return Z_1(z)
    else:
        return -2*z*Z_m(z, m-1) -2*(m-1)*Z_m(z, m-2)
            

# General recurrence function to evaluate F_q for q>3/2

def Fq(phi, psi, nq, phi_nonzero=None, psi_nonzero=None, phi_tol=None, 
       psi_tol=None):
    r"""General function to evaluate :math:`\mathcal{F}_{q}(\phi,\psi)`
    
    For non-zero psi, we use the following recurrence relation to evaluate 

    .. math::
    
        \mathcal{F}_{q+2}(\phi,\psi) = 
        (1+\phi^2\mathcal{F}_q-q\mathcal{F}_{q+1})/\psi^2
    
    Special caution is required to evaluate Fq when psi=0, because the 
    recurrence relation has 0 in denominator. It is convenient to observe that
    the above recurrence relation then requires the numerator equals 0 as well.
    So we have the following recurrence relation
    
    .. math::
    
        \mathcal{F}_{q+1} = \frac{1+\phi^2\mathcal{F}_q}{q}
        
    Another function will be dedicated to this special case, :py:func:`Fq0`.
    
    Note: refer to [1]_, the sign convention for :math:`\phi` is :
        
    .. math::
        \mathrm{Re}\phi > 0 \; ,\; \mathrm{Im}\phi < 0
    
    :param phi: :math:`\phi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :type phi: ndarray of complex
    :param psi: :math:`\psi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :type psi: ndarray of complex
    :param int nq: the numerator in q, must be odd, the denominator is default  
                   to be 2
    :param bool phi_nonzero: True if phi != 0 is guaranteed everywhere. If not
                             given, phi will be tested and divided into 
                             appropriate groups.
    :param bool psi_nonzero: True if psi != 0 is guaranteed everywhere. If not
                             given, psi will be tested and divided into 
                             appropriate groups.
    :param float phi_tol: tolerance for testing phi=0 condition. If not given, 
                         will try to choose a proper value automatically based 
                         on nq and m.
    :param float psi_tol: tolerance for testing psi=0 condition. If not given, 
                         will try to choose a proper value automatically based 
                         on nq and m.
    
    :return: :math:`\mathcal{F}_{q}(\phi,\psi)` evaluated at given 
             :math:`\phi` and :math:`\psi` mesh
    :rtype: ndarray of complex 
    
     .. [1] Weakly relativistic dielectric tensor and dispersion functions of a 
           Maxwellian plasma, V. Krivenski and A. Orefice, J. Plasma Physics 
           (1983), vol. 30, part 1, pp. 125-131
    """
    phi = np.array(phi)
    psi = np.array(psi)
    if (phi_tol is None):
        phi_tol = 1e-4
    if (psi_tol is None):
        if nq >=3:
            psi_tol = 2*10**(-14.0/(nq-2))
        else:
            psi_tol = 1e-12
    assert np.array(phi).shape == np.array(psi).shape
    assert np.all(np.logical_or(np.abs(np.real(phi)) <= phi_tol ,\
                                np.abs(np.imag(phi)) <= phi_tol) )
    assert isinstance(nq, int) and nq>0 and nq%2 == 1
    
    if (psi_nonzero is None) and (phi_nonzero is None):
        psi_nonzero_idx = np.logical_or( np.abs(np.real(psi)) >= psi_tol, 
                            np.abs(np.imag(psi)) >= psi_tol)
        phi_nonzero_idx = np.logical_or( np.abs(np.real(phi)) >= phi_tol, 
                            np.abs(np.imag(phi)) >= phi_tol)
        # Now, we have 4 cases: 
        # case 1: (psi != 0) and (phi != 0)
        all_nonzero_idx = np.logical_and(psi_nonzero_idx, phi_nonzero_idx)
        # case 2: (psi == 0) and (phi != 0)
        psi_zero_idx = np.logical_and(np.logical_not(psi_nonzero_idx), 
                                      phi_nonzero_idx)
        # case 3: (psi != 0) and (phi == 0)
        phi_zero_idx = np.logical_and(psi_nonzero_idx,
                                      np.logical_not(phi_nonzero_idx))
        # case 4: (psi == 0) adn (phi == 0)
        all_zero_idx = np.logical_and(np.logical_not(psi_nonzero_idx),
                                      np.logical_not(phi_nonzero_idx))
        
        result = np.empty_like(phi, dtype='complex')
        
        # modify phi so that real(phi)>0 and imag(phi)<0
        phi_m = np.abs(np.real(phi)) - 1j*np.abs(np.imag(phi))
                                      
        # for case 1           
        phi1 = phi_m[all_nonzero_idx]
        psi1 = psi[all_nonzero_idx]
        result[all_nonzero_idx] = Fq(phi1, psi1, nq, True, True, phi_tol, 
                                     psi_tol)
        
        # for case 2
        phi2 = phi_m[psi_zero_idx]
        psi2 = psi[psi_zero_idx]
        result[psi_zero_idx] = Fq(phi2, psi2, nq, True, False, phi_tol,psi_tol)
        
        # for case 3
        phi3 = phi_m[phi_zero_idx]
        psi3 = psi[phi_zero_idx]
        result[phi_zero_idx] = Fq(phi3, psi3, nq, False, True, phi_tol,psi_tol)
        
        # for case 4
        phi4 = phi_m[all_zero_idx]
        psi4 = psi[all_zero_idx]
        result[all_zero_idx] = Fq(phi4, psi4, nq, False, False,phi_tol,psi_tol)
        
        return result
    else:
    
        if(nq == 1):
            return _F12(phi, psi, phi_nonzero, psi_nonzero)
        elif(nq == 3): 
            return _F32(phi, psi, phi_nonzero, psi_nonzero)
        #elif(nq == 5):
        #    return _F52(phi, psi, phi_nonzero, psi_nonzero)
        else:
            if(phi_nonzero and psi_nonzero):
            # if psi is already checked at high order function, no more checking
                return (1 + phi*phi*Fq(phi,psi,nq-4, True, True) - 
                        (nq-4)/2.0*Fq(phi,psi,nq-2, True, True)) / (psi*psi)
            elif (not psi_nonzero) and phi_nonzero:
                return (1+ phi*phi*Fq(phi,psi,nq-2, True, False))*2/(nq-2)
            elif (not phi_nonzero) and psi_nonzero:
                return (1 - (nq-4)/2.0*Fq(phi, psi, nq-2, False, True)) /\
                       (psi*psi)
            else:
                return 2.0/(nq-2)
              
    
    
def _F12(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler for :math:`\mathcal{F}_{1/2}(\phi,\psi)`
    
    Do not call directly. Use Fq(phi,psi,1) instead.
    """   
    if not phi_nonzero:
        warnings.warn('F12 enconters phi=0 input, it diverges at {} points. \
Check the data to see what\'s going on.'.format(len(phi)))
        return np.zeros_like(phi) + np.nan
    else:
        return -(Z(psi-phi) +Z(-psi-phi))/(2*phi)
        

def _F32(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler for :math:`\mathcal{F}_{3/2}(\phi,\psi)`
    
    Do not call directly. Use Fq(phi,psi,3) instead
    """       
    if psi_nonzero:
        return -(Z(psi-phi) - Z(-psi-phi)) / (2*psi)
    else:
        return -Z_1(-phi)

'''
def _F52(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler for :math:`\mathcal{F}_{5/2}(\phi,\psi)`
    
    Do not call directly. Use Fq(phi, psi, 5) instead
    """
    phi = np.array(phi)
    psi = np.array(psi)
    assert(phi.shape == psi.shape)    
    # since physically phi^2 is real, phi is either pure real or imaginary, 
    # test if this condition is satisfied.
    assert(np.logical_or(np.abs(np.real(phi))<=tol, 
                         np.abs(np.imag(phi))<=tol).all())

    result = np.empty_like(phi, dtype='complex')
    
    # First, we modify phi so that it complies with our requirement
    phi_mod = np.abs(np.real(phi)) - 1j*np.abs(np.imag(phi))
    # Now, we process zero psi part
    psi0_idx = np.logical_and(np.abs(np.real(psi)) <= tol, 
                              np.abs(np.imag(psi)) <= tol)
    result[psi0_idx] = 2*(1 + phi_mod[psi0_idx]*phi_mod[psi0_idx]*\
                              F32(phi[psi0_idx], psi[psi0_idx])) / 3 
    
    # Finally, we deal with phi==0 part and phi!=0 part
    nonzero_idx = np.logical_or(np.abs(np.real(phi)) > tol, 
                                np.abs(np.imag(phi)) > tol)
    
    zero_idx = np.logical_and( np.logical_not(nonzero_idx),
                              np.logical_not(psi0_idx))
    nonzero_idx = np.logical_and(nonzero_idx, np.logical_not(psi0_idx))
    
    if psi_nonzero:
        return (1 + phi*phi* \
                           _F12(phi, psi) - 
                           0.5*F32(phi[nonzero_idx], psi[nonzero_idx])) / \
                                    (psi[nonzero_idx]*psi[nonzero_idx])
    result[zero_idx] =  (1 - 0.5*F32(phi[zero_idx], psi[zero_idx])) / \
                          psi[zero_idx]**2
    return result
'''    
  

def Fmq(phi, psi, m, nq, phi_nonzero=None, 
        psi_nonzero=None, phi_tol=None, psi_tol=None):
    r"""General function to evaluate m-th derivative of Fq respect to phi^2
    
    For each :math:`m`, starting from lowest two :math:`q` values , we use the    
    following recurrence relation to calculate larger :math:`q`'s. 
    
    .. math::
        
        \mathcal{F}_{q+2}^m = (\phi^2\mathcal{F}_q^m - q\mathcal{F}_{q+1}^m +
            m\mathcal{F}_q^{m-1})/\psi^2
            
    For :math:`\psi = 0` case, it is not useful, we use instead:
    
    .. math::
        
        \mathcal{F}_{q+1}^m = (\phi^2\mathcal{F}_q^m + m\mathcal{F}_q^{m-1})/q
        
    
    Further more, if :math:`\phi = 0` at the same time as :math:`\psi=0`, we
    have:
    
    .. math::
    
        \mathcal{F}^m_{q+3/2} = \frac{ m\mathcal{F}^{m-1}_{q+1/2} }{ q+1/2 }
    
    Note that in physical situations, ``m``>(``nq``-1)/2 is not used. So the 
    recurrence starts at ``nq`` = 2* ``m`` +1 and 2* ``m`` +3.
    
    Here we implement only m=1,2,3,4 cases, using formula given in [1]_. Higher
    order cases required analytical derivation of starting formula.
    
    :param phi: :math:`\phi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype phi: ndarray of complex
    :param psi: :math:`\psi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype psi: ndarray of complex
    :param int nq: the numerator in q, must be odd, the denominator is default  
                   to be 2
    :param bool phi_nonzero: True if phi != 0 is guaranteed everywhere. If not
                             given, phi will be tested and divided into 
                             appropriate groups.
    :param bool psi_nonzero: True if psi != 0 is guaranteed everywhere. If not
                             given, psi will be tested and divided into 
                             appropriate groups.
    :param float phi_tol: tolerance for testing phi=0 condition. If not given, 
                         will try to choose a proper value automatically based 
                         on nq and m.
    :param float psi_tol: tolerance for testing psi=0 condition. If not given, 
                         will try to choose a proper value automatically based 
                         on nq and m.
    
    :return: :math:`\mathcal{F}^m_{q}(\phi,\psi)` evaluated at given 
             :math:`\phi` and :math:`\psi` mesh
    :rtype: ndarray of complex 
    
    .. [1] I.P.Shkarofsky, "New representations of dielectric tensor elements 
           in magnetized plasma", J. Plasma Physics(1986), vol. 35, part 2, pp. 
           319-331
        
    """
    phi = np.array(phi)
    psi = np.array(psi)
    assert np.array(phi).shape == np.array(psi).shape    
    assert isinstance(m, int) and (m >= 0)   
    assert isinstance(nq, int) and (nq > 0) and (nq%2 == 1)
    assert (nq >= 2*m+1) # required for physically meaningful result
    
    if (phi_tol is None):
        phi_tol = 1e-4
    if (psi_tol is None):
        if nq >=3:
            psi_tol = 2*10**(-14.0/(nq-2))
        else:
            psi_tol = 1e-12
    
    if (psi_nonzero is None) and (phi_nonzero is None):
        psi_nonzero_idx = np.logical_or( np.abs(np.real(psi)) >= psi_tol, 
                            np.abs(np.imag(psi)) >= psi_tol)
        phi_nonzero_idx = np.logical_or( np.abs(np.real(phi)) >= phi_tol, 
                            np.abs(np.imag(phi)) >= phi_tol)
        # Now, we have 4 cases: 
        # case 1: (psi != 0) and (phi != 0)
        all_nonzero_idx = np.logical_and(psi_nonzero_idx, phi_nonzero_idx)
        # case 2: (psi == 0) and (phi != 0)
        psi_zero_idx = np.logical_and(np.logical_not(psi_nonzero_idx), 
                                      phi_nonzero_idx)
        # case 3: (psi != 0) and (phi == 0)
        phi_zero_idx = np.logical_and(psi_nonzero_idx,
                                      np.logical_not(phi_nonzero_idx))
        # case 4: (psi == 0) adn (phi == 0)
        all_zero_idx = np.logical_and(np.logical_not(psi_nonzero_idx),
                                      np.logical_not(phi_nonzero_idx))
        
        result = np.empty_like(phi, dtype='complex')
        
        # modify phi so that real(phi)>0 and imag(phi)<0
        phi_m = np.abs(np.real(phi)) - 1j*np.abs(np.imag(phi))
                                      
        # for case 1           
        phi1 = phi_m[all_nonzero_idx]
        psi1 = psi[all_nonzero_idx]
        result[all_nonzero_idx] = Fmq(phi1, psi1, m, nq, True, True)
        
        # for case 2
        phi2 = phi_m[psi_zero_idx]
        psi2 = psi[psi_zero_idx]
        result[psi_zero_idx] = Fmq(phi2, psi2, m, nq, True, False)
        
        # for case 3
        phi3 = phi_m[phi_zero_idx]
        psi3 = psi[phi_zero_idx]
        result[phi_zero_idx] = Fmq(phi3, psi3, m, nq, False, True)
        
        # for case 4
        phi4 = phi_m[all_zero_idx]
        psi4 = psi[all_zero_idx]
        result[all_zero_idx] = Fmq(phi4, psi4, m, nq, False, False)
        
        return result
    else:
        if (m == 0):
            warnings.warn('0-th derivative is encountered. Try use Fq directly\
             if possible.')
            return Fq(phi, psi, nq, phi_nonzero, psi_nonzero)
        elif (m == 1):
            return _Fq_1(phi, psi, nq, phi_nonzero, psi_nonzero)
        elif (m == 2):
            return _Fq_2(phi, psi, nq, phi_nonzero, psi_nonzero)
        elif (m == 3):
            return _Fq_3(phi, psi, nq, phi_nonzero, psi_nonzero)
        elif (m == 4):
            return _Fq_4(phi, psi, nq, phi_nonzero, psi_nonzero)
                
        else: # m>4 cases are not implemented for now. 
            raise ValueError('m={} is encountered. m>4 cases are not \
implemented for now. Please submit a request to shilei8583@gmail.com if this \
feature is needed.'.format(m))
        
        
def _Fq_1(phi, psi, nq, phi_nonzero, psi_nonzero):
    r"""Handler for :py:func:`Fmq` function when m == 1. 

    Calling this function directly is not recommended. Parameter validity is 
    not checked.
    
    Call :py:func`Fmq` with m=1 instead.
    """
    if (nq == 3):
        return _F32_1(phi, psi, phi_nonzero, psi_nonzero)
    elif (nq == 5):
        return _F52_1(phi, psi, phi_nonzero, psi_nonzero)
    else:
        if psi_nonzero and phi_nonzero:
            return (phi*phi*_Fq_1(phi, psi, nq-4, True, True) - \
                   (nq-4)/2.*_Fq_1(phi, psi, nq-2, True, True) + \
                    Fq(phi, psi, nq-4, True, True)) /\
                    (psi*psi)
        elif psi_nonzero and (not phi_nonzero):
            return (-(nq-4)/2.*_Fq_1(phi, psi, nq-2, False, True) + \
                    Fq(phi, psi, nq-4, False, True)) /\
                    (psi*psi)
        elif phi_nonzero and (not psi_nonzero):
            return (phi*phi*_Fq_1(phi, psi, nq-2, True, False) + \
                   Fq(phi, psi, nq-2, True, False)) *2 \
                   / (nq-2)
        else:
            return Fq(phi, psi, nq-2, False, False)*2/(nq-2)
        

def _F32_1(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}'_{3/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=1, nq=3 instead.
   
    """
    if not phi_nonzero:
        warnings.warn('zero phi encountered in F32_1, divergence occurs. Check\
input to make sure this is not an error.')
        return np.ones_like(phi)*np.nan
    elif psi_nonzero and phi_nonzero:
        return (Z_1(psi-phi)-Z_1(-psi-phi))/(4*psi*phi)
    else:
        return Z_m(-phi, 2)/ (2*phi)

                                                     
def _F52_1(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}'_{5/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=1, nq=3 instead.
   
    """
    if psi_nonzero:
        psi3 = psi*psi*psi
        plus = psi - phi
        minus = -psi - phi
        return -(Z(plus) - psi*Z_1(plus)) / (4*psi3) + \
               (Z(minus) + psi*Z_1(minus)) / (4*psi3)
    elif phi_nonzero:
        return Z_m(-phi, 3)/6
    else:
        return 4./3 


def _Fq_2(phi, psi, nq, phi_nonzero, psi_nonzero):
    r"""Handler for :py:func:`Fmq` function when m == 2. 

    Calling this function directly is not recommended. Parameter validity is 
    not checked.
    
    Call :py:func`Fmq` with m=2 instead.
    """
    if (nq == 5):
        return _F52_2(phi, psi, phi_nonzero, psi_nonzero)
    elif (nq == 7):
        return _F72_2(phi, psi, phi_nonzero, psi_nonzero)
    else:
        if psi_nonzero and phi_nonzero:
            return (phi*phi*_Fq_2(phi, psi, nq-4, True, True) - \
                   (nq-4)/2.*_Fq_2(phi, psi, nq-2, True, True) + \
                    2*_Fq_1(phi, psi, nq-4, True, True)) / (psi*psi)
        elif psi_nonzero and (not phi_nonzero):
            return (-(nq-4)/2.*_Fq_2(phi, psi, nq-2, False, True) + \
                    2*_Fq_1(phi, psi, nq-4, False, True)) / (psi*psi)
        elif phi_nonzero:
            return (phi*phi*_Fq_2(phi, psi, nq-2, True, False) + \
                    2* _Fq_1(phi, psi, nq-2, True, False)) *2 / (nq-2)
        else:
            return 2*_Fq_1(phi, psi, nq-2, False, False)*2/(nq-2)


def _F52_2(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}''_{5/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=2, nq=5 instead.
   
    """
    if not phi_nonzero:
        warnings.warn('zero phi encountered in F52_2, divergence occurs. Check\
input to make sure this is not an error.')
        return np.ones_like(phi)*np.nan
    elif psi_nonzero and phi_nonzero:
        plus = psi - phi
        minus = -psi - phi
        return ((Z_1(plus) - psi*Z_m(plus, 2)) - (Z_1(minus) + psi*Z_m(minus, 
                 2))) / (8*phi*psi*psi*psi)
    else:
        return -Z_m(-phi, 4) / (12*phi)

                                                      
def _F72_2(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}''_{7/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=2, nq=7 instead.
   
    """
    if psi_nonzero:
        psi2 = psi*psi
        psi5 = psi2*psi2*psi
        plus = psi - phi
        minus = -psi - phi
        return -(3*Z(plus) - 3*psi*Z_1(plus) + psi2*Z_m(plus, 2)) / (8*psi5) +\
               (3*Z(minus) + 3*psi*Z_1(minus) + psi2*Z_m(minus, 2)) / (8*psi5)
    elif phi_nonzero:
        return - Z_m(-phi, 5)/60
    else:
        return 16./15
        
def _Fq_3(phi, psi, nq, phi_nonzero, psi_nonzero):
    r"""Handler for :py:func:`Fmq` function when m == 3. 

    Calling this function directly is not recommended. Parameter validity is 
    not checked.
    
    Call :py:func`Fmq` with m=3 instead.
    """
    if (nq == 7):
        return _F72_3(phi, psi, phi_nonzero, psi_nonzero)
    elif (nq == 9):
        return _F92_3(phi, psi, phi_nonzero, psi_nonzero)
    else:
        if psi_nonzero and phi_nonzero:
            return (phi*phi*_Fq_3(phi, psi, nq-4, True, True) - \
                   (nq-4)/2.*_Fq_3(phi, psi, nq-2, True, True) + \
                    3*_Fq_2(phi, psi, nq-4, True, True)) / (psi*psi)
        elif psi_nonzero and (not phi_nonzero):
            return (-(nq-4)/2.*_Fq_3(phi, psi, nq-2,False, True) + \
                    3*_Fq_2(phi, psi, nq-4,False, True)) / (psi*psi)
        elif phi_nonzero:
            return (phi*phi*_Fq_3(phi, psi, nq-2, True, False) + \
                    3* _Fq_2(phi, psi, nq-2, True, False)) *2 / (nq-2)
        else:
            return 3*_Fq_2(phi, psi, nq-2, False, False)*2/(nq-2)


def _F72_3(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}'''_{7/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=3, nq=7 instead.
   
    """
    if not phi_nonzero:
        warnings.warn('zero phi encountered in F72_3, divergence occurs. Check\
input to make sure this is not an error.')
        return np.ones_like(phi)*np.nan
    elif psi_nonzero and phi_nonzero:
        plus = psi - phi
        minus = -psi - phi
        psi2 = psi * psi
        psi5 = psi2 * psi2 * psi
        return ((3*Z_1(plus) - 3*psi*Z_2(plus)+ psi2*Z_m(plus, 3)) - \
                (3*Z_1(minus) + 3*psi*Z_2(minus) + psi2*Z_m(minus, 3))) \
                / (16*phi*psi5)
    else:
        return Z_m(-phi, 6) / (120*phi)

                                                      
def _F92_3(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}'''_{9/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=3, nq=9 instead.
   
    """
    if psi_nonzero:
        psi2 = psi*psi
        psi3 = psi2*psi
        psi7 = psi2*psi2*psi3
        plus = psi - phi
        minus = -psi - phi
        return -(15*Z(plus) - 15*psi*Z_1(plus) + 6*psi2*Z_2(plus) - \
                 psi3*Z_m(plus, 3)) / (16*psi7) +\
               (15*Z(minus) + 15*psi*Z_1(minus) + 6*psi2*Z_2(minus) + \
                 psi3*Z_m(minus, 3)) / (16*psi7)
    elif phi_nonzero:
        return - Z_m(-phi, 7)/840
    else:
        return 96/105.


def _Fq_4(phi, psi, nq, phi_nonzero, psi_nonzero):
    r"""Handler for :py:func:`Fmq` function when m == 4. 

    Calling this function directly is not recommended. Parameter validity is 
    not checked.
    
    Call :py:func`Fmq` with m=3 instead.
    """
    if (nq == 9):
        return _F92_4(phi, psi, phi_nonzero, psi_nonzero)
    elif (nq == 11):
        return _F112_4(phi, psi, phi_nonzero, psi_nonzero)
    else:
        if psi_nonzero and phi_nonzero:
            return (phi*phi*_Fq_4(phi, psi, nq-4, True, True) - \
                   (nq-4)/2.*_Fq_4(phi, psi, nq-2, True, True) + \
                    4*_Fq_3(phi, psi, nq-4, True, True)) / (psi*psi)
        elif psi_nonzero and (not phi_nonzero):
            return (-(nq-4)/2.*_Fq_4(phi, psi, nq-2,False, True) + \
                    4*_Fq_3(phi, psi, nq-4, False, True)) / (psi*psi)
        elif phi_nonzero:
            return (phi*phi*_Fq_4(phi, psi, nq-2,  True, False ) + \
                    4* _Fq_3(phi, psi, nq-2,  True, False)) *2 / (nq-2)
        else:
            return 4*_Fq_3(phi, psi, nq-2, False, False)*2/(nq-2)


def _F92_4(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}^{IV}_{9/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=4, nq=9 instead.
   
    """
    if not phi_nonzero:
        warnings.warn('zero phi encountered in F92_4, divergence occurs. Check\
input to make sure this is not an error.')
        return np.ones_like(phi)*np.nan
    elif psi_nonzero and phi_nonzero:
        plus = psi - phi
        minus = -psi - phi
        psi2 = psi * psi
        psi3 = psi * psi2
        psi7 = psi2 * psi2 * psi3
        return ((15*Z_1(plus) - 15*psi*Z_2(plus) + 6*psi2*Z_m(plus, 3) - \
                 psi3*Z_m(plus, 4)) - \
                (15*Z_1(minus) + 15*psi*Z_2(minus) + 6*psi2*Z_m(minus, 3) + \
                 psi3*Z_m(plus, 4)) ) / (32*phi*psi7)
    else:
        return -Z_m(-phi, 8) / (1680*phi)

                                                      
def _F112_4(phi, psi, phi_nonzero, psi_nonzero):
    r"""Handler function for :math:`\mathcal{F}^{IV}_{11/2}(\phi,\psi)`
    
    Do not call directly. Parameter validity not checked. Use :py:func:`Fmq` 
    with m=4, nq=11 instead.
   
    """
    if psi_nonzero:
        psi2 = psi*psi
        psi3 = psi2*psi
        psi4 = psi2*psi2
        psi9 = psi2*psi3*psi4
        plus = psi - phi
        minus = -psi - phi
        return (-(105*Z(plus) - 105*psi*Z_1(plus) + 45*psi2*Z_2(plus) - \
                 10*psi3*Z_m(plus, 3) + psi4*Z_m(plus, 4)) +\
               (105*Z(minus) + 105*psi*Z_1(minus) + 45*psi2*Z_2(minus) + \
                 10*psi3*Z_m(minus, 3) + psi4*Z_m(minus, 4))) / (32*psi9)
    elif phi_nonzero:
        return - Z_m(-phi, 9)/15120
    else:
        return 96*8/(105.*9)


def _Fm_mp32_00(m, shape=(1)):
    r"""Handler for :math:`\mathcal{F}^m_{m+3/2}(0,0)`
    
    when :math:`\psi=0` and :math:`\phi=0`, we have the recurrence
    
    .. math::
    
        \mathcal{F}^m_{q+3/2} = \frac{ m\mathcal{F}^{m-1}_{q+1/2} }{ q+1/2 }
        
    especially when q == m, this recurrence finally ends at 
    :math:`\mathcal{F}_{3/2} = 2`. 
    
    We can then get the analycial formula for 
    :math:`\mathcal{F}^m_{m+3/2}(0,0)`:
    
    .. math::
    
        \mathcal{F}^m_{m+3/2}(0,0) = 2 \prod\limits_{i=1}^m 2i/(2i+1) = 
        \frac{m! \; 2^{m+1}}{(2m+1)!!}
        
    :param int m: the order of the derivative.
    :param shape: the shape of the return array
    :type shape: tuple of int, should be the same shape as phi/psi determined
                 by the caller
                 
    :return: the calculated value of the function
    :rtype: ndarray of complex with the shape same as ``shape``
    
    """
    
    result = 2.
    while(m > 0):
        result *= 2*m/(2*m+1.)
        m = m-1
    return np.ones(shape, dtype='complex')*result
    

# We define some recommended mesh for creation of fast evaluators of Fq and Fmq

mudelta_mesh = cubicspace(-50, 50, 2047)
psi_mesh = cubicspace(-30, 30, 1023)


class FqFastEvaluator(object):
    """Fast evaluator for Fq functions
    
    Initialization:
        FqFastEvaluator( nq, phi_sq_mesh, psi_mesh, **P)
        
        :param int nq: nq passed into Fq function, the order of the function is
                       nq/2.
        :param phi_sq_mesh: :math:`\phi^2` values for mesh points in phi_psi 
                            plane, we use :math:`\phi^2` because it's real.
        :type phi_mesh: 1D array of float, monotonic order 
        :param psi_mesh: psi values for mesh points in phi_psi plane
        :type psi_mesh: 1D array of float, monotonic order
        :param **P: additional keyword arguments passed into 
                    scipy.interpolate.RegularGridInteropolator.
                          
    Methods:
    
        __call__(phi, psi):
            return Fq value at (phi,psi) points. phi, psi are arrays with the 
            same shape.
        
        reconstruct(**P):
            reconstruct the interpolator using the new keyword arguments given 
            in **P
            
        test(phi_test, psi_test, abserr=1e-6, relerr=1e-4):
            evaluate Fq on (phi_test, psi_test) points using both original 
            function and interpolator, report the maximum absolute error and 
            relative error. Print a warning if either error is greater than the
            preset margin given by abserr and relerr arguments.
            
    """

    def __init__(self, nq, mudelta_mesh=mudelta_mesh, psi_mesh=psi_mesh, **P):
        self.psi_1D = psi_mesh        
        self.mudelta_1D = mudelta_mesh
        self.nq = nq
        
        mudelta_2D = np.zeros((len(mudelta_mesh), len(psi_mesh))) + \
                      mudelta_mesh[:, np.newaxis]
        psi_2D = np.zeros_like(mudelta_2D) + psi_mesh[np.newaxis, :]
        self.value = Fq(sqrt(psi_2D*psi_2D-mudelta_2D), psi_2D, self.nq)
        self.interpolator = RegularGridInterpolator((self.mudelta_1D, 
                                                        self.psi_1D), 
                                                        self.value, 
                                                        bounds_error=False, 
                                                        fill_value=0, **P)
        self.phi_bounds = ((sqrt(psi_mesh[0]-mudelta_mesh[0]), 
                            sqrt(psi_mesh[0]-mudelta_mesh[-1])), 
                           (sqrt(psi_mesh[-1]-mudelta_mesh[0]), 
                            sqrt(psi_mesh[-1]-mudelta_mesh[-1])) )
        self.psi_bounds = (psi_mesh[0], psi_mesh[-1])
    
    
    def reconstruct(self, mudelta_mesh=None, psi_mesh=None, **P):
        """reconstruct the interpolator using the new keyword arguments given 
        in **P and/or mudelta_mesh, psi_mesh
        """
        if (psi_mesh is not None):
            self.psi_1D = psi_mesh
        if (mudelta_mesh is not None):
            self.mudelta_1D = mudelta_mesh
            
        if (psi_mesh is not None) or (mudelta_mesh is not None):
            mudelta_2D = np.zeros((len(mudelta_mesh), len(psi_mesh))) + \
                          mudelta_mesh[:, np.newaxis]
            psi_2D = np.zeros_like(mudelta_2D) + psi_mesh[np.newaxis, :]
            self.value = Fq(sqrt(psi_2D*psi_2D-mudelta_2D), psi_2D, self.nq)
            self.interpolator = RegularGridInterpolator((self.mudelta_1D, 
                                                         self.psi_1D), 
                                                         self.value, 
                                                         bounds_error=False, 
                                                         fill_value=0, **P)
        else:
            self.interpolator = RegularGridInterpolator((self.mudelta_1D, 
                                                         self.psi_1D), 
                                                        self.values, 
                                                        bounds_error=False, 
                                                        fill_value=0, **P)
        
    def __call__(self, phi, psi):
        """Evaluate Fq at phi,psi using the internal interpolator
        """
        phi = np.array(phi)
        psi = np.array(psi)
        assert phi.shape == psi.shape
        # phi must be square root of a real number 
        assert np.all(np.logical_or(np.abs(np.real(phi)) <= 1e-10, 
                             np.abs(np.imag(phi)) <= 1e-10))
        phi2 = np.real(phi*phi)
        mudelta = psi*psi - phi2
        dims = range(1, mudelta.ndim+1)
        dims.extend([0])
        
        # construct the points structure for interpolation, transpose the array
        # so that the fastest changing index is length 2: (phi,psi)
        points = np.transpose(np.array([mudelta, psi]), axes=dims)
        
        return self.interpolator(points)
        
    def test(self, phi, psi, tolabs=1e-3, tolrel=5e-2, full_report=False):
        """evaluate Fq on (phi_test, psi_test) points using both original 
        function and interpolator, report the maximum absolute error and 
        relative error. Print a warning if either error is greater than the
        preset margin given by abserr and relerr arguments.
        
        If full_report == True, abserr and relerr on every phi,psi point will 
        be returned. Otherwise only the maximum value and corresponding phi,psi
        are returned.
        """
        
        exact_value = Fq(phi, psi, self.nq)
        interp_value = self(phi,psi)
        
        abs_err = interp_value - exact_value
        
        in_range_idx = interp_value != 0
        rel_err = np.zeros_like(abs_err)
        rel_err[in_range_idx] = abs_err[in_range_idx]/exact_value[in_range_idx]
        
        
        maxabs = np.abs(abs_err).max()
        maxrel = np.abs(rel_err).max()        
        
        arg_maxabs = np.where(np.abs(abs_err) == maxabs)
        arg_maxrel = np.where(np.abs(rel_err) == maxrel)
        
        if(maxabs > tolabs):
            warnings.warn('Absolute error exceeds limit({})'.format(tolabs))
        if(maxrel > tolrel):
            warnings.warn('Relative error exceeds limit({})'.format(tolrel))
        
        print '\
Max Absolute Error: {}, at\n\
    phi:{},\n\
    psi:{}.\n\
Max Relative Error: {}, at\n\
    phi:{},\n\
    psi:{}'.format(abs_err[arg_maxabs], phi[arg_maxabs], psi[arg_maxabs],
                   rel_err[arg_maxrel], phi[arg_maxrel], psi[arg_maxrel])         
        
        if full_report:
            return (abs_err, rel_err)


class FmqFastEvaluator(object):
    """Fast evaluator for Fmq functions
    
    Initialization:
        FqFastEvaluator(m, nq, mudelta_mesh, psi_mesh, **P)
        
        :param int m: m passed into Fmq function, the order of differentiation.
        :param int nq: nq passed into Fq function, the order of the function is
                       nq/2. 
        :param mudelta_mesh: :math:`\mu \delta \equiv \psi^2-\phi^2` values for
                             mesh points in phi_psi plane, we use this value 
                             because Fmq is most sensitive to it.
        :type mudelta_mesh: 1D array of float, monotonic order 
        :param psi_mesh: psi values for mesh points in phi_psi plane
        :type psi_mesh: 1D array of float, monotonic order       
        :param **P: additional keyword arguments passed into 
                    scipy.interpolate.RegularGridInteropolator.
                          
    Methods:
    
        __call__(phi, psi):
            return Fq value at (phi,psi) points. phi, psi are arrays with the 
            same shape.
        
        reconstruct(mudelta_mesh=None, psi_mesh=None, **P):
            reconstruct the interpolator using the new keyword arguments given 
            in **P and/or new meshes.
            
        test(phi_test, psi_test, abserr=1e-6, relerr=1e-4):
            evaluate Fmq on (phi_test, psi_test) points using both original 
            function and interpolator, report the maximum absolute error and 
            relative error. Print a warning if either error is greater than the
            preset margin given by abserr and relerr arguments.
            
    """

    def __init__(self, m, nq, mudelta_mesh=mudelta_mesh, psi_mesh=psi_mesh, 
                 **P):
        self.psi_1D = psi_mesh        
        
        self.mudelta_1D = mudelta_mesh
        self.m = m
        self.nq = nq
        mudelta_2D = np.zeros((len(mudelta_mesh), len(psi_mesh))) + \
                      mudelta_mesh[:, np.newaxis]
        psi_2D = np.zeros_like(mudelta_2D) + psi_mesh[np.newaxis, :]
        self.value = Fmq(sqrt(psi_2D*psi_2D-mudelta_2D), psi_2D, self.m, 
                         self.nq)
        self.interpolator = RegularGridInterpolator((self.mudelta_1D, 
                                                        self.psi_1D), 
                                                        self.value, 
                                                        bounds_error=False, 
                                                        fill_value=0, **P)
        self.phi_bounds = ((sqrt(psi_mesh[0]-mudelta_mesh[0]), 
                            sqrt(psi_mesh[0]-mudelta_mesh[-1])), 
                           (sqrt(psi_mesh[-1]-mudelta_mesh[0]), 
                            sqrt(psi_mesh[-1]-mudelta_mesh[-1])) )
        self.psi_bounds = (psi_mesh[0], psi_mesh[-1])
    
    
    def reconstruct(self, mudelta_mesh=None, psi_mesh=None, **P):
        """reconstruct the interpolator using the new keyword arguments given 
        in **P and/or mudelta_mesh, psi_mesh
        """
        if (psi_mesh is not None):
            self.psi_1D = psi_mesh
        if (mudelta_mesh is not None):
            self.mudelta_1D = mudelta_mesh
            
        if (psi_mesh is not None) or (mudelta_mesh is not None):
            mudelta_2D = np.zeros((len(mudelta_mesh), len(psi_mesh))) + \
                          mudelta_mesh[:, np.newaxis]
            psi_2D = np.zeros_like(mudelta_2D) + psi_mesh[np.newaxis, :]
            self.value = Fmq(sqrt(psi_2D*psi_2D-mudelta_2D), psi_2D, self.m, 
                            self.nq)
            self.interpolator = RegularGridInterpolator((self.mudelta_1D, 
                                                         self.psi_1D), 
                                                         self.value, 
                                                         bounds_error=False, 
                                                         fill_value=0, **P)
        else:
            self.interpolator = RegularGridInterpolator((self.mudelta_1D, 
                                                         self.psi_1D), 
                                                        self.values, 
                                                        bounds_error=False, 
                                                        fill_value=0, **P)
        
    def __call__(self, phi, psi):
        """Evaluate Fq at phi,psi using the internal interpolator
        """
        phi = np.array(phi)
        psi = np.array(psi)
        assert phi.shape == psi.shape
        # phi must be square root of a real number 
        assert np.all(np.logical_or(np.abs(np.real(phi)) <= 1e-10, 
                             np.abs(np.imag(phi)) <= 1e-10))
        phi2 = np.real(phi*phi)
        mudelta = psi*psi - phi2
        dims = range(1, mudelta.ndim+1)
        dims.extend([0])
        
        # construct the points structure for interpolation, transpose the array
        # so that the fastest changing index is length 2: (phi,psi)
        points = np.transpose(np.array([mudelta, psi]), axes=dims)
        
        return self.interpolator(points)
        
    def test(self, phi, psi, tolabs=1e-3, tolrel=1e-3, full_report=False):
        """evaluate Fmq on (phi_test, psi_test) points using both original 
        function and interpolator, report the maximum absolute error and 
        relative error. Print a warning if either error is greater than the
        preset margin given by abserr and relerr arguments.
        
        If full_report == True, abserr and relerr on every phi,psi point will 
        be returned. Otherwise only the maximum value and corresponding phi,psi
        are returned.
        """
        
        exact_value = Fmq(phi, psi, self.m, self.nq)
        interp_value = self(phi,psi)
        
        abs_err = interp_value - exact_value
        
        in_range_idx = interp_value != 0
        rel_err = np.zeros_like(abs_err)
        rel_err[in_range_idx] = abs_err[in_range_idx]/exact_value[in_range_idx]
        
        
        maxabs = np.abs(abs_err).max()
        maxrel = np.abs(rel_err).max()        
        
        arg_maxabs = np.where(np.abs(abs_err) == maxabs)
        arg_maxrel = np.where(np.abs(rel_err) == maxrel)
        
        if(maxabs > tolabs):
            warnings.warn('Absolute error exceeds limit({})'.format(tolabs))
        if(maxrel > tolrel):
            warnings.warn('Relative error exceeds limit({})'.format(tolrel))
        
        print '\
Max Absolute Error: {}, at\n\
    phi:{},\n\
    psi:{}.\n\
Max Relative Error: {}, at\n\
    phi:{},\n\
    psi:{}'.format(abs_err[arg_maxabs], phi[arg_maxabs], psi[arg_maxabs],
                   rel_err[arg_maxrel], phi[arg_maxrel], psi[arg_maxrel])         
        
        if full_report:
            return (abs_err, rel_err)
            
            
# We add the useful a_pn function here, since it's used in expressing weakly 
# relativistic tensor elements. The definition of a_pn can be find in Ref[2]
# in this module's docstring
    
def a_pn(p, n):
    r"""Evaluate :math:`a_{pn}` factor given in [1]_. 
    
    .. math::
        a_{pn} = (-1)^p (n+p-\frac{1}{2})! / [ p! (n+\frac{1}{2}p)!
                                            (n+\frac{1}{2}p-\frac{1}{2})! 2^n]
                                            
    .. [1] I.P.Shkarofsky, "New representations of dielectric tensor elements 
           in magnetized plasma", J. Plasma Physics(1986), vol. 35, part 2, pp. 
           319-331
    """
    
    assert isinstance(p, int)
    assert isinstance(n, int)
    
    if (p%2==0):
        sign = 1
    else:
        sign = -1
    
    return sign * gamma(n + p + 0.5)/(gamma(p + 1) * gamma(n + 0.5*p + 1) * \
                  gamma(n + 0.5*p + 0.5) * 2**n)








    
    