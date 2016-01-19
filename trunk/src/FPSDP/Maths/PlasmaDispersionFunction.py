r"""
This module provides ways to evaluate the Plasma Dispersion Function [1]_, 
:math:`Z(x)`, and other related functions, specifically, the 
:math:`\mathcal{F}_q(\phi,\psi)` Function [2]_. 

The Faddeeva function 

.. math::
   
  w(z) \equiv \exp(-z^2) \; {\mathrm {erfc}}(-{\mathrm i}z)
    
is used, where :math:`{\mathrm {erfc}}(z)` is the complementary error function. 
It is evaluated using the python wrapper of Steven G. Johnson's routine, 
provided by scipy, see :py:func:`scipy.spetial.wofz` for more details.

The Plasma Dispersion Function(PDF) is related to Faddeeva function as

.. math::
   
  Z(z) = {\mathrm i}\sqrt{\pi} \; w(z) \; ,
    
and :math:`\mathcal{F}_q` function is related to PDF as [2]_:

.. math:: 
   
  \mathcal{F}_{\frac{1}{2}}(\phi,\psi) = 
   -\frac{1}{2\phi}[Z(\psi-\phi)+Z(-\psi-\phi)] \; ,
   
  \mathcal{F}_{\frac{3}{2}}(\phi,\psi) = 
   -\frac{1}{2\psi}[Z(\psi-\phi)-Z(-\psi-\phi)] \; ,

  \mathcal{F}_{q+2}(\phi,\psi) = 
   (1+\phi^2\mathcal{F}_q-q\mathcal{F}_{q+1})/\psi^2 \; .
   
The derivatives of :math:`\mathcal{F}_q` respect to :math:`\phi^2` can be 
evaluated as:

.. math::
    
  \mathcal{F}_q^m \equiv \frac{\partial^m \mathcal{F}_q}{\partial(\phi^2)^m} 
  = \mathcal{F}_{q-1}^{m-1} - \mathcal{F}_q^{m-1} \; ,
    
  \mathcal{F}_{q+2}^m =
  (\phi^2\mathcal{F}_q^m - q\mathcal{F}_{q+1}^m + m\mathcal{F}_q^{m-1})/\psi^2

However, as pointed out in [2]_, evaluating derivatives using the first formula 
may suffer from the cancellation of two large numbers. A more reliable way is 
to express the derivatives of :math:`\mathcal{F}_{1/2}` and 
:math:`\mathcal{F}_{3/2}` in terms of derivatives of the PDF, and then use the 
second formula to evaluate larger q's.

PDF has the following property [1]_:

.. math::
   Z'(z) = -2(1+zZ(z)) \; ,

and it's easy to show the following recurrence relation 

.. math::
   Z^m(z) = -2[(m-1)Z^{m-2}(z) + zZ^{m-1}(z)] \quad \mathrm{for}\; m>2 \; .
   
Fianlly, for special case, :math:`\psi=0`, L'Hopital rule needs to be used to
evaluate the "0/0" kind expressions. More details in Appendix part of [2]_.

.. [1] https://farside.ph.utexas.edu/teaching/plasma/lectures1/node87.html

.. [2] I.P.Shkarofsky, "New representations of dielectric tensor elements in 
       magnetized plasma", J. Plasma Physics(1986), vol. 35, part 2, pp. 
       319-331

"""

from math import sqrt
import warnings

import numpy as np
from scipy.special import wofz


def Z(z):
    r"""Plasma Dispersion Function. See the module's documentation for details:
    :py:mod:`.PlasmaDispersionFunction`
    
    The Plasma Dispersion Function(PDF) is related to Faddeeva function as
    
    .. math::
       
      Z(z) = {\mathrm i}\sqrt{\pi} \; w(z) \; ,
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

def Fq(phi, psi, nq, tol=1e-14, no_psi_zero = False):
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
    
    :param phi: :math:`\phi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype phi: ndarray of complex
    :param psi: :math:`\psi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype psi: ndarray of complex
    :param int nq: the numerator in q, must be odd, the denominator is default  
                   to be 2
    :param float tol: tolerance for testing phi=0 condition
    
    :return: :math:`\mathcal{F}_{q}(\phi,\psi)` evaluated at given 
             :math:`\phi` and :math:`\psi` mesh
    :rtype: ndarray of complex 
    """
    
    assert np.array(phi).shape == np.array(psi).shape
    assert isinstance(nq, int) and nq>0 and nq%2 == 1
    
    if(nq == 1):
        return F12(phi,psi,tol)
    elif(nq == 3): 
        return F32(phi,psi,tol)
    elif(nq == 5):
        return F52(phi,psi,tol)
    else:
        if(no_psi_zero):
        # if psi is already checked at high order function, no more checking
            return (1 + phi*phi*Fq(phi,psi,nq-4, tol,True) - 
                    (nq-4)/2.*Fq(phi,psi,nq-2,tol,True)) / (psi*psi) 
        valid_idx = np.abs(psi)>tol
        zero_idx = np.abs(psi)<=tol
        phi_valid = phi[valid_idx]
        psi_valid = psi[valid_idx]
        phi_zero = phi[zero_idx]
        result = np.empty_like(phi, dtype='complex')
        result[valid_idx] = (1 + phi_valid*phi_valid*Fq(phi_valid, psi_valid, 
                                                        nq-4, tol, True) - \
                             (nq-4)/2.*Fq(phi_valid, psi_valid, nq-2, tol, 
                             True)) / (psi_valid*psi_valid)
                             
        result[zero_idx] = Fq0(phi_zero,nq)
        return result
              
    
    
def Fq0(phi, nq, tol=1e-14):
    r"""Special case psi=0 for :py:func:`Fq`, see the doc string there.
        
    """
    # nq must be an positive odd integer
    assert isinstance(nq, int) and nq>0 and nq%2 == 1

    if(nq == 3):
        return F32(phi,np.zeros_like(phi, dtype='complex'),tol)
    
    return (1+ phi*phi*Fq0(phi,nq-2))*2/(nq-2)
    
    

# TODO Add general recurrence function to evaluate F^m_q for m>1  

# Shorthand functions to evaluate low order F_q and F^m_q functions
def F12(phi, psi, tol=1e-14):
    r"""Shorthand function for :math:`\mathcal{F}_{1/2}(\phi,\psi)`
    
    Need to take care of special cases when phi=0 and phi is imaginary. 
    Note that :math:`\mathcal{F}_{1/2} \to +\infty` when :math:`\phi \to 0^+`.
    However, this singularity does not affect higher order functions. In this
    case, :math:`\mathcal{F}_{5/2}` needs to be evaluated directly from Z 
    function, and serve as a starting point for higher order functions.
    
    For :math:`\phi^2<0` case, refer to [1]_, We have modified recurrence 
    relation:
    
    Letting :math:`\phi = -\mathrm i \tilde{\phi}`
    
    .. math::
        
        \mathcal{F}_{1/2}(\phi,\psi) = \mathrm{Im} 
                                    Z(\psi+\mathrm{i}\tilde{\phi})/\tilde{\phi}
    
    :param phi: :math:`\phi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype phi: ndarray of complex
    :param psi: :math:`\psi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype psi: ndarray of complex
    :param float tol: tolerance for testing phi=0 condition    
    :return: :math:`\mathcal{F}_{1/2}(\phi,\psi)` evaluated at given 
             :math:`\phi` and :math:`\psi` mesh
    :rtype: ndarray of complex
    
    .. [1] Weakly relativistic dielectric tensor and dispersion functions of a 
           Maxwellian plasma, V. Krivenski and A. Orefice, J. Plasma Physics 
           (1983), vol. 30, part 1, pp. 125-131
    """    
    assert(np.array(phi).shape == np.array(psi).shape)
    
    # since physically phi^2 is real, phi is either pure real or imaginary, 
    # test if this condition is satisfied.
    assert(np.logical_or(np.real(phi)<tol, np.imag(phi)<tol).all())
   
    result = np.zeros_like(phi, dtype='complex')    
    real_idx = np.real(phi) >= tol
    imag_idx = np.real(phi) < tol
    diverge_idx = np.logical_and(np.real(phi) < tol, np.imag(phi) < tol) 
    
    # real phi follows the normal recurrence relation    
    result[real_idx] = -(Z(psi[real_idx]-phi[real_idx]) + 
                        Z(-psi[real_idx]-phi[real_idx])) / (2*phi[real_idx])
    # imaginary phi needs conversion
    phi_tilde = np.abs(np.imag(phi[imag_idx]))
    result[imag_idx] = np.imag(Z(psi[imag_idx]+1j*phi_tilde))/phi_tilde
    
    # phi=0 diverges
    if(diverge_idx.any()):
        warnings.warn('F12 enconters phi=0 input, it diverges at {} points. \
Check the data to see what\'s going on.'.format(np.count_nonzero(diverge_idx)))
    result[diverge_idx] = np.nan
    return result

def F32(phi, psi, tol = 1e-14):
    r"""Shorthand function for :math:`\mathcal{F}_{3/2}(\phi,\psi)`
    
    Need to take care of special cases when psi=0. 
    :math:`\mathcal{F}_{3/2}(\phi,\psi)=-Z'(-\phi)`  when :math:`\psi=0`
    
    For :math:`\phi^2<0` case, refer to [1]_, We have modified recurrence relation:
    Letting :math:`\phi = -\mathrm i \tilde{\phi}`
    
    .. math::
        
        \mathcal{F}_{3/2}(\phi,\psi) = -\mathrm{Re} Z(\psi+\mathrm{i}\tilde{\phi})/\psi
        
    if :math:`\psi=0`, then 
    
    :param phi: :math:`\phi` parameter defined in ref.[2] in :py:mod:`PlasmaDispersionFunction`
    :type phi: ndarray of complex
    :param psi: :math:`\psi` parameter defined in ref.[2] in :py:mod:`PlasmaDispersionFunction`
    :type psi: ndarray of complex
    :param float tol: tolerance for testing psi=0 condition    
    :return: :math:`\mathcal{F}_{1/2}(\phi,\psi)` evaluated at given :math:`\phi`
             and :math:`\psi` mesh
    :rtype: ndarray of complex
    
    .. [1] Weakly relativistic dielectric tensor and dispersion functions of a 
           Maxwellian plasma, V. Krivenski and A. Orefice, J. Plasma Physics (1983)
           , vol. 30, part 1, pp. 125-131
    """   


    assert(np.array(phi).shape == np.array(psi).shape)
    
    # since physically phi^2 is real, phi is either pure real or imaginary, 
    # test if this condition is satisfied.
    assert(np.logical_or(np.real(phi)<tol, np.imag(phi)<tol).all())
    
    result = np.zeros_like(phi, dtype='complex')    
    
    # here, we'll deal with real and imaginary phi together
    # the trick is, make sure real(phi)>0 and imaginary(phi)<0
    phi_mod = np.abs(np.real(phi)) - 1j*np.abs(np.imag(phi))    
    
    nonzero_idx = np.abs(psi) >= tol
    zero_idx = np.abs(psi) < tol
    result[nonzero_idx] = -(Z(psi[nonzero_idx]-phi_mod[nonzero_idx]) - 
                         Z(-psi[nonzero_idx]-phi_mod[nonzero_idx])) / \
                         (2*psi[nonzero_idx])
    
    result[zero_idx] = -Z_1(-phi_mod[zero_idx])
    return result

def F52(phi, psi, tol=1e-14):
    r"""Shorthand function for :math:`\mathcal{F}_{5/2}(\phi,\psi)`
    
    Need to take care of following special cases 
    
    1. psi=0
    
    .. math::    
    
        \mathcal{F}_{5/2}(\phi,0)=2(1+\phi^2\mathcal{F}_{3/2})/5  
    
    2. phi=0
    
    .. math::
    
        \mathcal{F}_{5/2}(0,\psi) = (1 - \frac{5}{2}\mathcal{F}_{3/2})/\psi^2
    
    3. phi is imaginary 
    
    refer to [1]_, We need to set :math:`\mathrm{Im}\phi<0`
    
    :param phi: :math:`\phi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype phi: ndarray of complex
    :param psi: :math:`\psi` parameter defined in ref.[2] in 
                :py:mod:`PlasmaDispersionFunction`
    :ptype psi: ndarray of complex
    :param float tol: tolerance for testing psi=0 condition    
    :return: :math:`\mathcal{F}_{1/2}(\phi,\psi)` evaluated at given 
             :math:`\phi` and :math:`\psi` mesh
    :rtype: ndarray of complex
    
    .. [1] Weakly relativistic dielectric tensor and dispersion functions of a 
           Maxwellian plasma, V. Krivenski and A. Orefice, J. Plasma Physics 
           (1983), vol. 30, part 1, pp. 125-131
    """   
    assert(np.array(phi).shape == np.array(psi).shape)    
    # since physically phi^2 is real, phi is either pure real or imaginary, 
    # test if this condition is satisfied.
    assert(np.logical_or(np.real(phi)<tol, np.imag(phi)<tol).all())

    result = np.empty_like(phi, dtype='complex')
    
    # First, we modify phi so that it complies with our requirement
    phi_mod = np.abs(np.real(phi)) - 1j*np.abs(np.imag(phi))
    # Now, we process zero psi part
    psi0_idx = np.logical_and(np.abs(np.real(psi)) < tol, 
                              np.abs(np.imag(psi)) < tol)
    result[psi0_idx] = 2*(1 + phi_mod*phi_mod*F32(phi[psi0_idx]), 
                                                  psi[psi0_idx]) / 5 
    
    # Finally, we deal with phi==0 part and phi!=0 part
    nonzero_idx = np.logical_or(np.real(phi) >= tol, np.imag(phi) >= tol)
    
    zero_idx = np.logical_and( np.logical_not(nonzero_idx),
                              np.logical_not(psi0_idx))
    nonzero_idx = np.logical_and(nonzero_idx, np.logical_not(psi0_idx))
    
    result[nonzero_idx] = (1 + phi[nonzero_idx]*phi[nonzero_idx]* \
                           F12(phi[nonzero_idx], psi[nonzero_idx]) - 
                           0.5*F32(phi[nonzero_idx], psi[nonzero_idx])) / \
                                    (psi[nonzero_idx]*psi[nonzero_idx])
    result[zero_idx] =  (1 - 0.5*F32(phi[zero_idx], psi[zero_idx])) / \
                          psi[zero_idx]**2
    return result

# TODO Modify higher order shorthand functions

def F32_1(phi, psi, tol = 1e-14):
    """Shorthand function for :math:`\mathcal{F}'_{3/2}(\phi,\psi)`
    
    Need to take care of special cases when psi=0. 
    :math:`\mathcal{F}_{3/2}(\phi,\psi)=-Z'(-\phi)`  when :math:`\psi=0`
    
    :param phi: :math:`\phi` parameter defined in ref.[2] in :py:mod:`PlasmaDispersionFunction`
    :ptype phi: ndarray of float
    :param psi: :math:`\psi` parameter defined in ref.[2] in :py:mod:`PlasmaDispersionFunction`
    :ptype psi: ndarray of float
    :param float tol: tolerance for testing psi=0 condition
    
    :return: :math:`\mathcal{F}_{1/2}(\phi,\psi)` evaluated at given :math:`\phi`
    and :math:`\psi` mesh
    :rtype: ndarray of complex
    """   
    return (Z_1(psi-phi)-Z_1(-psi-phi))/(4*psi*phi)


def F52_1(phi, psi):
    plus = psi-phi
    minus = -psi - phi
    return (-(Z(plus)-psi*Z_1(plus)) + (Z(minus)+psi*Z_1(minus))) / (4*psi**3)


def F52_2(phi, psi):
    plus = psi-phi
    minus = -psi-phi
    return ((Z_1(plus)-psi*Z_2(plus)) -
            (Z_1(minus)+psi*Z_2(minus))) / (8*phi*psi)


def F72_1(phi, psi):
    return (F32(phi, psi) + phi**2*F32_1(phi, psi) -
            1.5*F52_1(phi, psi)) / psi**2


def F72_2(phi, psi):
    plus = psi-phi
    minus = -psi-phi
    return (-(3*Z(plus)-3*psi*Z_1(plus)+psi**2*Z_2(plus)) +
            (3*Z(minus)+3*psi*Z_1(minus)+psi**2*Z_2(minus))) / (8*psi**5)


def F92_1(phi, psi):
    return (F52(phi, psi) + phi**2*F52_1(phi, psi) -
            2.5*F72_1(phi, psi)) / psi**2




    
    