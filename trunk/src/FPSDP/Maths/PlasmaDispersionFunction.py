r"""
This module provides ways to evaluate the Plasma Dispersion Function [1], 
:math:`Z(x)`, and other related functions, specifically, the 
:math:`\mathcal{F}_q(\phi,\psi)` Function [2]. 

The Faddeeva function 

.. math::
   
  w(z) \equiv \exp(-z^2) \; {\mathrm {erfc}}(-{\mathrm i}z)
    
is used, where :math:`{\mathrm {erfc}}(z)` is the complementary error function. 
It is evaluated using the python wrapper of Steven G. Johnson's routine, 
provided by scipy, see :py:func:`scipy.spetial.wofz` for more details.

The Plasma Dispersion Function(PDF) is related to Faddeeva function as

.. math::
   
  Z(z) = {\mathrm i}\sqrt{\pi} \; w(z) \; ,
    
and :math:`\mathcal{F}_q` function is related to PDF as[2]:

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

However, as pointed out in [2], evaluating derivatives using the first formula 
may suffer from the cancellation of two large numbers. A more reliable way is 
to express the derivatives of :math:`\mathcal{F}_{1/2}` and 
:math:`\mathcal{F}_{3/2}` in terms of derivatives of the PDF, and then use the 
second formula to evaluate larger q's.

PDF has the following property[1]:

.. math::
   Z'(z) = -2(1+zZ(z)) \; ,

and it's easy to show the following recurrence relation 

.. math::
   Z^m(z) = -2[(m-1)Z^{m-2}(z) + zZ^{m-1}(z)] \quad \mathrm{for}\; m>2 \; .
   
Fianlly, for special case, :math:`\psi=0`, L'Hopital rule needs to be used to
evaluate the "0/0" kind expressions. More details in Appendix part of [2].

[1] https://farside.ph.utexas.edu/teaching/plasma/lectures1/node87.html

[2] I.P.Shkarofsky, "New representations of dielectric tensor elements in magn-
etized plasma", J. Plasma Physics(1986), vol. 35, part 2, pp. 319-331

"""

from math import sqrt

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


# Deprecated versions to evaluate Z

def F12(phi, psi):
    return -(Z(psi-phi) + Z(-psi-phi)) / (2*phi)


def F32(phi, psi):
    return -(Z(psi-phi) - Z(-psi-phi)) / (2*psi)


def F32_1(phi, psi):
    return (Z_1(psi-phi)-Z_1(-psi-phi))/(4*psi*phi)


def F52(phi, psi):
    return (1 + phi**2*F12(phi, psi) - 0.5*F32(phi, psi)) / psi**2


def F52_1(phi, psi):
    plus = psi-phi
    minus = -psi - phi
    return (-(Z(plus)-psi*Z_1(plus)) + (Z(minus)+psi*Z_1(minus))) / (4*psi**3)


def F52_2(phi, psi):
    plus = psi-phi
    minus = -psi-phi
    return ((Z_1(plus)-psi*Z_2(plus)) -
            (Z_1(minus)+psi*Z_2(minus))) / (8*phi*psi)


def F72(phi, psi):
    return (1 + phi**2*F32(phi, psi) - 1.5*F52(phi, psi)) / psi**2


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


# Let's try to evaluate Plasma Dispersion Function, 'Z', from the faddeeva 
# function.

    
    