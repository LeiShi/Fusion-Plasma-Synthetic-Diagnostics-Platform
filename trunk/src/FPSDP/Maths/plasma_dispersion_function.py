import numpy as np
from ..Maths.Funcs import my_quad

def Z(x):
    x_arr = np.linspace(0,x,100)
    y_arr = np.exp(x_arr**2)
    I = my_quad(y_arr,x_arr)
    Z_Re = -2*np.exp(-x**2)*I
    Z_Im = np.exp(-x**2)*np.pi**0.5
    return complex(Z_Re,Z_Im)

def Z_1(x):
    return -2*(1+x*Z(x))

def Z_2(x):
    return -2*(x*Z_1(x)+Z(x))

def F12(phi,psi):
    return -(Z(psi-phi)+Z(-psi-phi))/(2*phi)
def F32(phi,psi):
    return -(Z(psi-phi)-Z(-psi-phi))/(2*psi)
def F32_1(phi,psi):
    return (Z_1(psi-phi)-Z_1(-psi-phi))/(4*psi*phi)
def F52(phi,psi):
    return (1+phi**2*F12(phi,psi)-0.5*F32(phi,psi))/psi**2
def F52_1(phi,psi):
    plus = psi-phi
    minus= -psi-phi
    return (-(Z(plus)-psi*Z_1(plus))+(Z(minus)+psi*Z_1(minus)))/(4*psi**3)
def F52_2(phi,psi):
    plus=psi-phi
    minus=-psi-phi
    return ((Z_1(plus)-psi*Z_2(plus))-(Z_1(minus)+psi*Z_2(minus)))/(8*phi*psi)
def F72(phi,psi):
    return (1+phi**2*F32(phi,psi)-1.5*F52(phi,psi))/psi**2
def F72_1(phi,psi):
    return (F32(phi,psi)+phi**2*F32_1(phi,psi)-1.5*F52_1(phi,psi))/psi**2
def F72_2(phi,psi):
    plus = psi-phi
    minus = -psi-phi
    return (-(3*Z(plus)-3*psi*Z_1(plus)+psi**2*Z_2(plus))+(3*Z(minus)+3*psi*Z_1(minus)+psi**2*Z_2(minus)))/(8*psi**5)

def F92_1(phi,psi):
    return (F52(phi,psi)+ phi**2*F52_1(phi,psi) - 2.5*F72_1(phi,psi))/psi**2