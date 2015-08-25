# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:28:04 2015

@author: lei
"""

"""Module calculates Chebyshev Polynomial integral arisen from electron cyclotron emission with FLR
"""
import numpy as np
from numpy.polynomial.chebyshev import poly2cheb, chebint, chebval, chebmul

def nthpoly(n):
    c = np.zeros((n+1))
    c[n] = 1
    return c
        

def C0(n):
    """ integral related to zero'th order finite Larmor radius effect
    """
    c1 = poly2cheb([0,1,0,-2])
    c2 = nthpoly(2*n)
    kernel = chebmul(c1,c2)
    cint = chebint(kernel)
    return chebval(1,cint)
    

def C1(n):
    """ integral related to first order finite Larmor radius effect
    """
    c1 = poly2cheb([0,0,0,0,-1.5,0,2])
    c2 = nthpoly(2*n)
    kernel = chebmul(c1,c2)
    cint = chebint(kernel)
    return chebval(1,cint)  

    