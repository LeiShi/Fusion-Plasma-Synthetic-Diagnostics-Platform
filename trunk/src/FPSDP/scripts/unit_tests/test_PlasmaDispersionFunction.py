# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 10:49:45 2016

@author: lei

test FPSDP.Maths.PlasmaDispersionFunction
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.scimath import sqrt

import FPSDP.Maths.PlasmaDispersionFunction as pdf
from FPSDP.Geometry.Grid import cubicspace

phi2 = np.linspace(-4, 10, 1001)
psi = np.zeros_like(phi2)
phi = sqrt(phi2)

F52 = pdf.Fq(phi,psi,5)
F72 = pdf.Fq(phi,psi,7)
F92 = pdf.Fq(phi,psi,9)


def plotF():
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(-phi2, np.real(F52), label='q=5/2')
    ax[0].plot(-phi2, np.real(F72), label='q=7/2')
    ax[0].plot(-phi2, np.real(F92), label='q=9/2')
    ax[0].set_xlim(-10, 4)
    ax[0].set_ylim(-0.8, 1)
    ax[0].set_aspect(14/1.8)
    ax[0].set_ylabel(r'Re(Fq(z))')
    ax[0].grid(True, which='both')
    ax[0].axhline(y=0, color='k')
    ax[0].axvline(x=0, color='k')
    ax[0].legend(loc='best')
    
    ax[1].plot(-phi2, -np.imag(F52), label='q=5/2')
    ax[1].plot(-phi2, -np.imag(F72), label='q=7/2')
    ax[1].plot(-phi2, -np.imag(F92), label='q=9/2')
    ax[1].set_xlim(-10, 4)
    ax[1].set_ylim(-0.8, 1)
    ax[1].set_aspect(14/1.8)
    ax[1].set_ylabel(r'-Im(Fq)')
    ax[1].set_xlabel(r'z')
    ax[1].grid(True, which='both')
    ax[1].axhline(y=0, color='k')
    ax[1].axvline(x=0, color='k')
    ax[1].legend(loc='best')
    
    return fig
    
    
mudelta = np.linspace(-30, 30, 1001)
psi1D = np.linspace(-20, 20, 1001)
mudelta2D = mudelta[np.newaxis, :] + np.zeros((1001,1001))
psi2D = psi1D[:,np.newaxis] + np.zeros((1001,1001))

phi2D = sqrt(psi2D*psi2D - mudelta2D)

F72_2D = pdf.Fq(phi2D, psi2D, 7)
F7_1 = pdf.Fmq(phi2D, psi2D, 1, 7)


def plotF2D(F):
    
    fig, ax = plt.subplots()
    
    image0= ax.imshow(F, extent=[-30, 30, -20, 20], 
                         origin='lower')
    ax.set_ylabel(r'$\psi$')
    ax.set_xlabel(r'$\mu\delta$')
    fig.colorbar(image0)
    
    return fig
    
    
    
    