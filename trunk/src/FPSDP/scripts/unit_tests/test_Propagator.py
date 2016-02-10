# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:29:38 2016

@author: lei

unit test for FPSDP.Models.Waves.Propagator
"""

import numpy as np

import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp
import FPSDP.Maths.LightBeam as lb
import FPSDP.Models.Waves.Propagator as prop
import FPSDP.Plasma.DielectricTensor as dt
from FPSDP.GeneralSettings.UnitSystem import cgs

tp.set_parameter1D(Te_0=10*cgs['keV'])
tp.set_parameter2D(Te_0=10*cgs['keV'])


start_plane = tp.Grid.Cartesian2D(DownLeft=(-10,-10), UpRight=(10,10), 
                                  NR=64, NZ=64)
x_start = 250
x_end = 230
nx = 100

Z1D, Y1D = start_plane.get_mesh()
Z2D,Y2D = start_plane.get_ndmesh()            
X2D = np.ones_like(Y2D)*x_start

omega = 8e11
gb = lb.GaussianBeam(2*np.pi*3e10/omega, 260, 0, 2, tilt_h=0, P_total=1)

E_start = gb([Z2D, Y2D, X2D])

def prop1d(mode, dielectric, max_harmonic, max_power):
    p1d = tp.create_profile1D(True, 0)
    p1d.setup_interps()
    propagator1d = prop.ParaxialPerpendicularPropagator1D(p1d, 
                                                     dielectric, 
                                                     mode, direction=-1,
                                                     max_harmonic=max_harmonic,
                                                     max_power=max_power)

    E = propagator1d.propagate(0, omega, x_start, x_end, nx, E_start, 
                         Y1D, Z1D)
    return (E, propagator1d)
                             

def compare_1d2d(mode, dielectric, max_harmonic, max_power):
    p1d = tp.create_profile1D(True, 0)
    p1d.setup_interps()
    propagator1d = prop.ParaxialPerpendicularPropagator1D(p1d, 
                                                     dielectric, 
                                                     mode, direction=-1,
                                                     max_harmonic=max_harmonic,
                                                     max_power=max_power)

    E1 = propagator1d.propagate(0, omega, x_start, x_end, nx, E_start, 
                         Y1D, Z1D)
    
    p2d = tp.create_profile2D(True, 0)
    p2d_uni = tp.simulate_2D(p1d, p2d.grid)                     
    p2d.setup_interps()
    p2d_uni.setup_interps()
    
    propagator2d_uni = prop.ParaxialPerpendicularPropagator2D(p2d_uni, 
                                                              dielectric, 
                                                        mode, direction = -1, 
                                                        ray_y=0, 
                                                    max_harmonic=max_harmonic,
                                                        max_power=max_power)
    
    
    E2d_uni = propagator2d_uni.propagate(0, omega, x_start, x_end, nx, E_start, 
                             Y1D, Z1D)
                             
    return (E1, propagator1d, E2d_uni, propagator2d_uni)
    
def test2D(mode, dielectric, max_harmonic, max_power):
    p = tp.create_profile2D(True, 0)
    p.setup_interps()
    
    propagator = prop.ParaxialPerpendicularPropagator2D(p, dielectric, 
                                                        mode, direction = -1, 
                                                        ray_y=0, 
                                                    max_harmonic=max_harmonic,
                                                        max_power=max_power)
    
    
    E = propagator.propagate(0, omega, x_start, x_end, nx, E_start, 
                             Y1D, Z1D)
                             
    return (E, propagator)


