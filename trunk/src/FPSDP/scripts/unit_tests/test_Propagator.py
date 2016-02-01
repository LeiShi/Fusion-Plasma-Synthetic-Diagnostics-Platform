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


def test1D(mode):
    p = tp.create_profile1D(True, 0)
    
    propagator = prop.ParaxialPerpendicularPropagator1D(p, dt.ColdDielectric, 
                                                        mode, -1)
    
    start_plane = tp.Grid.Cartesian2D(DownLeft=(-40,-40), UpRight=(40,40), 
                                      NR=256, NZ=256)
    
    x_start = 250
    x_end = 150
    nx = 200
    
    Z1D, Y1D = start_plane.get_mesh()
    Z2D,Y2D = start_plane.get_ndmesh()            
    X2D = np.ones_like(Y2D)*x_start
    
    omega = 8e11
    gb = lb.GaussianBeam(2*np.pi*3e10/omega, 280, 0, 1)
    
    E_start = gb([Z2D, Y2D, X2D])
    
    E = propagator.propagate(0, omega, x_start, x_end, nx, E_start, 
                             start_plane.R1D, start_plane.Z1D)
                             
    return (propagator, E, gb)
    
def test2D(mode):
    p = tp.create_profile2D(True, 0)
    
    propagator = prop.ParaxialPerpendicularPropagator2D(p, dt.ColdDielectric, 
                                                        mode, direction = -1, 
                                                        ray_y=0)
    
    start_plane = tp.Grid.Cartesian2D(DownLeft=(-40,-40), UpRight=(40,40), 
                                      NR=256, NZ=256)
    
    x_start = 250
    x_end = 150
    nx = 200
    
    Z1D, Y1D = start_plane.get_mesh()
    Z2D,Y2D = start_plane.get_ndmesh()            
    X2D = np.ones_like(Y2D)*x_start
    
    omega = 8e11
    gb = lb.GaussianBeam(2*np.pi*3e10/omega, 280, 0, 1)
    
    E_start = gb([Z2D, Y2D, X2D])
    
    E = propagator.propagate(0, omega, x_start, x_end, nx, E_start, 
                             Y1D, Z1D)
                             
    return (propagator, E, gb)


