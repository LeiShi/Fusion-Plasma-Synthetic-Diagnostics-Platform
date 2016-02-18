# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:29:38 2016

@author: lei

unit test for FPSDP.Models.Waves.Propagator
"""
import time

import numpy as np

import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp
import FPSDP.Maths.LightBeam as lb
import FPSDP.Models.Waves.Propagator as prop
import FPSDP.Plasma.DielectricTensor as dt
from FPSDP.GeneralSettings.UnitSystem import cgs

tp.set_parameter1D(Te_0=10*cgs['keV'])
tp.set_parameter2D(Te_0=10*cgs['keV'])

p1d = tp.create_profile1D(True, 0)
p1d.setup_interps()

p2d = tp.create_profile2D(True, 0)
p2d_uni = tp.simulate_1D(p1d, p2d.grid)                     
p2d.setup_interps()
p2d_uni.setup_interps()

start_plane = tp.Grid.Cartesian2D(DownLeft=(-20,-20), UpRight=(20,20), 
                                  NR=65, NZ=64)
x_start = 250
x_end = 150
nx = 100

Z1D, Y1D = start_plane.get_mesh()
Z2D,Y2D = start_plane.get_ndmesh()            
X2D = np.ones_like(Y2D)*x_start

omega = 8e11
gb = lb.GaussianBeam(2*np.pi*3e10/omega, 260, 0, 1, tilt_h=0, P_total=1)

E_start = gb([Z2D, Y2D, X2D])

def prop1d(mode, dielectric, max_harmonic, max_power):    
    propagator1d = prop.ParaxialPerpendicularPropagator1D(p1d, 
                                                     dielectric, 
                                                     mode, direction=-1,
                                                     max_harmonic=max_harmonic,
                                                     max_power=max_power)

    E = propagator1d.propagate(omega, x_start, x_end, nx, E_start, 
                         Y1D, Z1D, mute=False, debug_mode=True)
    return (E, propagator1d)
    
    
def prop2d(mode, dielectric, max_harmonic, max_power):    
    propagator = prop.ParaxialPerpendicularPropagator2D(p2d, dielectric, 
                                                        mode, direction = -1, 
                                                        ray_y=0, 
                                                    max_harmonic=max_harmonic,
                                                        max_power=max_power)
    
    
    E = propagator.propagate(omega, x_start, x_end, nx, E_start, 
                             Y1D, Z1D, mute=False, debug_mode=True)
                             
    return (E, propagator)
    

def prop2d_uni(mode, dielectric, max_harmonic, max_power):    
    propagator = prop.ParaxialPerpendicularPropagator2D(p2d_uni, dielectric, 
                                                        mode, direction = -1, 
                                                        ray_y=0, 
                                                    max_harmonic=max_harmonic,
                                                        max_power=max_power)
    
    
    E = propagator.propagate(omega, x_start, x_end, nx, E_start, 
                             Y1D, Z1D, mute=False, debug_mode=True)
                             
    return (E, propagator)
                             

def compare_1d2d(mode, dielectric, max_harmonic, max_power):

    propagator1d = prop.ParaxialPerpendicularPropagator1D(p1d, 
                                                     dielectric, 
                                                     mode, direction=-1,
                                                     max_harmonic=max_harmonic,
                                                     max_power=max_power)

    E1 = propagator1d.propagate(omega, x_start, x_end, nx, E_start, 
                         Y1D, Z1D)
        
    propagator2d_uni = prop.ParaxialPerpendicularPropagator2D(p2d_uni, 
                                                              dielectric, 
                                                        mode, direction = -1, 
                                                        ray_y=0, 
                                                    max_harmonic=max_harmonic,
                                                        max_power=max_power)
    
    
    E2d_uni = propagator2d_uni.propagate(omega, x_start, x_end, nx, E_start, 
                             Y1D, Z1D)
                             
    return (E1, propagator1d, E2d_uni, propagator2d_uni)
    

def error_1d(nx_power_min, nx_power_max, nx_power_step, mode='O', 
             dielectric=dt.ColdElectronColdIon, max_harmonic=2, max_power=2):
    """
    nx_step: the step size in nx power
    """
    nx_power_array = np.arange(nx_power_min, nx_power_max+1, nx_power_step)             
    nx_array = 2**nx_power_array
    x_stepsize = np.float(np.abs(x_end - x_start))/nx_array
    
    p1 = prop.ParaxialPerpendicularPropagator1D(p1d, 
                                                dielectric, 
                                                mode, direction=-1,
                                                max_harmonic=max_harmonic,
                                                max_power=max_power)
    max_abs_err = np.zeros_like(nx_array, dtype='float')
    max_rel_err = np.zeros_like(max_abs_err)
    
    tstart = time.clock()    
    
    E_mark = p1.propagate(omega, x_start, x_end, nx_array[-1], E_start, Y1D, 
                          Z1D)    
    
    for i, ni in enumerate(nx_array[:-1]):
        
        Ei = p1.propagate(omega, x_start, x_end, ni, E_start, 
                          Y1D, Z1D)
        
        step = nx_array[-1]/nx_array[i]        
        E_marki = E_mark[..., ::step]
        abs_err = np.abs(Ei - E_marki)
        max_abs_err[i] = np.max(abs_err)
        
        rel_idx = np.abs(E_marki)>1e-3*np.max(np.abs(E_marki))
        max_rel_err[i] = np.max(np.abs(abs_err[rel_idx] / E_marki[rel_idx]))
        
    tend = time.clock()
    print('Total time used: {:.4}s'.format(tend-tstart))
        
    return max_abs_err, max_rel_err, nx_array, x_stepsize    
    
    
def error_2d(nx_power_min, nx_power_max, nx_power_step, mode='O', 
             dielectric=dt.ColdElectronColdIon, max_harmonic=2, max_power=2):
    """
    nx_step: the step size in nx power
    """
    nx_power_array = np.arange(nx_power_min, nx_power_max+1, nx_power_step)             
    nx_array = 2**nx_power_array
    x_stepsize = np.float(np.abs(x_end - x_start))/nx_array
    
    p2 = prop.ParaxialPerpendicularPropagator2D(p2d_uni, 
                                                dielectric, 
                                                mode, direction=-1,
                                                ray_y=0,
                                                max_harmonic=max_harmonic,
                                                max_power=max_power)
    max_abs_err = np.zeros_like(nx_array, dtype='float')
    max_rel_err = np.zeros_like(max_abs_err)
    
    tstart = time.clock()    
    
    E_mark = p2.propagate(omega, x_start, x_end, nx_array[-1], E_start, Y1D, 
                          Z1D)    
    
    for i, ni in enumerate(nx_array[:-1]):
        
        Ei = p2.propagate(omega, x_start, x_end, ni, E_start, 
                          Y1D, Z1D)
        
        step = nx_array[-1]/nx_array[i]        
        E_marki = E_mark[..., ::step]
        abs_err = np.abs(Ei - E_marki)
        max_abs_err[i] = np.max(abs_err)
        
        rel_idx = np.abs(E_marki)>1e-3*np.max(np.abs(E_marki))
        max_rel_err[i] = np.max(np.abs(abs_err[rel_idx] / E_marki[rel_idx]))
        
    tend = time.clock()
    print('Total time used: {:.4}s'.format(tend - tstart))
        
    return max_abs_err, max_rel_err, nx_array, x_stepsize        
    
    
def benchmark_1d2d(nx_power_min, nx_power_max, nx_step, mode='O',
                   dielectric=dt.ColdElectronColdIon, max_harmonic=2, 
                   max_power=2):
    """calculate error convergence against x step size
    
    :param float nx_min: log2(nx) minimum, start point of log mesh of total x 
                         steps.
    :param float nx_max: log2(nx) maximum, end point of log mesh of total x 
                         steps.
    :param int nx_step: step size in power array
    :param string mode: polarization mode
    :param dielectric: dielectric tensor type
    :type dielectric: :py:class:`FPSDP.Plamsa.DielectricTensor.Dielectric`
    :param int max_harmonic: highest harmonic resonance included
    :param int max_power: highest FLR effect power included
    """
    tstart = time.clock()    
    
    nx_power = np.arange(nx_power_min, nx_power_max+1, nx_step)    
    
    nx_array = 2**nx_power
    x_stepsize = np.float(np.abs(x_end - x_start))/nx_array
    
    p1 = prop.ParaxialPerpendicularPropagator1D(p1d, 
                                                dielectric, 
                                                mode, direction=-1,
                                                max_harmonic=max_harmonic,
                                                max_power=max_power)
    p2 = prop.ParaxialPerpendicularPropagator2D(p2d_uni, dielectric, mode,
                                                direction=-1, ray_y=0,
                                                max_harmonic=max_harmonic,
                                                max_power=max_power)    
    
    max_abs_err = np.empty_like(nx_array, dtype='float')
    max_rel_err = np.empty_like(max_abs_err)
    
    for i, ni in enumerate(nx_array):
        
        E1 = p1.propagate(omega, x_start, x_end, ni, E_start, 
                          Y1D, Z1D)
        E2 = p2.propagate(omega, x_start, x_end, ni, E_start, 
                          Y1D, Z1D)
        
        abs_err = np.abs(E2 - E1)
        max_abs_err[i] = np.max(abs_err)
        
        rel_idx = np.abs(E1)>1e-3*np.max(np.abs(E1))
        max_rel_err[i] = np.max(np.abs(abs_err[rel_idx] / E1[rel_idx]))
        
    print('Total time: {:.4}s'.format(time.clock()-tstart))
        
    return max_abs_err, max_rel_err, nx_array, x_stepsize
        


