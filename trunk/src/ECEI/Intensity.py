"""Calculate the recieved light intensity by carrying out the integral along the light path

I(s) = integral(s_0,s)[alpha(s')*exp(-(tau - tau'))*omega^2/(8 pi^3 * c^2) * T(s')] ds' --- ref[1] Eq(2.2.13-2.2.15)

[1] 1983 Nucl. Fusion 23 1153
"""

from .Detector import create_spatial_frequency_grid
from .Alpha1 import get_alpha_table
from .Alpha1 import DefaultFqzTableFile
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from ..GeneralSettings.UnitSystem import cgs

import matplotlib.pyplot as plt

def get_intensity(Dtcs,RawProfile,n=2,FqzFile = DefaultFqzTableFile):
    """Calculate the intensity recieved by Detectors given by Dtcs
    
    Dtcs: list, containing all Detector objects
    RawProfile: Dictionary, Plasma Profile given on 2D grids
    n: harmonic number, default to be 2
    FqzFile: Function look up table file, default is set in Alpha1.py
    
    """
    specProfile = create_spatial_frequency_grid(Dtcs,RawProfile)
    intensity = np.zeros((len(Dtcs)))
    norms = np.zeros((len(Dtcs)))
    tau_arrays = []
    normal_arrays = []
    integrand_arrays = []
    for i in range(len(Dtcs)):
        s_array = specProfile[i]['Grid'].s2D[0,:] #extract the path length coordinate array, see detailed format of s2D in '..Geometry.Grid', class Path2D
        Te_array = specProfile[i]['Te'][0,:] #electron temperature along the path 
        alpha_array = get_alpha_table(specProfile[i],n,FqzFile)[:,0,:] #extract the 2D alpha array with the first dimention frequency and second path length
        dtc = Dtcs[i] # corresponding detector object
        for j in range(len(dtc.f_flt)):
            #alpha = InterpolatedUnivariateSpline(s_array,alpha_array[j,:])
            tau_array = np.zeros( (len(s_array)) )
            for k in np.arange( len(s_array) -1 ):
                tau_array[k+1] = tau_array[k] + (alpha_array[j,k]+alpha_array[j,k+1])*(s_array[k+1]-s_array[k])/2
            normal_array = alpha_array[j,:]*np.exp(tau_array - tau_array[-1])* dtc.f_flt[j]**2 /(2*np.pi*cgs['c']**2) #normalization integrand
            integrand_array = normal_array * Te_array #evaluate the integrand on each grid point

            tau_arrays.append(tau_array)
            normal_arrays.append(normal_array)
            integrand_arrays.append(integrand_array)

            integrand = InterpolatedUnivariateSpline(s_array,integrand_array) #interpolate
            intensity_f = quad(integrand,s_array[0],s_array[-1])[0] #integration over the path, result is for the given frequency
            intensity[i] += intensity_f * dtc.p_flt[j] # multiply with the pattern ratio, and add on to the total receivedintensity of channel i
            if (dtc.f_flt[j] == dtc.f_ctr):
                normal = InterpolatedUnivariateSpline(s_array,normal_array)
                normalization_f = quad(normal,s_array[0],s_array[-1])[0]
                norms[i]=normalization_f

    return (tuple(intensity),tuple(norms),tuple(tau_arrays),tuple(normal_arrays),tuple(integrand_arrays))