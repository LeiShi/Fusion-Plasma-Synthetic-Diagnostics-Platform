"""Create virtual detectors 
"""

#module depends on numpy
import numpy as np
from ..GeneralSettings.UnitSystem import cgs
from scipy.interpolate import InterpolatedUnivariateSpline
from ..Geometry.Grid import Path2D

class path:
    """class of the light path, basically just a series of points

    Attributes:
    n: int, number of points on the path
    R: double[n], R coordinates of the points
    Z: double[n], Z coordinates of the points 
    """    
    def __init__(this, n=0, R=np.zeros(1), Z=np.zeros(1)):
        this.n = n
        this.R = R
        this.Z = Z
    def __setitem__(this,p2):
        this.n = p2.n
        this.R = np.copy(p2.R)
        this.Z = np.copy(p2.Z)

class detector:
    """class of the detector which contains the frequency and path information

    Attributes:

    frequency related attributes:
    f_ctr: double, the center frequency
    n_f  : int, the number of frequencies stored in f_flt array
    f_flt: double[n], the frequencies chosen within the pass band
    p_flt: double[n], the pass coefficient for corresponding frequencies

    light path related attributes:
    pth : class path, the ligth path information

    Methods:
    """
    def __init__(this,f_ctr,n_f,f_flt,p_flt,pth):
        this.f_ctr = f_ctr
        this.n_f = n_f
        this.f_flt = np.copy(f_flt)
        this.p_flt = np.copy(p_flt)
        this.pth = pth
    def __setitem__(this,d2):
        this.f_ctr = d2.f_ctr
        this.n_f = d2.n_f
        this.f_flt = np.copy(d2.f_flt)
        this.p_flt = np.copy(d2.p_flt)
        this.pth = d2.pth


def create_spatial_frequency_grid(Detectors, Profile):
    """create grids points on spatial and frequency space for alpha calculation suitable for given detector parameters

    Detectors: detector[], an array contains all the detector objects
    Profile: a dictionary contains the plasma data
    return value: Profs, an array contains all the quantities dictionaries on grids that feed to alpha calculation function, each profile corresponds to one detector.
    """
    n_dtc = len(Detectors)
    Profs = []
    for dtc in Detectors:
        
        
            