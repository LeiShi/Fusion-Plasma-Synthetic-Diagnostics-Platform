"""Create virtual detectors 
"""

#module depends on numpy
import numpy as np
import scipy as sp
from ..GeneralSettings.UnitSystem import cgs
from scipy.interpolate import RectBivariateSpline
from ..Geometry.Grid import Path2D

class detector:
    """class of the detector which contains the frequency and path information

    Attributes:

    frequency related attributes:
    f_ctr: double, the center frequency ,in GHz
    n_f  : int, the number of frequencies stored in f_flt array
    f_flt: double[n], the frequencies chosen within the pass band
    p_flt: double[n], the pass coefficient for corresponding frequencies, may not be normalized

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
    ResS = 0.05
    Profs = []
    for dtc in Detectors:
        path2D = Path2D(dtc.pth,ResS)
        R_fld = Profile['Grid'].R2D[0,:]
        Z_fld = Profile['Grid'].Z2D[:,0]
        ne_interp = RectBivariateSpline(Z_fld,R_fld,Profile['ne'])
        Te_interp = RectBivariateSpline(Z_fld,R_fld,Profile['Te'])
        B_interp = RectBivariateSpline(Z_fld,R_fld,Profile['B'])
        ne_path = ne_interp.ev(path2D.Z2D[0,:],path2D.R2D[0,:])
        Te_path = Te_interp.ev(path2D.Z2D[0,:],path2D.R2D[0,:])
        B_path = B_interp.ev(path2D.Z2D[0,:],path2D.R2D[0,:])
        new_prof = {}
        new_prof['Grid'] = path2D
        new_prof['ne'] = ne_path[np.newaxis,:]
        new_prof['Te'] = Te_path[np.newaxis,:]
        new_prof['B'] = B_path[np.newaxis,:]
        new_prof['omega'] = 2*np.pi*dtc.f_flt 
        Profs.append(new_prof)
    return tuple(Profs)
        
        
            