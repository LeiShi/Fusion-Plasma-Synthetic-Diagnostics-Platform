"""Create virtual detectors 
"""

#module depends on numpy
import numpy as np
import scipy as sp
from scipy.interpolate import RectBivariateSpline

from ...GeneralSettings.UnitSystem import cgs
from ...Geometry.Grid import path, Path2D, Cartesian2D
from ...Plasma.PlasmaProfile import ECEI_Profile

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

def create_2D_pointlike_detector_array(plasma):
    """create testing 2D detector array which covers all the given plasma region
    
    plasma: dictionary, contains all the plasma profiles on 2D grids, see package Plasma.TestParameters for detailed format
    """
    
    assert isinstance(plasma.grid, Cartesian2D)
    #get 1D coordinates
    Z = plasma.grid.Z2D[:,0] 
    R = plasma.grid.R2D[0,:]
    
    Z_idx_max = len(Z)-1
    Detectors = []
    for i in range(len(Z)):
        if (i%10 == 0):
            for j in range(len(R)):
                if(j%10==0):
                    Z_idx = Z_idx_max - i #images are stored from top to bottom, s.t. y axis needs to be inverted
                    R_idx = j
                    #start creating detector correspond to this point
                    f_ctr =  cgs['e']*plasma.B[Z_idx,R_idx]/(cgs['m_e']*cgs['c']) / np.pi
                    f_flt = [f_ctr]
                    p_flt = [1]
                    pth = path(2,[R[0],R[-1]],[Z[Z_idx],Z[Z_idx]]) #light path is assumed horizontal
                    Detectors.append(detector(f_ctr,1,f_flt,p_flt,pth))
    return tuple(Detectors)
            


def create_spatial_frequency_grid(Detectors, Profile):
    """create grids points on spatial and frequency space for alpha calculation suitable for given detector parameters

    Detectors: detector[], an array contains all the detector objects
    Profile: a dictionary contains the plasma data
    return value: Profs, an array contains all the quantities dictionaries on grids that feed to alpha calculation function, each profile corresponds to one detector.
    """
    ResS = 0.05
    Profs = []
    for dtc in Detectors:
        path2D = Path2D(dtc.pth,ResS)
        R_fld = Profile.grid.R2D[0,:]
        Z_fld = Profile.grid.Z2D[:,0]
        ne_interp = RectBivariateSpline(Z_fld,R_fld,Profile.ne)
        Te_interp = RectBivariateSpline(Z_fld,R_fld,Profile.Te)
        B_interp = RectBivariateSpline(Z_fld,R_fld,Profile.B)
        ne_path = ne_interp.ev(path2D.Z2D[0,:],path2D.R2D[0,:])
        Te_path = Te_interp.ev(path2D.Z2D[0,:],path2D.R2D[0,:])
        B_path = B_interp.ev(path2D.Z2D[0,:],path2D.R2D[0,:])
        new_prof = {}
        new_prof['Profile'] = ECEI_Profile(path2D, ne_path[np.newaxis,:], 
                                           Te_path[np.newaxis,:],
                                           Te_path[np.newaxis,:], 
                                           B_path[np.newaxis,:])
        new_prof['omega'] = 2*np.pi*dtc.f_flt 
        Profs.append(new_prof)
    return tuple(Profs)
        
        
            
