"""Create virtual detectors 
"""

#module depends on numpy
import numpy as np
from ..GeneralSettings.UnitSystem import cgs
from scipy.interpolate import InterpolatedUnivariateSpline 

class path:
"""
class of the light path, basically just a series of points

n: int, number of points on the path
R: double[n], R coordinates of the points
Z: double[n], Z coordinates of the points 
"""    
    def __init__(this, n=0, R=np.zeros(1), Z=np.zeros(1)):
        this.n = n
        this.R = R
        this.Z = Z
        
    def linear_interp(this,n=200):
        """
        interpolate the existing (R,Z) path, and get a new path with at least n points
        
        n: int, the minimun value of the number of points on the new path
        """ 
        