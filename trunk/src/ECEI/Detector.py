"""Create virtual detectors as well as the frequency table in which is interested 
"""

#module depends on numpy
import numpy as np
from ..GeneralSettings.UnitSystem import cgs


def make_frequency_table(Profile, Harmonic = 2 ,ResOmega = -1):
    """make the frequency table based on the Profile data, namely the B field range on Grid

    Profile: a dictionary contains all the plasma profile data. 'Grid' : Grid object, 'B' : B field on grids
    Harmonic: an integer indicates the targeting harmonic mode. default to be the second harmonics.
    ResF: an interger indecates the resolution on frequency table. default to be equally distributed on frequency with grid number equals NR.
    """

    Bmax = np.max(Profile['B'])
    Bmin = np.min(Profile['B'])
    
    Omega_max,Omega_min = cgs.e * Harmonic/(cgs.m_e * cgs.c) * (Bmax, Bmin)
    if(ResOmega < 0):
        NOmega = Profile['Grid'].NR
    else:
        NOmega = np.floor((Omega_max - Omega_min)/ResOmega) + 2 #make sure the Omega mesh is finer than the desired resolution

    return np.linspace(Omega_min,Omega_max,NOmega)

