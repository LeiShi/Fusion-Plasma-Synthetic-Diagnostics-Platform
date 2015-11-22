"""Plotting functions for visualization and diagnostic purpose
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ...GeneralSettings.UnitSystem import cgs

def calc_frequencies(pro):
    """Calculate relevant cold plasma frequencies for a given profile
    """
    ne = pro['ne']
    B = pro['B']
    omega_p = np.sqrt(4*np.pi*ne*cgs['e']**2 / cgs['m_e'])
    omega_c = cgs['e']*B / (cgs['m_e']*cgs['c'])
    omega_uh = np.sqrt(omega_p**2 + omega_c**2 )
    omega_lh = np.sqrt(omega_p * omega_c)
    return dict(w_p = omega_p, w_c = omega_c, w_uh = omega_uh, w_lh = omega_lh)

def plot_f(R,f):
    """plot frequencies
    f: the dictionary that created by calc_frequencies
    R: the 1D radial coordinates corresponding to f
    """
    Mid_idx = int(f['w_p'].shape[0]/2) 
    plt.plot(R,f['w_p'][Mid_idx,:])
    plt.plot(R,f['w_c'][Mid_idx,:])
    plt.plot(R,2*f['w_c'][Mid_idx,:])
    plt.plot(R,f['w_uh'][Mid_idx,:])
    plt.plot(R,f['w_lh'][Mid_idx,:])
    labels=['f_p','f_c','2f_c','f_uh','f_lh']
    plt.legend(labels)
    
def plot_2D_profiles(plasma):
    """plot the 2D ne,te profiles for given plasma
    plasma: dictionary, contains all the profiles, see Plasma.TestParameter for details
    """
    
    #rmin