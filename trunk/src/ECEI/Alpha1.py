"""Module contains Functions that calculate the local absorption coefficient alpha.
"""

#module depends on scipy.integrate and scipy.interpolate package 
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import gamma
#rename numpy as np for convention
import numpy as np
import pickle

#module depends on detector which contains the function of making omega table
from .Detector import make_frequency_table

from ..GeneralSettings.UnitSystem import cgs

#The default path and filename for the file that stores the Fqz tables
DefaultTableFile = './Fqz.sav'

def create_Fqz_table(zmin = -20., zmax = 20., nz = 401, q = 3.5, filename = './Fqz.sav', overwrite = True):
    """create the F_q(z_n) function value table using exact integration and summation formula[1]. Save the results into a file.  

    zmin,zmax : float; the lower and upper boudary of z table
    nz : float; total knots of z table
    q : float; parameter related to harmonic n, usually q = n+3/2
    filename : string; stroes the path and filename to save the Fqz function
    overwrite : bool; indicate overwrite the existing saving file or not.

    [1] 1983 Nucl. Fusion 23 1153 (Eqn. 2.3.68 and 2.3.70) 
    """

    z = np.linspace(zmin,zmax,nz)
    F_re = np.zeros(nz)
    F_re_err = np.zeros(nz)
    F_im = np.zeros(nz)
    for i in range(nz):
        F_re[i],F_re_err[i] = quad(lambda x: -1j*np.exp(1j*z[i]*x)/(1-1j*x)**q, 0, np.inf)
        if( z[i] < 0):
            F_im[i] = -np.pi*(-z[i])**(q-1)*np.exp(z[i])/gamma(q)
    if( overwrite ):
        f = open(filename,'w')
    else:
        f = open(filename,'w-')
        
    pickle.dump(dict(zmin=zmin, zmax=zmax, nz=nz, q=q, z=z, F_re = F_re, F_re_err = F_re_err, F_im = F_im),f)
    f.close()

def create_interp_Fqz(filename = DefaultTableFile):
    """create the interpolated function based on the table value stored in file.close, return a tuple contains (Fqz_real, Fqz_imag)

    filename : string; the full path of the table file
    
    """
    with open(filename,'r') as f:
        F_dict = pickle.load(f)
    z = F_dict['z']
    F_re = F_dict['F_re']
    F_im = F_dict['F_im']
    Fqz_real = InterpolatedUnivariateSpline(z, F_re)
    Fqz_imag = InterpolatedUnivariateSpline(z, F_im)
    return (Fqz_real,Fqz_imag)



    
    
    
    
    
    
    
    
    

