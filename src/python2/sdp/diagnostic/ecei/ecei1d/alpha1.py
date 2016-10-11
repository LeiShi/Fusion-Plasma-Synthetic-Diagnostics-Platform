"""Module contains Functions that calculate the local absorption coefficient 
alpha.

Method 1:
Analytical expression for alpha is used. Assume weakly relativistic Maxwellian 
distriution, and weak absorption. [1]_

.. [1] 1983 Nucl. Fusion 23 1153
"""

import pickle

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline 
from scipy.special import gamma
from scipy import select
import numpy as np

from ....settings.unitsystem import cgs
from ....math.pdf import Fq
#The default path and filename for the file that stores the Fqz tables
DefaultFqzTableFile = './Fqz.dat'

def make_frequency_table(Profile, Harmonic = 2 ,ResOmega = None):
    """make the frequency table based on the Profile data, namely the B field 
    range on Grid.

    :param Profile: the plasma profile data. 
    :type Profile: :py:class:`sdp.plasma.profile.ECEI_Profile` object
    :param Harmonic: an integer indicates the targeting harmonic mode. 
                     default to be the second harmonics.
    :param int ResOmega: the number of grids on frequency table. 
                         Default to be ``Profile.grid.NR``.
    """

    Bmax = np.max(Profile.B0)
    Bmin = np.min(Profile.B0)
    
    Omega_max = cgs['e'] * Harmonic/(cgs['m_e'] * cgs['c']) * Bmax
    Omega_min = cgs['e'] * Harmonic/(cgs['m_e'] * cgs['c']) * Bmin
    if(ResOmega is None):
        NOmega = Profile.grid.NR
    else:
        # make sure the Omega mesh is finer than the desired resolution
        NOmega = np.floor((Omega_max - Omega_min)/ResOmega) + 2 

    return np.linspace(Omega_min,Omega_max,NOmega)



def create_Fqz_table(zmin=-30., zmax=30., nz=1001, q=3.5, 
                     filename=DefaultFqzTableFile, overwrite = True):
    """create the F_q(z_n) function value table using exact integration and 
    summation formula[1]. Save the results into a file.  

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
        F_re[i],F_re_err[i] = quad(lambda x: \
                                   (-1j*np.exp(1j*z[i]*x)/(1-1j*x)**q).real, 
                              0, np.inf, epsrel = 1e-8,epsabs = 1e-10, 
                              limit = 500)
        if( z[i] < 0):
            F_im[i] = -np.pi*(-z[i])**(q-1)*np.exp(z[i])/gamma(q)
    if( overwrite ):
        f = open(filename,'w')
    else:
        f = open(filename,'w+')
        
    pickle.dump(dict(zmin=zmin, zmax=zmax, nz=nz, q=q, z=z, F_re=F_re, 
                     F_re_err=F_re_err, F_im=F_im),f)
    f.close()

def create_interp_Fqz(filename = DefaultFqzTableFile):
    """create the interpolated function based on the table value stored in 
    file.close, return a tuple contains (Fqz_real, Fqz_imag)

    filename : string; the full path of the table file
    """
    with open(filename,'r') as f:
        F_dict = pickle.load(f)
    z = F_dict['z']
    z_min = F_dict['zmin']
    z_max = F_dict['zmax']
    F_re = F_dict['F_re']
    F_im = F_dict['F_im']

    # raw interpolated functions, need to be screened outside (z_min, z_max) 
    # range
    Fqz_real_raw = InterpolatedUnivariateSpline(z, F_re)
    Fqz_imag_raw = InterpolatedUnivariateSpline(z, F_im)

    # screen out the outside part, set exponential decay outside the z range, 
    # if z>zmax, f(z) = f(zmax) * exp(-2(z-zmax)/(zmax-zmin)), 
    # if z<zmin, f(z) = f(zmin) * exp(-2(zmin-z)/(zmax-zmin))
    def Fqz_real(z):
        z_scr = select([z<z_min,z>z_max,z>=z_min] , [z_min,z_max,z])
        mask = select( [z<z_min-20*(z_max-z_min), z>z_max+20*(z_max-z_min), 
                        z<z_min, z>z_max, z>=z_min], 
                       [np.exp(-40) , np.exp(-40) , 
                        np.exp(-2*(z_min-z)/(z_max-z_min)), 
                        np.exp(-2*(z-z_max)/(z_max-z_min)), 1]  )
        return Fqz_real_raw(z_scr) * mask
    def Fqz_imag(z):
        z_scr = select([z<z_min,z>z_max,z>=z_min] , [z_min,z_max,z])
        mask = select( [z<z_min-20*(z_max-z_min), z>z_max+20*(z_max-z_min), 
                        z<z_min, z>z_max, z>=z_min], 
                       [np.exp(-40), np.exp(-40), 
                        np.exp(-2*(z_min-z)/(z_max-z_min)), 
                        np.exp(-2*(z-z_max)/(z_max-z_min)), 1])
        return Fqz_imag_raw(z_scr) * mask
    return (Fqz_real,Fqz_imag)

def get_alpha_table(SpecProfile , n=2):
    """Main function that calculates the alpha coefficients.

    :param SpecProfile: Contains the frequency band array, and the plasma 
                        profile data. 
    :type SpecProfile: Dictionary with keywrods ``omega`` and ``Profile``.
    1. ``omega``
        float array contains selected frequencies on which 
        detector gain is specified. See 
        :py:class:`.Detector.Detector` for more details
    2. ``Profile``
        dictionary containing 'ne','Te' and 'B' along light path
    :param n: an integer indicates the targeting harmonic mode. 
              default to be the second harmonics.
    """
    # define local names for physical constants
    e = cgs['e']
    m_e = cgs['m_e']
    c = cgs['c']
    
    # define the local names, expand 1D into 2D, dimension order: [F,s] 
    # F:frequency s: light path length
    Profile = SpecProfile['Profile']
    ne,Te,B = Profile['ne'][np.newaxis,:] , Profile['Te'][np.newaxis,:], \
              Profile['B'][np.newaxis,:]
    
    # calculate frequency table, expand to 2D for later use
    omega = SpecProfile['omega'][:,np.newaxis]
    omega2 = omega**2

    # now calculate all the useful local quantities on the grid
    # plasma frequency is on RZ grid, i.e. 1D line, but expands to 2D 
    # Note that the dimension order convention is (F,s)
    omega2_p = 4*np.pi*ne*e**2/m_e
    
    # electron cyclotron frequency is also on 1D path, but naturally expands 
    # to 2D as B did
    omega_c = e*B/(m_e*c)
    omega2_c = omega_c**2
    
    # the ratio between omega2_p and omega2_c is frequently used
    omega2_pc_ratio = omega2_p/omega2_c

    # z values, which measures the distance from resonance, taken thermal 
    # effect into account. It's a function of omega, so 2D
    z = c**2 * m_e/Te *(omega - n*omega_c)/omega
    
    # Fq function is function of phi instead of z
    phi = np.sqrt((-1+0j)*z)
    psi = np.zeros_like(phi)
    
    # Calculate Fq(z) values using new Fq functions
    Fqz = Fq(phi, psi, 2*n+3)
    F_re = np.real(Fqz)
    F_im = np.imag(Fqz)
    F_cplx = F_re + F_im*1j
    
    #refraction index N_perp is a function of frequency, such that on 3D space
    #real part of N_perp_c squared, as defined in ref[1] Eq. 3.1.12
    N2_perp_c = 1 - (omega2_p/omega2) * (omega2 - omega2_p)/\
                    (omega2 - omega2_c - omega2_p)
    #case 1, n=2
    if(n == 2):        
        # local names a,b are used here, they are the same as in ref.[1]
        # Eq.3.1.18
    
        # just the real parts of a,b are used
        a = -0.5*(omega2_pc_ratio) * (omega2-omega2_c)/\
                 (omega2-omega2_c-omega2_p)* F_cplx
        b = -2*(1- omega2_p/(omega*(omega+omega_c)))*a
        N2_perp_plus = (-(1+b)+((1+b)**2 + 4*a*N2_perp_c)**0.5)/(2*a)
        N_perp_plus_re = np.sqrt(N2_perp_plus).real
        
        # a_2, b_2 are used in A_n, and defined in Eq. 3.1.20 and 3.1.38
        a_2 = 0.5*omega2_pc_ratio *(1+ 3* N2_perp_plus * F_cplx) /\
              (3-(omega2_pc_ratio)*(1+1.5*N2_perp_plus*F_cplx))
        b_2 = 1/np.absolute(1+0.5*omega2_pc_ratio*(1+a_2)**2*F_re)
        a_2_re = np.absolute(a_2)
        
        # now calculate A_2 and alpha_2_o, as Eq. 3.1.37 and 3.1.36
        A_2 = N_perp_plus_re * np.absolute(1+a_2)**2 * b_2
        
        # note that n**(2n-1)/(2**n * n!) = 1 when n=2 and vt/c term disappears
        alpha_2_o = omega2_pc_ratio*omega_c/c*(-F_im)
        
        # exponential thermal correction as given in Eq. 3.3.4
        a_n = omega2_pc_ratio/(n*(n**2-1-omega2_pc_ratio))
        gamma_2 = 0.75 - 2*a_2_re/(1+a_2_re) + 8./7*(1+1/(1+a_2_re))*N_perp_plus_re**2  
        
        # finally alpha_2
        alpha_2 = A_2 * alpha_2_o *np.exp(gamma_2 * (1- n*omega_c/omega))
        return alpha_2
    else:
        # for n >= 3,
        # N_perp squared is just the cold limit value ,i.e. the N2_perp_c in 
        # former case

        N_perp_plus_re = np.sqrt(N2_perp_c).real
        # a_n is used in A_n, and defined in Eq. 3.1.14b and 3.1.38
        a_n = omega2_pc_ratio/(n*(n**2-1-omega2_pc_ratio))
        # now calculate A_n and alpha_n_o, as Eq. 3.1.37 and 3.1.36
        A_n = N_perp_plus_re * (1+a_n)**2 
        # note that gamma here is the special gamma function, which essentially
        # gives n!
        alpha_n_o = n**(2*n-1)/( 2**n*gamma(n+1) ) *omega2_pc_ratio * \
                    (m_e/(Te*c**2))**(n-2) *omega_c/c*(-F_im)
        # exponential thermal correction as given in Eq. 3.3.4
        gamma_n = 0.75 - 2*a_n/(1+a_n) + 8./7*(1+1/(1+a_n))*N_perp_plus_re**2  
        
        # finally alpha_n
        alpha_n = A_n * alpha_n_o *np.exp(gamma_n * (1- n*omega_c/omega))    
        return alpha_n
    
