import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp
from FPSDP.GeneralSettings.UnitSystem import cgs
import numpy as np
from FPSDP.Maths.Funcs import my_quad
import FPSDP.Maths.PlasmaDispersionFunction as pdf

plasma = tp.create_profile2D()

NR = plasma.grid.NR
NZ = plasma.grid.NZ

R1D = plasma.grid.R2D[0,:]
Z1D = plasma.grid.Z2D[:,0]

c = cgs['c']
m_e = cgs['m_e']
e = cgs['e']
keV = cgs['keV']

Zmid = NZ/2+1

ne = plasma.ne[Zmid,:]
B = plasma.B[Zmid,:]
Te = plasma.Te[Zmid,:]

vt = np.sqrt(Te/m_e)
omega_c = e*B/(m_e*c)
alpha = m_e*c**2/Te

N = 2

omega = N*omega_c[NR/2+1]

def show_omega():
    print omega
    


def CalcM12(k_ll,N,p):
    n = np.absolute(N)

    global omega
    global omega_c
    global vt

    psi = k_ll*c**2/(np.sqrt(2)*omega*vt)
    phi2 = (psi**2-alpha*(omega-N*omega_c)/omega)
    phi = np.sqrt(phi2)
    if(n+p == 1):
        raw = np.array([pdf.F52(phi[i],psi[i]).imag for i in range(len(phi))])
#        print raw
        result = np.select([phi2 >= 0],[raw])
#        print result
        return result
    elif(n+p == 2):
        raw = 4*np.array([pdf.F72(phi[i],psi[i]).imag for i in range(len(phi))])
        result = np.select([phi2 >= 0], [raw])
        return result
    else:
        print 'wrong parameter, n+p should be 1 or 2.'
        return np.nan
        
def CalcM13(k_ll,N,p):
    n = np.absolute(N)
    psi = k_ll*c**2/(np.sqrt(2)*omega*vt)
    phi2 = (psi**2-alpha*(omega-N*omega_c)/omega)
    phi = np.sqrt(phi2)
    if(n+p == 1):
        raw = k_ll*c/omega_c * np.array([pdf.F72_1(phi[i],psi[i]).imag for i in range(len(phi))])
#        print raw
        result = np.select([phi2 >= 0],[raw])
#        print result
        return result
    elif(n+p == 2):
        raw = 2*k_ll*c/omega_c * np.array([pdf.F92_1(phi[i],psi[i]).imag for i in range(len(phi))])
        result = np.select([phi2 >= 0], [raw])
        return result
    else:
        print 'wrong parameter, n+p should be 1 or 2.'
        return np.nan
        

def scan_k_12(raw_k):
    global NR
    global N
    Nk = len(raw_k)
    k = np.select([raw_k != 0, raw_k == 0],[raw_k,1e-5])
    result = np.zeros((Nk,NR))
    for i in range(Nk):
        result[i,:]=CalcM12(k[i],N,0)
    return result

def scan_k_13(raw_k):
    global NR
    global N
    Nk = len(raw_k)
    k = np.select([raw_k != 0, raw_k == 0],[raw_k,1e-5])
    result = np.zeros((Nk,NR))
    for i in range(Nk):
        result[i,:]=CalcM13(k[i],N,0)
    return result

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def draw3D(k,M):
    global R1D
    x = R1D[66:133]
    fig = plt.figure()
    ax = Axes3D(fig)
    X,K = np.meshgrid(x,k)
    ax.plot_surface(X,K,M[:,66:133],cmap=cm.coolwarm,antialiased = False)
    return fig
    
    

