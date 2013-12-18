import FPSDP.Plasma.TestParameter as tp
from FPSDP.GeneralSettings.UnitSystem import cgs
import numpy as np
from FPSDP.Maths.Funcs import my_quad
from FPSDP.Maths.PlasmaDispersionFunction import *

plasma = tp.create_profile()

NR = plasma['Grid'].NR
NZ = plasma['Grid'].NZ

R1D = plasma['Grid'].R2D[0,:]
Z1D = plasma['Grid'].Z2D[:,0]

c = cgs['c']
m_e = cgs['m_e']
e = cgs['e']
keV = cgs['keV']

Zmid = NZ/2+1

ne = plasma['ne'][Zmid,:]
B = plasma['B'][Zmid,:]
Te = plasma['Te'][Zmid,:]

vt = np.sqrt(Te/m_e)
omega_c = e*B/(m_e*c)

omega = 2*omega_c[NR/2+1]


def CalcM12(k_ll,N,p):
    n = np.absolute(N)

    global omega
    global omega_c
    global vt
    alpha = m_e*c**2/Te
    psi = k_ll*c**2/(np.sqrt(2)*omega*vt)
    phi2 = (psi**2-alpha*(omega-N*omega_c)/omega)
    phi = np.sqrt(phi2)
    if(n+p == 1):
        raw = np.array([F52(phi[i],psi[i]).imag for i in range(len(phi))])
#        print raw
        result = np.select([phi2 >= 0],[raw])
#        print result
        return result
    elif(n+p == 2):
        raw = np.array([F72(phi[i],psi[i]).imag for i in range(len(phi))])
        result = np.select([phi2 >= 0], [raw])
        return result
    else:
        print 'wrong parameter, n+p should be 1 or 2.'
        return np.nan

def scan_k(raw_k):
    global NR
    Nk = len(raw_k)
    k = np.select([raw_k != 0, raw_k == 0],[raw_k,1e-5])
    result = np.zeros((Nk,NR))
    for i in range(Nk):
        result[i,:]=CalcM12(k[i],2,0)

    return result



