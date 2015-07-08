from scipy.io.netcdf import netcdf_file
import numpy as np
import matplotlib.pyplot as plt

eqf = netcdf_file('/p/gkp/lshi/XGC1_NSTX_Case/new_3D_fluctuations/time_step_8/eqfile108.cdf','r')

ne = eqf.variables['ne']
r = eqf.variables['rr'][:]
nz = eqf.dimensions['nz']
midz = (nz-1)/2

ne_midz = ne[midz,:]

freqs = np.array([30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,65,66.5,67.5,70,72.5,75])*1e9

ref_ne = (freqs/8.98e3)**2 *1e6 

ref_lines = np.zeros((2,len(freqs)))+ref_ne

bot_range = [2,8]
top_range = [8,14]

def plot():

    fig = plt.figure()

    plt.plot(r,ne_midz)

    plt.plot(r[[0,-1]],ref_lines[:,0],'b-.',label = 'outer')
    plt.plot(r[[0,-1]],ref_lines[:,1:bot_range[0]],'b-.')
    plt.plot(r[[0,-1]],ref_lines[:,bot_range[0]],'b-',label = 'lower pedestal')
    plt.plot(r[[0,-1]],ref_lines[:,bot_range[0]+1:bot_range[1]],'b-')
    plt.plot(r[[0,-1]],ref_lines[:,top_range[0]],'b--',label = 'upper pedestal')
    plt.plot(r[[0,-1]],ref_lines[:,top_range[0]+1:top_range[1]],'b--')
    plt.plot(r[[0,-1]],ref_lines[:,top_range[1]],'b:',label = 'inner')
    plt.plot(r[[0,-1]],ref_lines[:,top_range[1]+1:],'b:')
    
    
    plt.legend()
    plt.title('NSTX Reflectometry Layout')
    plt.xlabel('$R(M)$')
    plt.ylabel('$ne(m^{-3})$')
    
    
