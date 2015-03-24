"""simple math functions used for debugging and/or productive runs
"""

import numpy as np

def my_quad(y,x):
    """quadratic integration on given grids
        I = Sum (y[i+1]+y[i])*(x[i+1]-x[i])/2  
    """
    I = 0.
    for i in range(len(x)-1):
        I += (y[i+1]+y[i])*(x[i+1]-x[i])/2.
        
    return I

def determinent3d(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    """calculate the determinent of 3*3 matrix
    """

    return (x1*y2*z3 + y1*z2*x3 + z1*x2*y3 - x1*z2*y3 - y1*x2*z3 - z1*y2*x3)


def sweeping_correlation(s1,s2,dt=1,nt_min=100):
    """Calculate the correlation of two given time-series signals.
    
    Correlation is defined as:

    gamma(s1,s2) = <(s1_tilde * conj(s2_tilde))>/sqrt(<|s1_tilde|^2> <|s2_tilde|^2>)

    where s1_tilde = s1 - <s1>, <...> denotes time average.

    Sweeping correlation is carried out by correlating one signal to a delayed(or advanced) version of the other signal.
    
    Arguments:
        s1,s2: signals to be correlated, ndarray with same shape, the first dimension is "time".
        dt: int, sweeping step size, move s2 dt units in time every step, and carryout another correlation with s1
        nt_min: optional, int, the minimum time overlap for sweeping correlation, the average must be taken over longer time period than set by this, otherwise sweeping will stop. Default to be 100. 
        
    Returns:
        SCorrelation: ndarray, same shape as s1 and s2 except for the first dimension, the first dimension length will be total number of sweeping correlations, it's determined by dt, nt_min, and the original time series length. Indexing convention for time dimension is similar to that in fft, if total length is 2n+1, index 0 is for correlation without moving, index 1 to n for s2 delayed compared to s1, index -1 to -n for s2 advanced compared to s1.
        
    """

    assert (s1.shape == s2.shape),'Shapes of two signals don\'t match. s1:{0},s2:{1}'.format(str(s1.shape),str(s2.shape))
    
    shape = s1.shape
    nt = shape[0]
    assert (nt >= nt_min ),'signal length {0} is shorter than minimum length: {1}.'.format(nt,nt_min)
    spatial_shape = shape[1:]
    
    n_wing = int((nt-nt_min)/dt) #length of single wing of the result    
    
    n_sweep = n_wing*2 + 1 #total sweep correlation numbers
    
    SCorrelation = np.empty((n_sweep,)+spatial_shape) #concatenate last dimension to spatial dimensions
    
    #first get rid of the mean signal
    s1 = s1 - np.mean(s1,axis=0)
    s2 = s2 - np.mean(s2,axis=0)    
    
    for i in range(-n_wing,n_wing+1):
        delta_t = dt*i
        if delta_t < 0:#when s2 is moved advance to s1
            s1_moved = s1[:delta_t,...]
            s2_moved = s2[-delta_t:,...]
            SCorrelation[i] = np.average(s1_moved*np.conj(s2_moved),axis=0)/np.sqrt(np.average(s1_moved*np.conj(s1_moved),axis=0) * np.average(s2_moved*np.conj(s2_moved), axis=0))
        elif delta_t == 0:#when not moved
            SCorrelation[i] = np.average(s1*np.conj(s2),axis=0)/np.sqrt(np.average(s1*np.conj(s1),axis=0) * np.average(s2*np.conj(s2),axis=0))
        else: #when s2 is delayed to s2
            s1_moved = s1[delta_t:,...]
            s2_moved = s2[:-delta_t,...]
            SCorrelation[i] = np.average(s1_moved*np.conj(s2_moved),axis=0)/np.sqrt(np.average(s1_moved*np.conj(s1_moved),axis=0) * np.average(s2_moved*np.conj(s2_moved), axis=0))
    return SCorrelation
        
    