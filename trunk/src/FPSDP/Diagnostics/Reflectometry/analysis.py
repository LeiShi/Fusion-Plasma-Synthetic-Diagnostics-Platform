"""General analysis functions for reflectometry related data

Contains:
    phase(raw_sig) : processes the raw signal's phase, creates a new phase series such that the phase change in one time step is not larger than PI. In this way, the phase jump across -PI and PI boundary is avoided. The phase curve is more smooth and the range is also extended to (-inf,inf). 
    magnitude(raw_sig): obtains raw signal's magnitude. Equivalent to np.abs .
    
    Self
"""

import numpy as np

###########################################################
#  Two functions used for analysing the time series of complex reflected signal
###########################################################

def phase(raw_sig):
    """Calculate the extended phase curve for a given complex signal array.

    The purpose of extending the phase range to (-infi,+infi) is to avoid jumps from +pi -> -pi or the other way around on the normal [-pi,pi) range. In this case, the phase curve looks much smoother and more meaningful.
    The method we are using is first calculate the phase for each time step in the normal [-pi,pi) range, then, calculate the phase change for each time interval : dphi. For dphi>pi, we pick dphi-2*pi as the new phase change; and for dphi < -pi, we pick dphi+2*pi. In other words, we minimize the absolute value of the phase change. This treatment is valid if time step is small compared to plasma changing time scale, so the change of reflected phase shouldn't be very large.

    Arguments:
        raw_sig: 1d complex array, the complex signal gotten from either measurement or synthetic reflectometry runs.
    Return:
        (new_phase,new_dph) tuple: new_phase is the modified phase series, new_dph is the phase change series.
    """
    phase_raw = np.angle(raw_sig) # numpy.angle function gives the angle of a complex number in range[-pi,pi)

    dph = phase_raw[1:]-phase_raw[0:-1] #the phase change is defined on each time intervals, so the total length will be 1 shorter than the phase array.

    dph_ext = np.array([dph-2*np.pi,dph,dph+2*np.pi]) #intermediate array that contains all 3 posibilities of the phase change

    dph_arg = np.argmin(np.abs(dph_ext),axis = 0) #numpy.argmin function pick out the index of the first occurance of the minimun value in the array along one chosen axis. Since the axis 0 in our array has just 3 elements, the dph_arg will contain only 0,1,2's.

    new_dph = dph + (dph_arg-1)*2*np.pi # notice that in dph_arg, 0 corresponds dph-2*pi being the chosen one, 1 -> dph, and 2 -> dph+2*pi, therefore, this expression is valid for all 3 cases.

    new_phase = np.zeros_like(phase_raw)+phase_raw[0]
    new_phase[1:] += new_dph.cumsum() # numpy.ndarray.cumsum method returns the accumulated array, since we are accumulating the whole dph_new array, the phase we got is relative to the initial phase at the start of the experiment.

    return (new_phase,new_dph)

def magnitude(raw_sig):
    """Calculate the magnitude of the raw signal
    
    This is simple because the magnitude is single valued on the whole complex plane, so no special treatment is needed.
    
    Arguments:
        raw_sig: array-like, complex, time series of the complex signal
    Return:
        magnitude: array-like, float, same shape as raw_sig, the magnitude of the complex signals.
    """
    
    return np.abs(raw_sig)
    

#####################################################################
# Coherent Signal and Cross Correlation analysis
#####################################################################

    
def Coherent_Signal(sig):
    """Calculate the coherent signal g for the given series of signal

    coherent signal g is defined as:

    g = <sig>/sqrt(<|sig|^2>)

    where <...> denotes the emsemble average, which in this case, is calculated by averaging over all time steps(whole array). And sig is the signal.

    input: 
        sig: array-like, complex or real. The time series or whole ensemble of the  signal.
    """
    
    #rename the signal to a shorter form
    M = sig

    M_bar = np.average(M)

    M2_bar = np.average(M*np.conj(M))

    return M_bar/np.sqrt(M2_bar)

def Cross_Correlation(sig1,sig2,mode='REF'):
    """Calculate the cross correlation between 2 series of signals

    cross correlation function is defined as:

    r = <sig1 * conj(sig2)> / sqrt(<|sig1|^2><|sig2|^2>)

    where conj means complex conjugate, and <...> denotes the average over timesteps (whole array)

    Input:
        sig1,sig2: array-like, complex or real. The two signals that need to be cross-correlated.
        mode: default to be 'REF', denotes reflectometry mode. In this mode, two signals are complex signal who's phase fluctuations are of most importance, no modification will be done before correlation.
              another valid mode is 'NORM', denotes normal mode. In this mode, two signals' mean value will be subtracted before correlation. So only the fluctuated part will be correlated.
    Return:
        the cross-correlation calculated at zero time delay
    """
    
    assert(sig1.shape == sig2.shape)
    assert(mode in ['REF','NORM'])
    #remove the mean value to get the fluctuating part of the two signal
    if (mode == 'NORM'):
        sig1 = sig1-np.mean(sig1)
        sig2 = sig2-np.mean(sig2)    
    
        
    sig1_2_bar = np.average(sig1 * np.conj(sig1))
    sig2_2_bar = np.average(sig2 * np.conj(sig2))
    cross_bar = np.average(sig1 * np.conj(sig2))

    r = cross_bar / np.sqrt(sig1_2_bar * sig2_2_bar)       
    
    return r

def Cross_Correlation_by_fft(sig1,sig2,mode = 'REF'):
    """Calculate the cross correlation using fft method. Details can be found in ref.[1] and in Appendix part of ref.[2]
    
    The strength of calculating cross_correlation using fft method, is that it can obtain all time delayed correlations automatically, with a very fast calculation.
    
    Input:
        sig1,sig2: array-like, complex or real. The two signals that need to be cross-correlated
        mode: default to be 'REF', denotes reflectometry mode. In this mode, two signals are complex signal who's phase fluctuations are of most importance, no modification will be done before correlation.
              another valid mode is 'NORM', denotes normal mode. In this mode, two signals' mean value will be subtracted before correlation. So only the fluctuated part will be correlated.

    Return:
        gamma_t: array-like, the cross-correlation of the two signals, gamma_t[t] will be the t time-delayed correlation, in which sig2 is put t time steps ahead of sig1, and they are both periodically extended to (-inf,inf)
    
    [1] Cross-correlation caluclation using Fast Fourier Transform(FFT), FPSDP.Diagnostics.Reflectometry Documentation. 
    [2] Observation of ion scale fluctuations in the pedestal region during the edge-localized-mode cycle on the National Spherical torus Experiment. A.Diallo, G.J.Kramer, at. el. Phys. Plasmas 20, 012505(2013)
    """
    assert(sig1.shape == sig2.shape)    
    assert(mode in ['REF','NORM'])
    
    if(mode == 'NORM'):
        #remove mean value of the two signals
        sig1 = sig1-np.mean(sig1)
        sig2 = sig2-np.mean(sig2)    
        
    
    f1 = np.fft.fft(sig1)
    f2 = np.fft.fft(sig2)
    norm1 = np.sqrt(np.sum(sig1*np.conj(sig1)))
    norm2 = np.sqrt(np.sum(sig2 * np.conj(sig2)))
    
    cross_f = np.conj(f1) * f2
    gamma_f = cross_f / (norm1 * norm2)
    gamma_t = np.fft.ifft(gamma_f)
    
    return gamma_t
    
    
    
