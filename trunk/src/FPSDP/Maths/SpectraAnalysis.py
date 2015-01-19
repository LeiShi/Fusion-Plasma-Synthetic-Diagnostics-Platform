"""Spectra Analysis tools

This module contains functions that do spectra analysis on given 'time' series data. 
"""

from numpy.fft import rfft,fft,fftfreq,fftshift
import numpy as np

def Short_Time_Fourier_Transform(data,time,fft_window, tstart = 0,tend = -1, overlap_rate = 0.5):
    """ The function creates frequency versus time data. Need to specify the window width(in term of time steps) over which each fft is calculated. The fft result is assigned to the center time as the spectrum of that time.  Can specify the start and end time, by default, the whole data set will be used.
    Inputs:
        data: 1D array, time series of the data under investigation
        time: 1D array, same length as data, the corresponding 'time' labels of the data, has unit.
        fft_window: int, the number of time steps using for each fft calculation.
        tstart: int, optional, starting time for the analysis, default to be 0.
        tend: int, optional, ending time for the analysis, default to be -1, which means the last time step of the given data.
        overlap_rate: double, (0,1), optional, the overlap ratio between adjacent FFT windows. It is easy to show that delta_t = (1-overlap_rate)*fft_window is the corresponding window center displacement. 
    Outputs:
        (Outputs are returned as a tuple of the following quantities)
        spectra: 2D complex array with shape (NT,NF), NT is the total time steps of the spectra-time analysis, NF is the frequency grids obtained for each time. The array contains all the coefficients obtained from fft.
        time_spec: 1D array of length NT, the center time of each spectrum.
        freq: 1D array of length NF, the frequency grid
    """

    if(len(data) != len(time)):
        raise Exception('Spectra Input Error:Time array doesn\'t match data array, please make sure these two arrays have the same length!')

    if tend<0:
        tend += len(data)
    # reset the start and end time so that they are valid center time for the given window width
    if (tstart < (fft_window-1)/2):
        tstart = (fft_window-1)/2
    if (tend + (fft_window+1)/2 > len(data)-1 ):
        tend = len(data)-1 - (fft_window+1)/2
    if (overlap_rate>=1 ):
        print('Overlap_rate Error: overlap_rate cannot be greater or equal to 1. It is reset to 0.5')
        overlap_rate = 0.5
    if (overlap_rate<0):
        #note that a negative overlap may be considered as a indicator of disconnected FFT windows, however this will be a waste of original data! Here we enforce the full use of given data, so reset the overlap_rate to 0
        overlap_rate = 0
    delta_t = int((1-overlap_rate)*fft_window)
    t_centers = np.arange(tstart,tend+1,delta_t)
    NT = len(t_centers)
    if NT==0:
        raise Exception('Spectra Input Error: No time-spectra result can be obtained by given dataset for the chosen starting/ending time and the window width. Check if the ending time is greater than the starting time, and the time window width isn\'t too large for the limited length dataset.\n tstart={0},tend={1},window={2},overlap_rate={4}'.format(tstart,tend,fft_window,overlap_rate))

    #Note that the frequency grid number equals the time window width used for fft analysis
    NF = fft_window
    try:
        dt = time[1]-time[0]
    except IndexError:
        print 'The data has only one time step. No spectra analysis can be done.'
        raise

    print tstart,tend,NF,NT
    
    freq = fftshift(fftfreq(NF,dt))
    time_spec = time[t_centers]
    spectra = np.empty((NF,NT)) + 1j*np.empty((NF,NT))
    
    for i in range(NT):
        t = t_centers[i]
        begin = t - fft_window/2 +1
        end = t + fft_window/2 +1 
        d = data[begin:end]
        f = fft(d)
        spectra[:,i] = fftshift(f)

    return (spectra,time_spec,freq)
        
        
