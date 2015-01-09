"""Spectra Analysis tools

This module contains functions that do spectra analysis on given 'time' series data. 
"""

from numpy.fft import rfft,fft,fftfreq,fftshift
import numpy as np

def spectra_over_time(data,time,fft_window, tstart = 0,tend = -1):
    """ The function creates frequency versus time data. Need to specify the window width(in term of time steps) over which each fft is calculated. The fft result is assigned to the center time as the spectrum of that time.  Can specify the start and end time, by default, the whole data set will be used.
    Inputs:
        data: 1D array, time series of the data under investigation
        time: 1D array, same length as data, the corresponding 'time' labels of the data, has unit.
        fft_window: int, the number of time steps using for each fft calculation.
        tstart: int, optional, starting time for the analysis, default to be 0.
        tend: int, optional, ending time for the analysis, default to be -1, which means the last time step of the given data.
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
    NT = tend-tstart+1
    if NT<=0:
        raise Exception('Spectra Input Error: No time-spectra result can be obtained by given dataset for the chosen starting/ending time and the window width. Check if the ending time is greater than the starting time, and the time window width isn\'t too large for the limited length dataset.')

    #Note that the frequency grid number equals the time window width used for fft analysis
    NF = fft_window
    try:
        dt = time[1]-time[0]
    except IndexError:
        print 'The data has only one time step. No spectra analysis can be done.'
        raise

    print tstart,tend,NF,NT
    
    freq = fftfreq(NF,dt)
    time_spec = time[tstart:tend+1]
    spectra = np.empty((NT,NF)) + 1j*np.empty((NT,NF))
    
    for t in range(tstart,tend+1):
        begin = t - (fft_window-1)/2
        end = t + (fft_window+1)/2 + 1
        d = data[begin:end]
        f = fft(d)
        spectra[t-tstart,:] = f

    return (spectra,time_spec,freq)
        
        
