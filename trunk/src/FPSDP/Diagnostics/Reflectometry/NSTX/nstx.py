""" Reading and post-processing functions for NSTX correlation reflectometry output. Raw data given by Dr. Ahmed Diallo.

Program by Lei Shi, 05/16/2014

modules needed: h5py, numpy, scipy
"""

import h5py as h5
import numpy as np
from scipy.interpolate import interp1d

class NSTX_REF_Loader:
    """ Loader class which contains all the reading and post-processing methods
    """

    def __init__(this,filename):
        """initialize with a hdf5 filename, should be a raw data output from NSTX reflectometry measurement.

        
        """
        this.filename = filename
        f = h5.File(filename,'r')
        this.t0 = f['mydata'][0]['T0']
        this.dt = f['mydata'][0]['DT']
        this.freq = f['mydata'][0]['FREQUENCY']*1e-9 #Change to GHz
        this.nt = len(f['mydata'][0]['INPHASE'])
        f.close()    
    def getI(this):
        """returns the inphase component of the reflectometry signal 
        """
        f = h5.File(this.filename,'r')
        I = f['mydata'][0]['INPHASE']
        f.close()
        return I
        
    def getQ(this):
        """returns the out of phase component of the reflectometry signal 
        """
        f= h5.File(this.filename,'r')
        Q = f['mydata'][0]['QUADRATURE']
        f.close()
        return Q
    
    def getT(this):
        """returns the time array with the same shape as I and Q
        """

        return this.t0 + this.dt*np.arange(this.nt)
            

class FFT_result:
    """Contains returned arrays from fft analysis

    Attributes:
        origin: original time series data
        shift_fft: array after fft, and shifted so that zero frequency is located in middle
        t: time array corresponds to original data
        f: frequency array corresponds to fft data
    """

    def __init__(this, origin,shift,t,f):
        this.origin = origin
        this.shift_fft = shift
        this.t = t
        this.f = f

class Analyser:
    """ Contains all the Post-process methods 
    """

    def __init__(this, nstx_loaders):
        """ Initialize with an NSTX_REF_loader array
        """
        this.loaders = nstx_loaders

    def fft(this,tol = 1e-5, **params):
        """FFT analysis in time.
        arguments:
            keyword list:
            1)Time steps can be given by either of the following ways:
                tstart,tend,nt: time steps = np.linspace(tstart,tend,nt)
                tstart,dt,nt: time step = tstart + np.arange(nt)*dt
            2)Loader is specified by either of the following ways:
                loader_num : loader = this.loaders[loader_num]
                frequency : check if abs(loader.freq-frequency)/frequency<tol, if find one, then use this loader, if not, raise an error.
            3)Chose the In phase component or Quadrature component
                component = 'I' or 'Q'
        returns:
            FFT_result object.
        """

        if ('tend' in params.keys()):
            t = np.linspace(params['tstart'],params['tend'],params['nt'])
        else:
            t = params['tstart'] + np.arange(params['nt']) * params['dt']

        if('loader_num' in params.keys()):
            loader = this.loaders[loader_num]
        else:
            loader_found = False
            for l in this.loaders:
                if(np.abs(params['Frequency']-l.freq)/float(l.freq) < tol):
                    loader = l
                    loader_found = True
                    break
            if(not loader_found):
                raise Exception('fft initialization error: no matching frequency data')

        if(params['component'] == 'I'):
            raw_data = loader.getI()
        elif(params['component'] == 'Q'):
            raw_data = loader.getQ()
        else:
            raise Exception('fft initialization error: component must be either "I" or "Q"')

        raw_t = loader.getT()

        interp = interp1d(raw_t,raw_data)

        origin = interp(t)
        f = np.fft.fftshift(np.fft.fftfreq(len(t),t[1]-t[0]))
        shift_fft = np.fft.fftshift(np.fft.fft(origin))

        return FFT_result(origin,shift_fft,t,f)
        
