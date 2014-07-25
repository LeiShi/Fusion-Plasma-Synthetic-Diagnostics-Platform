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
                component = 'I', 'Q', 'Amp' or 'Phase' 
        returns:
            FFT_result object.
        """

        if ('tend' in params.keys()):
            t = np.linspace(params['tstart'],params['tend'],params['nt'])
        else:
            t = params['tstart'] + np.arange(params['nt']) * params['dt']

        if('loader_num' in params.keys()):
            loader = this.loaders[params['loader_num']]
        else:
            loader_found = False
            for l in this.loaders:
                if(np.abs(params['frequency']-l.freq)/float(l.freq) < tol):
                    loader = l
                    loader_found = True
                    break
            if(not loader_found):
                raise Exception('fft initialization error: no matching frequency data')

        if(params['component'] == 'I'):
            raw_data = loader.getI()
        elif(params['component'] == 'Q'):
            raw_data = loader.getQ()
        elif(params['component']=='Amp'):
            signal = loader.getI()+loader.getQ()* 1j
            raw_data = np.absolute(signal)
        elif(params['component']=='Phase'):
            raw_data = np.angle(signal)
        else:
            raise Exception('fft initialization error: component must be either "I" or "Q"')

        raw_t = loader.getT()

        interp = interp1d(raw_t,raw_data)

        origin = interp(t)
        f = np.fft.fftshift(np.fft.fftfreq(len(t),t[1]-t[0]))
        shift_fft = np.fft.fftshift(np.fft.fft(origin))

        return FFT_result(origin,shift_fft,t,f)


    def Self_and_Cross_Correlation(this,tstart,tend):
        """Calculate the self_correlation and cross_correlation between channels provided in this.loaders, in the given time inteval.

        self_correlation function g(w) is defined as: (ref.[1])
        g(w)=<M(w)>/sqrt(<|M(w)|^2>)

        cross_correlation function r(w0,w1) is defined as:(see ref.[1])
        r(w0,w1) = < M(w0)M(w1) >/ sqrt(<|M(w0)|^2> <|M(w1)|^2>)

        where M(w) is the complex received signal for channel with frequency w, <...> denotes the ensemble average, which in this case, is the average over all time steps.

        arguments: tstart, tend: start and end time for calculation, unit: second.

        Returns: tuple of two components:
        (  1D array (n), contains all the self correlation results;
           2D array (n,n), where row (n0,:) is the cross-correlation of channel n0 with respect to all n channels. The diagonal terms should always be 1. )

        Reference:
        [1] Two-dimensional simulations of correlation reflectometry in fusion plasmas, E.J. Valeo, G.J. Kramer and R. Nazikian, Plasma Phys. Control. Fusion 44(2002)L1-L10
        """

        nf = len(this.loaders)
        
        #first load all the signals from the loaders

        M = []
        
        for i in range(nf):
            loader = this.loaders[i]
            I = loader.getI()
            Q = loader.getQ()
            sig = I + 1j*Q
            nstart = int( (tstart-loader.t0)/loader.dt )
            nend = int( (tend-loader.t0)/loader.dt )
            
            M.append(sig[nstart:nend])

        M = np.array(M)
        M_bar = np.average(M,axis = 1)
        M2_bar = np.average(M*np.conj(M),axis = 1)
        self = M_bar/np.sqrt(M2_bar)

        cross = np.zeros((nf,nf)) + 1j* np.zeros((nf,nf))

        for f0 in np.arange(nf):
            M0 = M[f0,:]
            for f1 in np.arange(nf):
                if (f1>=f0):
                    M1 = M[f1,:]
                    cross_bar = np.average(M0 * np.conj(M1))
                    denominator = np.sqrt(M2_bar[f0]*M2_bar[f1])
                    cross[f0,f1] = cross_bar / denominator
                    cross[f1,f0] = np.conj(cross[f0,f1])
                else:
                    pass
        
        return (self,cross)
            
    def Coherent_over_time(this,start, end, step, window, loader_num = 'all'):
        """The coherent signal (also called 'self_correlation' before) is defined in function Self_and_Cross_Correlation.
        Arguments:
            loader_num: int, the index of loader to use. If not given, default to be string 'all', such that all the channels will be calculated and returned
            start, end, step: double, units: second, the start and end time, and the time step to calculate each time coherent signal.
            window: double, units: second, the length of time to carry out the ensemble average

        Return:
            2D double array,if loader_num is specified,then the time series of coherent signals from the corresponding channel is returned, if not, results for all channels are returned.
        """
        if(loader_num != 'all'):
            loaders =  [this.loaders[loader_num]]
        else:
            loaders = this.loaders

        if(start < window/2):
                start = window/2 

        t_arr = np.arange(start,end,step)

        NL = len(loaders)
        NT = len(t_arr)
        coh_sig = np.zeros((NL,NT)) + 1j* np.zeros((NL,NT))

        for i in np.arange(NL):

            loader = loaders[i]

            I = loader.getI()
            Q = loader.getQ()
            sig = I+ 1j*Q
            
            for j in np.arange(len(t_arr)):
                t = t_arr[j]
            
                left_bdy = t-window/2
                right_bdy = t+window/2
            
                n_left = int((left_bdy - loader.t0)/loader.dt)
                n_right = int((right_bdy - loader.t0)/loader.dt)

                M = sig[n_left:n_right]
                
                M_bar = np.average(M)
                M2_bar = np.average(M*np.conj(M))
                
                coh_sig[i,j] = M_bar/np.sqrt(M2_bar)

        return coh_sig

        
