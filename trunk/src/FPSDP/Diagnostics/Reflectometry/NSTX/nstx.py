""" Reading and post-processing functions for NSTX correlation reflectometry output. Raw data given by Dr. Ahmed Diallo.

Program by Lei Shi, 05/16/2014

modules needed: h5py, numpy, scipy
"""

import h5py as h5
import numpy as np
from scipy.interpolate import interp1d

class NSTX_Error(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

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
        this.I = f['mydata'][0]['INPHASE']
        f.close()
        return this.I
        
    def getQ(this):
        """returns the out of phase component of the reflectometry signal 
        """
        f= h5.File(this.filename,'r')
        this.Q = f['mydata'][0]['QUADRATURE']
        f.close()
        return this.Q
    
    def getT(this):
        """returns the time array with the same shape as I and Q
        """

        this.T = this.t0 + this.dt*np.arange(this.nt)
        return this.T

    def signal(this,tstart,tend):
        """ returns the complex signal for a chosen time period, and the corresponding time array.
        Inputs:
            tstart,tend: double, the start and end of the chosen time period in seconds.
        Outputs:
            output1: the complex signal,with original resolution
            output2: the corresponding time array
        """
        
        try:
            if(tstart< this.t0 or tend > this.T[-1]):
                raise NSTX_Error('Reading raw signal error: time period outside original data.')
        except AttributeError:
            this.getT()
            if(tstart< this.t0 or tend > this.T[-1]):
                raise NSTX_Error('Reading raw signal error: time period outside original data.')
            
        nstart = int( (tstart-this.t0)/this.dt )
        nend = int( (tend-this.t0)/this.dt )

        try:
            I = this.I[nstart:nend+1]
        except AttributeError:
            this.getI()
            I = this.I[nstart:nend+1]

        try:
            Q = this.Q[nstart:nend+1]
        except AttributeError:
            this.getQ()
            Q = this.Q[nstart:nend+1]

        try:
            T = this.T[nstart:nend+1]
        except AttributeError:
            this.getT()
            T = this.T[nstart:nend+1]

        return (I + 1j * Q, T)


class FFT_result:
    """Contains returned arrays from fft analysis

    Attributes:
        origin: original time series data
        shift_fft: array after fft, and shifted so that zero frequency is located in middle
        t: time array corresponds to original data
        f: frequency array corresponds to fft data
    """

    def __init__(this, origin,fft,t,f):
        this.origin = origin
        this.fft = fft
        this.t = t
        this.f = f

class Analyser:
    """ Contains all the Post-process methods 
    """

    def __init__(this, nstx_loaders):
        """ Initialize with an NSTX_REF_loader array
        """
        this.loaders = nstx_loaders
        

    def phase(this, time_arr, tol = 1e-5, **params):
        """Calculate the extended phase curve in a given time.

        The purpose of extending the phase range to (-infi,+infi) is to avoid jumps from +pi -> -pi or the other way around on the normal [-pi,pi) range. In this case, the phase curve looks much smoother and more meaningful.
        The method we are using is first calculate the phase for each time step in the normal [-pi,pi) range, then, calculate the phase change for each time interval : dphi. For dphi>pi, we pick dphi-2*pi as the new phase change; and for dphi < -pi, we pick dphi+2*pi. In other words, we minimize the absolute value of the phase change. This treatment is valid if time step is small compared to plasma changing time scale, so the change of reflected phase shouldn't be very large.

        Arguments:
            time_arr: ndarray double, the time (real time in experimental record, unit: second) array on which we acquire the phase.
            keyword list:
            1)Loader is specified by either of the following ways:
                loader_num : loader = this.loaders[loader_num]
                frequency : check if abs(loader.freq-frequency)/frequency<tol, if find one, then use this loader, if not, raise an error.

        Return:
            Phase on the time points is returned in an ndarray. The phase is the accumulated value with respect to the initial phase at the beginning of the experimental record.
        """
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
        T = loader.getT()
        S = loader.getI()+loader.getQ()*1j #get the complex signal
        phase_raw = np.angle(S) # numpy.angle function gives the angle of a complex number in range[-pi,pi)
        dph = phase_raw[1:]-phase_raw[0:-1] #the phase change is defined on each time intervals, so the total length will be 1 shorter than the phase array.
        dph_ext = np.array([dph-2*np.pi,dph,dph+2*np.pi]) #intermediate array that contains all 3 posibilities of the phase change
        dph_arg = np.argmin(np.abs(dph_ext),axis = 0) #numpy.argmin function pick out the index of the first occurance of the minimun value in the array along one chosen axis. Since the axis 0 in our array has just 3 elements, the dph_arg will contain only 0,1,2's.
        dph_new = dph + (dph_arg-1)*2*np.pi # notice that in dph_arg, 0 corresponds dph-2*pi being the chosen one, 1 -> dph, and 2 -> dph+2*pi, therefore, this expression is valid for all 3 cases.
        phase_mod = dph_new.cumsum() # numpy.ndarray.cumsum method returns the accumulated array, since we are accumulating the whole dph_new array, the phase we got is relative to the initial phase at the start of the experiment.
        phase_interp = interp1d(T[1:-1],phase_raw[0]+phase_mod) # note that the time array now needs to be shorten by 1.
        return (phase_interp(time_arr),phase_interp,phase_mod,dph_new)

    
    def amp(this, time_arr, tol = 1e-5, **params):
        """calculates the amplitude of the fluctuating signal
        Since amplitude is much simpler than phase, we can simply calculate sqrt(I**2 + Q**2) where I,Q are in-phase and out-of-phase components.
        """
        
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
        T = loader.getT()
        S = loader.getI()+loader.getQ()*1j #get the complex signal
        amp = np.abs(S)
        amp_interp = interp1d(T,amp)
        return amp_interp(time_arr)
        

    def fft(this,tol = 1e-5, **params):
        """OUT OF DATE. WILL BE UPDATED SOON.

        FFT analysis in time.
        arguments:
            keyword list:
            1)Time steps can be given by either of the following ways:
                tstart,tend,nt: time steps = np.linspace(tstart,tend,nt)
                tstart,dt,nt: time step = tstart + np.arange(nt)*dt
            2)Loader is specified by either of the following ways:
                loader_num : loader = this.loaders[loader_num]
                frequency : check if abs(loader.freq-frequency)/frequency<tol, if find one, then use this loader, if not, raise an error.
            3)Chose the In phase component or Quadrature component
                component = 'I', 'Q', 'Amp', 'Phase' or 'Cplx' 
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
            signal = loader.getI()+loader.getQ()*1j
            raw_data = np.angle(signal)
        elif(params['component']=='Cplx'):
            raw_data = loader.getI()+loader.getQ()*1j
        else:
            raise Exception('fft initialization error: component must be either "I" or "Q"')

        raw_t = loader.getT()

        interp = interp1d(raw_t,raw_data)

        origin = interp(t)
        f = np.fft.fftfreq(len(t),t[1]-t[0])
        fft = np.fft.fft(origin)

        return FFT_result(origin,fft,t,f)


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
            M.append(loader.signal(tstart,tend))

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
            
            for j in np.arange(NT):
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

    def Cross_Correlation_by_fft(this,start,end,nt,loader_nums = 'all'):
        """Another way to calculate the cross correlation between channels. Assume f(t) and g(t) are signals from two channels, and F(w), G(w') are the corresponding Forier transform of them. Then the cross correlation (not normalized) is[1]:
            \gamma(tau) = FT(F*(w)G(w))

            a proper normalization would be (|F|*|G|)^-1 where |F| = sqrt(integral F*(w)F(w)dw)

        Arguments:
            start,end,nt: double; time inteval chosen to carry out the cross correlation. The time series will be determined as t_arr = np.linspace(start,end,nt)
            loader_num: list of int (default to be a string 'all');the loaders used in calculating the cross correlation. if given, need to be a list of int. Otherwise, by default, all the channels in the analyser will be used. 
        [1] Observation of ion scale fluctuations in the pedestal region during the edge-localized-mode cycle on the National Spherical torus Experiment. A.Diallo, G.J.Kramer, at. el. Phys. Plasmas 20, 012505(2013)
        """

        if(loader_nums == 'all'):
            loader_nums = np.arange(len(this.loaders))

        NL = len(loader_nums)

        cross_corr = np.zeros((NL,NL)) + np.zeros((NL,NL))*1j

        F = []#a list of forier transforms of each channel signal
        F2 = []# list of square of F
        F_norm = []#list of normalization term related to F
        for i in loader_nums:
            f = this.fft(tstart = start,tend = end,nt = nt,loader_num = i,component = 'Cplx').fft
            f2 = np.conj(f)*f
            f_norm = np.sqrt(np.average(f2))
            F.append(f)
            F2.append(np.conj(f)*f)
            F_norm.append(f_norm)

        for i in range(NL):
            f = F[i]
            f2 = F2[i]
            f_norm = F_norm[i]
            for j in range(NL):
                if(j == i): # if on the diagonal
                    gamma_f = f2/f_norm**2
                    gamma_t = np.fft.ifft(gamma_f)
                    cross_corr[i,i] = gamma_t[0]
                elif(j>i): #if in upper triangle region, need to calculate this term
                    g = F[j]
                    g_norm = F_norm[j]
                    gamma_f = np.conj(f)*g/(np.conj(f_norm)*g_norm)
                    gamma_t = np.fft.ifft(gamma_f)
                    cross_corr[i,j] = gamma_t[0]
                else: #if in lower triangle region, use the Hermitian property of the cross_correlation matrix
                    cross_corr[i,j] = np.conj(cross_corr[j,i])
        return cross_corr


    def Phase_Correlation(this,time_arr,loader_nums = 'all'):
        """Calculate the time translated cross correlation of the phase fluctuations between channels.

        Arguments:
            time_arr: double ndarray, contains all the time steps for calculation, (units: second)
            loader_nums: (optional) the channel numbers chosen for cross correlation. default to use all the channels in Analyser.

        Output: 3D array: shape (NL,NL,NT), NL = len(loader_nums) is the number of chosen channels, NT = len(time_arr) is the length of time series. The component (i,j,k) is the cross correlation between channel i and channel j. k <= [(NT-1)/2] and >= [-(NT-1)/2] denotes the time displacement between these two channels. Our convention is that i is delayed k*dT time compared to j. If k<0, it means that i is putting ahead of j. 
        
        """

        if(loader_nums == 'all'):
            loader_nums = np.arange(len(this.loaders))
        NL = len(loader_nums)
        NT = len(time_arr)
        corr = np.zeros((NL,NL,NT))
        phase = np.array([ this.phase(time_arr,loader_num = i)[0] for i in loader_nums ])
        phase_fluc = phase - np.mean(phase,axis = 1)[:,np.newaxis]
        for i in range(NL):
            for j in range(NL):
                for k in np.arange(NT)+np.floor(-(NT-1)/2):
                    if k<0: # i is ahead of j by k step
                        p1 = phase_fluc[i,-k:-1]
                        p2 = phase_fluc[j,0:k-1]
                        corr[i,j,k] = np.mean(p1*p2)/np.sqrt(np.mean(p1**2)*np.mean(p2**2)) #cross correlation is normalized to the averaged intensity of the two phase.
                    else: # i is delayed compared to j by k step
                        p1 = phase_fluc[i,0:-k-1]
                        p2 = phase_fluc[j,k:-1]
                        corr[i,j,k] = np.mean(p1*p2)/np.sqrt(np.mean(p1**2)*np.mean(p2**2))

        return corr



  #  def 
            
            

                
        
            
          
        
                    
        
        
