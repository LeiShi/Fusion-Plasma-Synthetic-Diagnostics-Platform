"""Postprocess classes and functions
including:

Raw postprocesses:
    Loading reflectometry output file module,
    Generating recieved signal module

Advanced Postprocesses:
    Correlation reflectometry analysis,
    Frequency spectrum analysis,
"""
import Code5 as c5

import numpy as np
import scipy.io.netcdf as nc
from scipy.optimize import curve_fit


class Reflectometer_Output:
    """Class that deal with the raw output from synthetic reflectometry code(FWR2D / FWR3D)

    Initialize with a output path, a frequency array , and a time sequence array
    
    """

    def __init__(this,file_path,f_arr,t_arr,n_cross_section, full_load = True):
        """initialize the output object, read the output files in file_path specified by frequencies and timesteps.
        """
        this.file_path = file_path
        this.frequencies = f_arr
        this.NF = len(f_arr)
        this.timesteps = t_arr
        this.NT = len(t_arr)
        this.n_cross_section = n_cross_section
        if (full_load):
            this.create_received_signals()
        else:
            print 'Initializer didn\'t creat E_out. Need to read E_out from somewhere else.' 
    
    def create_received_signals(this):
        """This method actually recursively read all the needed output files and produce the recieved signal for each file.
        """
        this.E_out = np.zeros((this.NF,this.NT,this.n_cross_section))
        this.E_out = this.E_out + 1j * this.E_out

        
        for f_idx in range(this.NF):
            for t_idx in range(this.NT):
                for i in range(this.n_cross_section):
                    fwr_file = this.make_file_name(this.frequencies[f_idx],this.timesteps[t_idx],i)
                    receiver_file = this.make_receiver_file_name(this.frequencies[f_idx],this.timesteps[t_idx],i)
                    this.E_out[f_idx,t_idx,i] = this.read_E_out(fwr_file,receiver_file)
            print 'frequency '+str(this.frequencies[f_idx])+' read.'
        return 0
    

    def make_file_name(this,f,t,nc):
        """create the corresponding output file name based on the given frequency and time
        """

        full_path = this.file_path + str(f)+'/'+str(t)+'/'+str(nc)+'/'
        file_name = 'out_{0:0>6.2f}_equ.cdf'.format(f)
        return full_path + file_name

    def make_receiver_file_name(this,f,t,nc):
        """create the receiver antenna pattern file name. Rightnow, it works with the NSTX_FWR_Driver.py script default.
        """

        full_path = '{0}{1}/{2}/{3}/'.format(this.file_path,str(f),str(t),str(nc))
        file_name = 'receiver_pattern.txt'
        return full_path + file_name

    def read_E_out(this,ref_file,rec_file):
        """read data from the output file, produce the received E signal, return the complex E
        """
        #print 'reading file:',file
        f = nc.netcdf_file(ref_file,'r')
        #print 'finish reading.'

        
        
        if 's_z' in f.variables.keys():
            this.dimension = 3
        else:
            this.dimension = 2

        if(this.dimension == 2):
            y = f.variables['a_y'][:]
            z = f.variables['a_z'][:]
            z_idx = f.dimensions['a_nz']/2 -1
            E_ref = f.variables['a_Er'][1,z_idx,:] + 1j*f.variables['a_Ei'][1,z_idx,:]
            f.close()

            receiver = c5.C5_reader(rec_file)

            E_rec = receiver.E_re_interp(z,y) + 1j* receiver.E_im_interp(z,y)
            
            E_out = np.trapz(E_ref*np.conj(E_rec[z_idx,:]),x=y)
            return E_out
        else:
            f.close()
            return 0

    def save_E_out(this,filename = 'E_out.sav'):
        """Save the output signal array to a binary file
        Default filename is 'E_out.sav.npy'
        Note that a '.npy' extension will be added automatically
        """
        np.save(this.file_path + filename,this.E_out)

    def load_E_out(this,filename = 'E_out.sav'):
        """load an existing E_out array from previously saved datafile.
        Default filename is E_out.sav.npy
        Note that a '.npy' extension will be automatically added if not given in filename.
        """
        if('.npy' not in filename):
            filename = filename+'.npy'
        this.E_out =  np.load(this.file_path + filename)
        
        

def Self_Correlation(ref_output):
    """Calculate the self correlation function for each frequency

    self correlation function is defined as:

    g(w) = <M(w)>/sqrt(<|M(w)|^2>)

    where <...> denotes the ensemble average, which in this case, is calculated by averaging over all time steps. And M(w) is the output signal calculated by read_E_out method in class Reflectometer_Output.

    input: Reflectometer_Output object
    """

    M = ref_output.E_out

    M_bar = np.average(np.average(M,axis = 2),axis = 1)

    M2_bar = np.average(np.average(M*np.conj(M),axis = 2),axis = 1)

    return M_bar/np.sqrt(M2_bar)

def Cross_Correlation(ref_output):
    """Calculate the cross correlation between different frequency channels

    cross correlation function is defined as:

    r(w0,w1) = <M(w0)M*(w1)>/sqrt(<|M(w0)|^2><|M(w1)|^2>)

    where M(w) is the output signal, and <...> denotes the average over timesteps

    input: Reflectometer_Output object
    """

    M = ref_output.E_out
    NF = ref_output.NF
    
    r = np.zeros((NF,NF))
    r = r + 1j*r

    M2_bar = np.average(np.average(M*np.conj(M),axis = 2),axis=1)
        
    for f0 in np.arange(NF):
        M0 = M[f0,...]        
        for f1 in np.arange(NF):
            if (f1 >= f0):
                M1 = M[f1,...]
                cross_bar = np.average(np.average(M0 * np.conj(M1),axis = 1),axis=0)
                denominator = np.sqrt(M2_bar[f0]*M2_bar[f1])
                r[f0,f1] = cross_bar / denominator
                r[f1,f0] = np.conj(r[f0,f1])
            else:
                pass

    return r

def Cross_Correlation_by_fft(ref_output):
    """Calculate teh cross correlation using fft method. Details can be found in Appendix part of ref.[1]

    [1] Observation of ion scale fluctuations in the pedestal region during the edge-localized-mode cycle on the National Spherical torus Experiment. A.Diallo, G.J.Kramer, at. el. Phys. Plasmas 20, 012505(2013)
    """

    E = ref_output.E_out
    NF = ref_output.NF
    NT = ref_output.NT
    n_cross = ref_output.n_cross_section

    E = E.reshape((NF,NT*n_cross)) #reshape the signal such that all the data from one frequency channel forms a one dimensional array

    r = np.zeros((NF,NF))
    r = r + r*1j #the complex array for results

    F = np.fft.fft(E,axis = 1)
    F2 = np.conj(F)*F
    F_norm = np.sqrt(np.average(F2,axis = 1))
    for i in range(NF):
        f = F[i,:]
        f2 = F2[i,:]
        f_norm = F_norm[i]
        for j in range(NF):
            if(j==i):
                gamma_f = f2/f_norm**2
                gamma_t = np.fft.ifft(gamma_f)
                r[i,i] = gamma_t[0]
            elif(j>i):
                g = F[j,:]
                g_norm = F_norm[j]
                gamma_f = np.conj(f)*g/(f_norm*g_norm)
                gamma_t = np.fft.ifft(gamma_f)
                r[i,j] = gamma_t[0]
            else:
                r[i,j] = np.conj(r[j,i])
    return r


def Coherent_signal_Expriment_Comparison(ref_output,exp_coherent_signals):
    """compare the simulated self_correlation signal with the coherent_signal time series from experiment. Find the least square difference time for all channels.

    Arguments:
        ref_output: Reflectometry_Output object. Make sure using the same frequency array as that in experiment analysis.
        exp_coherent_signals: complex array shaped (NF,NT), returned from ..NSTX.nstx.Analyser.Coherent_over_time function.

    Return:
        tuple(time_idx,square_error)
        time_idx:int, the index of the least error time
        squre_error: double, the sum of the square of the errors in all the channels 
    """

    self_correlation = Self_Correlation(ref_output)

    diff = exp_coherent_signals - self_correlation[:,np.newaxis]

    square_error = np.sum(np.conj(diff)*diff,axis = 0)

    time_idx = np.argmin(square_error)

    return (time_idx,square_error[time_idx])


def gaussian_fit(x,a):
    """function used for curve_fit
    """

    return np.exp(-x**2/a)

def exponential_fit(x,a):

    return np.exp(-x/a)

def fitting_cross_correlation(cross_cor_arr,dx_arr,fitting_type = 'gaussian'):
    """ fit the cross_correlation points with chosen function, default to be gaussian.

    Argument:
        cross_cor_arr:double array, absolute values of cross correlation results from chosen channels
        dx_arr:double array, the corresponding distances in major radius for each channel from the central channel
        fitting_type: string, can be 'gaussian' or 'exponential',default to be gaussian.
            gaussian fitting: y = exp(-x**2/a)
            exponential fitting: y = exp(-x/a)
    return:
        tuple, first element is the optimized fitting parameter a, second the standard deviation of a.
    """

    if(fitting_type == 'gaussian'):
        fit_func = gaussian_fit
    elif(fitting_type == 'exponential'):
        fit_func = exponential_fit
    else:
        print 'unknown fitting type:'+fitting_type
        print 'choose from gaussian or exponential'
        return (-1,-1)

    a,sigma_a2 = curve_fit(fit_func,dx_arr,cross_cor_arr)

    return (a,np.sqrt(sigma_a2))
    

    
    
    
