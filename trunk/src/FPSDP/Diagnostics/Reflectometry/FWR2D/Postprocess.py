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


class Reflectometer_Output:
    """Class that deal with the raw output from synthetic reflectometry code(FWR2D / FWR3D)

    Initialize with a output path, a frequency array , and a time sequence array
    
    """

    def __init__(this,file_path,f_arr,t_arr):
        """initialize the output object, read the output files in file_path specified by frequencies and timesteps.
        """
        this.file_path = file_path
        this.frequencies = f_arr
        this.NF = len(f_arr)
        this.timesteps = t_arr
        this.NT = len(t_arr)

        this.create_received_signals()

    def create_received_signals(this):
        """This method actually recursively read all the needed output files and produce the recieved signal for each file.
        """
        this.E_out = np.zeros((this.NF,this.NT))
        this.E_out = this.E_out + 1j * this.E_out

        
        for f_idx in range(this.NF):
            for t_idx in range(this.NT):
                fwr_file = this.make_file_name(this.frequencies[f_idx],this.timesteps[t_idx])
                receiver_file = this.make_receiver_file_name(this.frequencies[f_idx],this.timesteps[t_idx])
                this.E_out[f_idx,t_idx] = this.read_E_out(fwr_file,receiver_file)
            print 'frequency '+str(this.frequencies[f_idx])+' read.'
        return 0
    

    def make_file_name(this,f,t):
        """create the corresponding output file name based on the given frequency and time
        """

        full_path = this.file_path + str(f)+'/'+str(t)+'/'
        file_name = 'out_{0:0>6.2f}_equ.cdf'.format(f)
        return full_path + file_name

    def make_receiver_file_name(this,f,t):
        """create the receiver antenna pattern file name. Rightnow, it works with the NSTX_FWR_Driver.py script default.
        """

        full_path = '{0}{1}/{2}/'.format(this.file_path,str(f),str(t))
        file_name = 'receiver_pattern.txt'
        return full_path + file_name

    def read_E_out(this,ref_file,rec_file):
        """read data from the output file, produce the received E signal, stored as [E_real, E_imaginary]
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
        np.save(filename,this.E_out)

def load_E_out(filename = 'E_out.sav'):
    """load an existing E_out array from previously saved datafile.
    Default filename is E_out.sav.npy
    Note that a '.npy' extension will be automatically added if not given in filename.
    """
    if('.npy' not in filename):
        filename = filename+'.npy'
    return np.load(filename)
        
        

def Self_Correlation(ref_output):
    """Calculate the self correlation function for each frequency

    self correlation function is defined as:

    g(w) = <M(w)>/sqrt(<|M(w)|^2>)

    where <...> denotes the ensemble average, which in this case, is calculated by averaging over all time steps. And M(w) is the output signal calculated by read_E_out method in class Reflectometer_Output.

    input: Reflectometer_Output object
    """

    M = ref_output.E_out

    M_bar = np.sum(M,axis = 1)/ref_output.NT

    M2_bar = np.sum(M*np.conj(M),axis = 1)/ref_output.NT

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
    NT = ref_output.NT
    
    r = np.zeros((NF,NF))
    r = r + 1j*r

    M2_bar = np.sum(M*np.conj(M),axis = 1)/NT
        
    for f0 in np.arange(NF):
        M0 = M[f0,:]        
        for f1 in np.arange(NF):
            if (f1 >= f0):
                M1 = M[f1,:]
                cross_bar = np.sum(M0 * np.conj(M1))/NT
                denominator = np.sqrt(M2_bar[f0]*M2_bar[f1])
                r[f0,f1] = cross_bar / denominator
                r[f1,f0] = np.conj(r[f0,f1])
            else:
                pass

    return r
