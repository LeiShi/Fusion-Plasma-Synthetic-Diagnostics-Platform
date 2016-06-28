# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:01:05 2015

@author: lshi

Multiprocess version to calculate reflected signals from FWR2D/3D

useful when have multiple frequencies and time steps to analyse
"""

# using IPython multiprocessing modules, need ipcluster to be started.
import time

from IPython.parallel import Client
import numpy as np

from . import postprocess as pp

def dv_initialize(n_engine,profile = 'default'):
    c = Client(profile=profile)
    
    # The engine needs time to start, so check when all the engines are 
    # connected before take a direct view of the cluster.
    
    # Make sure this desired_engine_num is EXACTLY the same as the engine 
    # number you initiated with ipengine
    desired_engine_num = n_engine 
    
    # check if the engines are ready, if the engines are not ready after 1 min,
    # something might be wrong. Exit and raise an exception.
    waiting=0
    while(len(c) < desired_engine_num and waiting<=60):
        time.sleep(10)
        waiting += 10
    
    if(len(c) != desired_engine_num):
        raise Exception('usable engine number is not the same as the desired \
engine number! usable:{0}, desired:{1}.\nCheck your cluster status and the \
desired number set in the Driver script.'.format(len(c),desired_engine_num))
    
    
    dv = c[:]
    
    with dv.sync_imports():
        import sdp.diagnostic.fwr.fwr2d.postprocess
        import numpy 
    
    dv.execute('pp=sdp.diagnostic.fwr.fwr2d.Postprocess')
    dv.execute('np=numpy')
    
    return dv

class Reflectometer_Output_Params:
    """container for non-essential parameters used in Reflectometer_Output 
    class
    
    :param string file_path: Path to FWR2D/3D output files
    :param int n_cross_section: total number of cross-section planes 
    :param int FWR_dimension: either 2 or 3, default to be 2
    :param bool full_load: Optional, default is True. If False, data won't be
                           loaded during initialization. Pre-saved will be 
                           needed for further use.
    :param string receiver_file_name: Optional, filename to the Code5 file 
                                      specifying the receiver electric field.
                                      Default is the default file name set by
                                      FWR_Driver script.
    """
    def __init__(self,file_path,n_cross_section,FWR_dimension=2,
                 full_load=True, receiver_file_name='receiver_pattern.txt'):
        self.file_path = file_path
        self.n_cross_section = n_cross_section
        self.FWR_dimension = FWR_dimension
        self.full_load = full_load
        self.receiver_file_name = receiver_file_name
        
def single_freq_time(params):
    """single frequency-time run for collecting all cross-section signals, 
    this function is supposed to be scattered to all engines with different f 
    and t parameter

    params: 
        A tuple containing the following parameters:
        
        f: 
            float, frequency in GHz
                   
        t: 
            int, time step number
        
        Ref_param: 
            Reflectometer_Output_Params object, containing other preset 
            parameters
    
    Returns:
        E_out: (1,1,nc) shaped complex array, the calculated reflected signal
    """
    f = params[0]
    t = params[1]
    Ref_param = params[2]    
    Ref = pp.Reflectometer_Output(Ref_param.file_path, [f], [t],
                                  Ref_param.n_cross_section,
                                  Ref_param.FWR_dimension, True, 
                                  Ref_param.receiver_file_name)
    return Ref.E_out
    
def full_freq_time(freqs, time_arr, Ref_param, dv):
    """Master function to collect all frequencies and time steps reflectometer 
    signals.
        
    :param freqs:  all the frequencies in GHz
    :type freqs: array of floats
    :param time_arr: all the time steps
    :type time_arr: array of ints
    :param Ref_param: containing other preset parameters
    :type Ref_param: :py:class`Reflectometer_Output_Params` object
    :param dv: direct-view of an IPython parallel cluster, obtained by function
               dv_initialize()
    
    :returns: Reflectometer_Output object with parameters given by freqs, 
              time_arr, and Ref_param. Its E_out attribute contains the 
              corresponding complex signals.
    """
    Ref_all = pp.Reflectometer_Output(Ref_param.file_path, freqs, time_arr, 
                                      Ref_param.n_cross_section, 
                                      Ref_param.FWR_dimension, False, 
                                      Ref_param.receiver_file_name)
    Ref_all.E_out = np.zeros((Ref_all.NF, Ref_all.NT, Ref_all.n_cross_section),
                             dtype='complex')
    
    parallel_param_list = [(f,t,Ref_param) for f in freqs for t in time_arr]
    parallel_result = dv.map_async(single_freq_time, parallel_param_list)
    print('Parallel runs started.')
    parallel_result.wait_interactive()
    print('All signals computed!')
    E_out_scattered = parallel_result.get()
    print('signals collected.')
    for i in range(Ref_all.NF):
        for j in range(Ref_all.NT):
            Ref_all.E_out[i, j, :] = E_out_scattered[i*len(time_arr)+j][0, 0,:]
            print(('freq {0},time {1} stored.'.format(freqs[i], time_arr[j])))
            
    return Ref_all
    
