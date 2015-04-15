# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:01:05 2015

@author: lshi

Multiprocess version to calculate reflected signals from FWR2D/3D

useful when have multiple frequencies and time steps to analyse
"""

#using IPython multiprocessing modules, need ipcluster to be started.
import time
from IPython.parallel import Client
import FPSDP.Diagnostics.Reflectometry.FWR2D.Postprocess as pp
import numpy as np

def dv_initialize(n_engine):
    c = Client(profile='pbs')
    
    #the engine needs time to start, so check when all the engines are connected before take a direct view of the cluster.
    
    desired_engine_num = n_engine #Make sure this number is EXACTLY the same as the engine number you initiated with ipengine
    
    waiting=0
    while(len(c) < desired_engine_num and waiting<=60):#check if the engines are ready, if the engines are not ready after 1 min, something might be wrong. Exit and raise an exception.
        time.sleep(10)
        waiting += 10
    
    if(len(c) != desired_engine_num):
        raise Exception('usable engine number is not the same as the desired engine number! usable:{0}, desired:{1}.\nCheck your cluster status and the desired number set in the Driver script.'.format(len(c),desired_engine_num))
    
    
    dv = c[:]
    
    with dv.sync_imports():
        import FPSDP.Diagnostics.Reflectometry.FWR2D.Postprocess as pp
        import numpy as np
    
    dv.execute('pp=FPSDP.Diagnostics.Reflectometry.FWR2D.Postprocess')
    dv.execute('np=numpy')
    
    return dv

class Reflectometer_Output_Params:
    """container for non-essential parameters used in Reflectometer_Output class
    file_path:string
    n_cross_section:int
    FWR_dimension:int, 2 or 3
    """
    def __init__(self,file_path,n_cross_section,FWR_dimension=2,full_load = True,receiver_file_name = 'receiver_pattern.txt'):
        self.file_path = file_path
        self.n_cross_section = n_cross_section
        self.FWR_dimension = FWR_dimension
        self.full_load = full_load
        self.receiver_file_name = receiver_file_name
        
def single_freq_time(params):
    """single frequency-time run for collecting all cross-section signals, 
    this function is supposed to be scattered to all engines with different f and t parameter
    Argument:
        params is a tuple consists of 3 components:
            f: float, frequency in GHz
            t: int, time step number
            Ref_param: Reflectometer_Output_Params object, containing other preset parameters
    Returns:
        E_out: (1,1,nc) shaped complex array, the calculated reflected signal
    """
    f = params[0]
    t = params[1]
    Ref_param = params[2]    
    Ref = pp.Reflectometer_Output(Ref_param.file_path,[f],[t],Ref_param.n_cross_section,Ref_param.FWR_dimension,True,Ref_param.receiver_file_name)
    return Ref.E_out
    
def full_freq_time(freqs,time_arr,Ref_param,dv):
    """Master function to collect all frequencies and time steps reflectometer signals.
    Arguments:
        freqs: array of floats, all the frequencies in GHz
        time_arr: array of ints, all the time steps
        Ref_param: Reflectometer_Output_Params object, containing other preset parameters
        dv: direct-view of an IPython parallel cluster, obtained by function dv_initialize()
    Returns:
        Reflectometer_Output object with parameters given by freqs, time_arr, and Ref_param. It's E_out attribute contains the corresponding complex signals
    """
    Ref_all = pp.Reflectometer_Output(Ref_param.file_path,freqs,time_arr,Ref_param.n_cross_section,Ref_param.FWR_dimension,False,Ref_param.receiver_file_name)
    Ref_all.E_out = np.zeros((Ref_all.NF,Ref_all.NT,Ref_all.n_cross_section),dtype='complex')
    
    parallel_param_list = [(f,t,Ref_param) for f in freqs for t in time_arr]
    parallel_result = dv.map_async(single_freq_time,parallel_param_list)
    print('Parallel runs started.')
    parallel_result.wait_interactive()
    print('All signals computed!')
    E_out_scattered = parallel_result.get()
    for i in range(Ref_all.NF):
        for j in range(Ref_all.NT):
            Ref_all.E_out[i,j] = E_out_scattered[i*len(time_arr)+j]
            
    return Ref_all
    
    
    

