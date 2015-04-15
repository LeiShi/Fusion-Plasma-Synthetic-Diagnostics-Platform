# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:13:51 2015

@author: lshi

Script to calculate density sweep correlation in NSTX shot 139047 XGC1 simulation

data files are saving files from XGC_Loader class.
"""

from FPSDP.Maths.Funcs import sweeping_correlation
import numpy as np

work_dir = '/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/'

time_arr = np.arange(100,200,20)

def make_filename(t):
    return 'xgc_prof2D_all_time{0}.sav.npz'.format(t)
    
class density_data:
    """Collect all time series density data from multiple time files
    """
    
    def __init__(self,time_arr,work_dir=work_dir):
        """initialize with a time array.
        argument:
            time_arr: array of ints, should be created by range(start,end,step) or equivalent form.
                In this array, starting time of each sub-series stored in the files is given. So we can restore the whole series.
        """         
        dt = time_arr[1]-time_arr[0]
        NT = len(time_arr)*dt
        self.time = np.arange(time_arr[0],time_arr[0]+NT,1)
        fn = work_dir + make_filename(time_arr[0])
        loader_file = np.load(fn)
        
        self.dne_ad = np.copy(loader_file['dne_ad'])
        self.nane = np.copy(loader_file['nane'])
        
        for t in time_arr[1:]:
            fn = work_dir + make_filename(t)
            loader_file = np.load(fn)
            self.dne_ad = np.concatenate([self.dne_ad,loader_file['dne_ad']],axis=1)
            self.nane = np.concatenate([self.nane,loader_file['nane']],axis=1)            
        
        


        

