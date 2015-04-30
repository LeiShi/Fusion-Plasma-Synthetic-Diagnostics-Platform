import FPSDP.Diagnostics.Reflectometry.FWR2D.Postprocess as pp
import numpy as np

freqs=[55]
time_start,time_end,time_step = 100,220,1
nc_2d = 32
nc_3d = 16

t_arr = np.arange(time_start,time_end+1,time_step)

fwr_run_path = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/'

def load_2d(freqs = freqs, t_arr = t_arr):
    ref2d_out = pp.Reflectometer_Output(fwr_run_path,freqs,t_arr,nc_2d,full_load = False)
    ref2d_out.load_E_out('E_out_{0}.sav'.format(freqs[0]))
    return ref2d_out
    
    
fwr3d_run_path = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_FullF_multi_cross_min_out/'

def load_3d(freqs = freqs,t_arr = t_arr):
    ref3d_out = pp.Reflectometer_Output(fwr3d_run_path,freq,t_arr,nc_3d,FWR_dimension=3,full_load = False)
    ref3d_out.load_E_out()
