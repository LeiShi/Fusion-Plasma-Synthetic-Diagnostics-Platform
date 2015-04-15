import FPSDP.Diagnostics.Reflectometry.FWR2D.Postprocess as pp

from NSTX_139047_loader import freqs,time_start,time_end,time_step

import numpy as np

t_arr = np.arange(time_start,time_end+1,time_step)

fwr_run_path = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_FullF_multi_cross/'

ref2d_out = pp.Reflectometer_Output(fwr_run_path,freqs,t_arr,16,full_load = False)

ref2d_out.load_E_out()

fwramp2_run_path = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_FullF_multi_cross_Amp2/'

ref2d_amp2_out = pp.Reflectometer_Output(fwramp2_run_path,freqs,t_arr,16,full_load = False)
ref2d_amp2_out.load_E_out()

fwramp01_run_path = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_FullF_multi_cross_Amp01/'

ref2d_amp01_out = pp.Reflectometer_Output(fwramp01_run_path,freqs,t_arr,16,full_load = False)
ref2d_amp01_out.load_E_out()

fwr3d_run_path = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_FullF_multi_cross_min_out/'

ref3d_out = pp.Reflectometer_Output(fwr3d_run_path,freqs,t_arr,16,full_load = False)
ref3d_out.load_E_out()
