import FPSDP.scripts.load_nstx_exp_ref as nstx_exp
#import FPSDP.scripts.FWR2D_NSTX_139047_Postprocess as fwrpp
import pickle
import numpy as np

with open('/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/ref_pos.pck','r') as f:
    ref_pos = pickle.load(f)

channel = 9
nt = 50
llim = 1e-7
ulim = 1e-4
time_array = np.linspace(llim,ulim,nt)
cs_mean = np.zeros((nt))
cs_median = np.zeros((nt))
cs_std = np.zeros((nt))

def cs_scan(cha = channel):
    global cs_mean,cs_std,time_array,cs_median
    time_array = np.linspace(llim,ulim,nt)
    cs_median = np.zeros((nt))
    cs_mean = np.zeros((nt))
    cs_std = np.zeros((nt))
    for t in range(nt):
        cs_exp = nstx_exp.analyser.Coherent_over_time(0.632,0.640,1e-6,time_array[t],loader_num = cha)
        cs_mean[t] = np.mean(np.abs(cs_exp))
        cs_median[t] = np.median(np.abs(cs_exp))
        cs_std[t] = np.std(np.abs(cs_exp))
    return cs_mean,cs_median,cs_std

def get_coh_time(cha = channel):
    mean,median,std = cs_scan(cha)

    t_idx = np.argmax(std)
    print 'optimal window for channel {1}= {0:.4}'.format(time_array[t_idx],cha)
    return time_array[t_idx]

def get_coh_median_std(cha = channel, window = None):
    if(window == None):  
        window = get_coh_time(cha)

    cs_exp = nstx_exp.analyser.Coherent_over_time(0.632,0.640,1e-6,window,loader_num = cha)

    cs_ab = np.abs(cs_exp)

    #median = np.median(cs_ab)

    #print 'divider set to be {0:.4}'.format(median)

    # cs_tophalf = cs_ab[np.nonzero(cs_ab>median)]

    return np.median(cs_ab),np.std(cs_ab),cs_ab


    

def all_channel_coh_sigs(window = None):
    m = np.zeros((16))
    std = np.zeros((16))
    cs_sig = []

    for i in range(16):
        m[i],std[i],cs_tmp = get_coh_median_std(i,window = window)
        cs_sig.append(cs_tmp)

    return m,std,cs_sig

