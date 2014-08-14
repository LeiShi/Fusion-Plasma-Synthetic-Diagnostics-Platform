import FPSDP.scripts.load_nstx_exp_ref as nstx_exp
import FPSDP.scripts.FWR2D_NSTX_139047_Postprocess as fwrpp
import matplotlib.pyplot as plt

import pickle
import numpy as np

with open('/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/ref_pos.pck','r') as f:
    ref_pos = pickle.load(f)

n_channel = 16

#create the distance matrix, dx[i,j] is the absolute distance between the reflection points of i-th and j-th channel 
dx = np.absolute(np.zeros((n_channel,n_channel))+ref_pos[np.newaxis,:]-ref_pos[:,np.newaxis])

#calculate cross-correlation matrix from synthetic signals
cc_fwr = fwrpp.pp.Cross_Correlation_by_fft(fwrpp.ref2d_out)
cc_fwr2 = fwrpp.pp.Cross_Correlation_by_fft(fwrpp.ref2d_amp2_out)
cc_fwr01 = fwrpp.pp.Cross_Correlation_by_fft(fwrpp.ref2d_amp01_out)
print 'FWR data loaded'

#calculate cross-correlation matrix from experimental signals, note that for our case, the simulated time slice is at t=0.632s, so we choose corresponding experimental data from 0.632-0.640, the total sample number is chosen to be 2000 because larger sample doesn't bring in any difference, since the increased samples are not statistical independent.  
cc_exp = nstx_exp.analyser.Cross_Correlation_by_fft(0.632,0.640,2000)

print 'nstx data loaded'

#choose the channel ranges representing top/bottom part of pedestal, and center channels for each region. 
top_center = 11
top_range = [8,12]

bottom_center = 7
bottom_range = [0,8]

#pick chosen data from whole correlation matrices

fwr_top=[]
fwr2_top = []
fwr01_top=[]
exp_top = []
dx_top=[]
def pick_top():
    global fwr_top,fwr2_top,exp_top,dx_top,fwr01_top
    fwr_top = np.absolute(cc_fwr[top_center,top_range[0]:top_range[1]])
    fwr2_top = np.absolute(cc_fwr2[top_center,top_range[0]:top_range[1]])
    fwr01_top = np.absolute(cc_fwr01[top_center,top_range[0]:top_range[1]])
    exp_top = np.absolute(cc_exp[top_center,top_range[0]:top_range[1]])
    dx_top = dx[top_center,top_range[0]:top_range[1]]

pick_top()

fwr_bot=[]
fwr2_bot=[]
fwr01_bot = []
exp_bot=[]
dx_bot=[]
def pick_bottom():
    global fwr_bot,fwr2_bot,fwr01_bot,exp_bot,dx_bot
    fwr_bot = np.absolute(cc_fwr[bottom_center,bottom_range[0]:bottom_range[1]])
    fwr2_bot = np.absolute(cc_fwr2[bottom_center,bottom_range[0]:bottom_range[1]])
    fwr01_bot = np.absolute(cc_fwr01[bottom_center,bottom_range[0]:bottom_range[1]])
    exp_bot = np.absolute(cc_exp[bottom_center,bottom_range[0]:bottom_range[1]])
    dx_bot = dx[bottom_center,bottom_range[0]:bottom_range[1]]

pick_bottom()

#fitting with gaussian(for bottom) and exponential(for top)
xmax_t = 0
xfit_t = 0
fwr_fit_t = 0
fwr2_fit_t = 0
fwr01_fit_t = 0
exp_fit_t = 0
fwr_t_a,fwr_t_sa = 0,0
fwr2_t_a,fwr2_t_sa = 0,0
fwr01_t_a,fwr01_t_sa = 0,0
exp_t_a,exp_t_sa = 0,0


def fit_top():
    global fwr_t_a,fwr_t_sa,fwr2_t_a,fwr2_t_sa,fwr01_t_a,fwr01_t_sa,exp_t_a,expt_sa,xmax_t,xfit_t,fwr_fit_t,fwr2_fit_t,exp_fit_t,fwr01_fit_t
    fwr_t_a,fwr_t_sa = fwrpp.pp.fitting_cross_correlation(fwr_top,dx_top,'exponential')
    fwr2_t_a,fwr2_t_sa = fwrpp.pp.fitting_cross_correlation(fwr2_top,dx_top,'exponential')
    fwr01_t_a,fwr01_t_sa = fwrpp.pp.fitting_cross_correlation(fwr01_top,dx_top,'exponential')
    exp_t_a,exp_t_sa = fwrpp.pp.fitting_cross_correlation(exp_top,dx_top,'exponential')
    xmax_t = 3*np.max((np.abs(fwr_t_a),np.abs(fwr2_t_a),np.abs(exp_t_a)))
    xfit_t = np.linspace(0,xmax_t,500)
    fwr_fit_t = fwrpp.pp.exponential_fit(xfit_t,fwr_t_a)
    fwr2_fit_t = fwrpp.pp.exponential_fit(xfit_t,fwr2_t_a)
    fwr01_fit_t = fwrpp.pp.exponential_fit(xfit_t,fwr01_t_a)
    exp_fit_t = fwrpp.pp.exponential_fit(xfit_t,exp_t_a)

fit_top()

xmax_b = 0
xfit_b = 0
fwr_fit_b = 0
fwr2_fit_b = 0
fwr01_fit_b = 0
exp_fit_b = 0
fwr_b_a,fwr_b_sa = 0,0
fwr2_b_a,fwr2_b_sa = 0,0
fwr01_b_a,fwr01_b_sa = 0,0
exp_b_a,exp_b_sa = 0,0

def fit_bot():
    global fwr_b_a,fwr_b_sa,fwr2_b_a,fwr2_b_sa,fwr01_b_a,fwr01_b_sa,exp_b_a,expt_sa,xmax_b,xfit_b,fwr_fit_b,fwr2_fit_b,exp_fit_b,fwr01_fit_b
    fwr_b_a,fwr_b_sa = fwrpp.pp.fitting_cross_correlation(fwr_bot,dx_bot,'gaussian')
    fwr2_b_a,fwr2_b_sa = fwrpp.pp.fitting_cross_correlation(fwr2_bot,dx_bot,'gaussian')
    fwr01_b_a,fwr01_b_sa = fwrpp.pp.fitting_cross_correlation(fwr01_bot,dx_bot,'gaussian')
    exp_b_a,exp_b_sa = fwrpp.pp.fitting_cross_correlation(exp_bot,dx_bot,'gaussian')
    xmax_b = 3*np.sqrt(np.max((np.abs(fwr_b_a),np.abs(fwr2_b_a),np.abs(exp_b_a))))
    xfit_b = np.linspace(0,xmax_b,500)
    fwr_fit_b = fwrpp.pp.gaussian_fit(xfit_b,fwr_b_a)
    fwr2_fit_b = fwrpp.pp.gaussian_fit(xfit_b,fwr2_b_a)
    fwr01_fit_b = fwrpp.pp.gaussian_fit(xfit_b,fwr01_b_a)
    exp_fit_b = fwrpp.pp.gaussian_fit(xfit_b,exp_b_a)

fit_bot()

print 'fitting complete'
print 'fitting curve ready. call plot() to plot. note that the default region is top, pass "bottom" as the argument to plot bottom region. '
#plot the data points and curves

total_plot = 0

#top data
def plot(region = 'top'):
    global total_plot
    plt.figure()
    total_plot += 1
    if(region == 'top'):
        plt.title('Cross-Correlation at Pedestal Top,center_chennal='+str(top_center))
        plt.plot(dx_top,exp_top,'bs',label = 'exp data')
        plt.plot(dx_top,fwr_top,'ro',label = 'FWR data amp=1')
        plt.plot(dx_top,fwr2_top,'r^',label = 'FWR data amp=2')
        plt.plot(dx_top,fwr01_top,'r+',label = 'FWR data amp=0.1')
        plt.plot(xfit_t,exp_fit_t,'b-',label = 'exp exponential fit')
        plt.plot(xfit_t,fwr_fit_t,'r--',label = 'FWR fit')
        plt.plot(xfit_t,fwr2_fit_t,'r-.',label = 'FWR amp2 fit')
        plt.plot(xfit_t,fwr01_fit_t,'r:',label = 'FWR amp0.1 fit')
        plt.xlabel('distance from center channel reflection($m$)')
        plt.ylabel('cross-correlation')
        plt.legend()
    elif(region == 'bottom'):
        plt.title('Cross-Correlation at Pedestal bottom,center_channel='+str(bottom_center))
        plt.plot(dx_bot,exp_bot,'bs',label = 'exp data')
        plt.plot(dx_bot,fwr_bot,'ro',label = 'FWR data amp=1')
        plt.plot(dx_bot,fwr2_bot,'r^',label = 'FWR data amp=2')
        plt.plot(dx_bot,fwr01_bot,'r+',label = 'FWR data amp=0.1')
        plt.plot(xfit_b,exp_fit_b,'b-',label = 'exp gaussian fit')
        plt.plot(xfit_b,fwr_fit_b,'r--',label = 'FWR fit')
        plt.plot(xfit_b,fwr2_fit_b,'r-.',label = 'FWR amp2 fit')
        plt.plot(xfit_b,fwr01_fit_b,'r:',label = 'FWR amp0.1 fit')
        plt.xlabel('distance from center channel reflection($m$)')
        plt.ylabel('cross-correlation')
        plt.legend()


def clear_all():
    global total_plot

    for i in range(total_plot):
        plt.close()



