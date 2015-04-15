import FPSDP.scripts.load_nstx_exp_ref as nstx_exp
import FPSDP.scripts.FWR2D_NSTX_139047_Postprocess as fwrpp
import FPSDP.Plasma.analysis as ana
import matplotlib.pyplot as plt

import pickle
import numpy as np

with open('/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/ref_pos.pck','r') as f:
    ref_pos = pickle.load(f)

dne_ana = ana.XGC_Density_Loader('/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/dne_file.sav.npz')


n_channel = 16

#create the distance matrix, dx[i,j] is the absolute distance between the reflection points of i-th and j-th channel 
dx = np.absolute(np.zeros((n_channel,n_channel))+ref_pos[np.newaxis,:]-ref_pos[:,np.newaxis])

#calculate cross-correlation matrix from synthetic signals
cc_fwr = fwrpp.pp.Cross_Correlation_by_fft(fwrpp.ref2d_out)
cc_fwr2 = fwrpp.pp.Cross_Correlation_by_fft(fwrpp.ref2d_amp2_out)
cc_fwr01 = fwrpp.pp.Cross_Correlation_by_fft(fwrpp.ref2d_amp01_out)
cc_3d = fwrpp.pp.Cross_Correlation_by_fft(fwrpp.ref3d_out)

cs_fwr = fwrpp.pp.Self_Correlation(fwrpp.ref2d_out)
cs_fwr2 = fwrpp.pp.Self_Correlation(fwrpp.ref2d_amp2_out)
cs_fwr01 = fwrpp.pp.Self_Correlation(fwrpp.ref2d_amp01_out)
cs_3d = fwrpp.pp.Self_Correlation(fwrpp.ref3d_out)

print 'FWR data loaded'

#calculate cross-correlation matrix from experimental signals, note that for our case, the simulated time slice is at t=0.632s, so we choose corresponding experimental data from 0.632-0.640, the total sample number is chosen to be 2000 because larger sample doesn't bring in any difference, since the increased samples are not statistical independent.  

cc_exp = nstx_exp.analyser.Cross_Correlation_by_fft(0.632,0.640,8000)
#cc_exp_short = nstx_exp.analyser.Cross_Correlation_by_fft(0.634,0.6348,8000)

#calculate coherent signal for all channels from NSTX. The result is an 2D array containing time series of coherent signal from all the channels.

cs_exp = nstx_exp.analyser.Coherent_over_time(0.632,0.640,2e-5,1e-4)
print 'nstx data loaded'

#choose the channel ranges representing top/bottom part of pedestal, and center channels for each region. 
top_center = 11
top_range = [8,12]

bottom_center = 6
bottom_range = [2,7]

#pick chosen data from whole correlation matrices

fwr_top=[]
fwr2_top = []
fwr01_top=[]
fwr3d_top=[]
exp_top = []
dx_top=[]
def pick_top():
    global fwr_top,fwr2_top,exp_top,dx_top,fwr01_top,fwr3d_top
    fwr_top = np.absolute(cc_fwr[top_center,top_range[0]:top_range[1]])
    fwr2_top = np.absolute(cc_fwr2[top_center,top_range[0]:top_range[1]])
    fwr01_top = np.absolute(cc_fwr01[top_center,top_range[0]:top_range[1]])
    fwr3d_top = np.absolute(cc_3d[top_center,top_range[0]:top_range[1]])
    exp_top = np.absolute(cc_exp[top_center,top_range[0]:top_range[1]])
    dx_top = dx[top_center,top_range[0]:top_range[1]]

pick_top()

fwr_bot=[]
fwr2_bot=[]
fwr01_bot = []
fwr3d_bot = []
exp_bot=[]
dx_bot=[]
def pick_bottom():
    global fwr_bot,fwr2_bot,fwr01_bot,exp_bot,dx_bot,fwr3d_bot
    fwr_bot = np.absolute(cc_fwr[bottom_center,bottom_range[0]:bottom_range[1]])
    fwr2_bot = np.absolute(cc_fwr2[bottom_center,bottom_range[0]:bottom_range[1]])
    fwr01_bot = np.absolute(cc_fwr01[bottom_center,bottom_range[0]:bottom_range[1]])
    fwr3d_bot = np.absolute(cc_3d[bottom_center,bottom_range[0]:bottom_range[1]])
    exp_bot = np.absolute(cc_exp[bottom_center,bottom_range[0]:bottom_range[1]])
    dx_bot = dx[bottom_center,bottom_range[0]:bottom_range[1]]

pick_bottom()

#fitting with gaussian(for bottom) and exponential(for top)
xmax_t = 0
xfit_t = 0
fwr_fit_t = 0
fwr2_fit_t = 0
fwr01_fit_t = 0
fwr3d_fit_t = 0
exp_fit_t = 0
fwr_t_a,fwr_t_sa = 0,0
fwr2_t_a,fwr2_t_sa = 0,0
fwr01_t_a,fwr01_t_sa = 0,0
fwr3d_t_a,fwr3d_t_sa = 0,0
exp_t_a,exp_t_sa = 0,0

xgc_fit_t = 0
xgc_t_a,xgc_t_sa = 0,0
x_t,dne_c_t = 0,0

def fit_top():
    global fwr_t_a,fwr_t_sa,fwr2_t_a,fwr2_t_sa,fwr01_t_a,fwr01_t_sa,fwr3d_t_a,fwr3d_t_sa,exp_t_a,expt_sa,xmax_t,xfit_t,fwr_fit_t,fwr2_fit_t,exp_fit_t,fwr01_fit_t,fwr3d_fit_t,xgc_fit_t,xgc_t_a,xgc_t_sa,x_t,dne_c_t
    fwr_t_a,fwr_t_sa = fwrpp.pp.fitting_cross_correlation(fwr_top,dx_top,'exponential')
    fwr2_t_a,fwr2_t_sa = fwrpp.pp.fitting_cross_correlation(fwr2_top,dx_top,'exponential')
    fwr01_t_a,fwr01_t_sa = fwrpp.pp.fitting_cross_correlation(fwr01_top,dx_top,'exponential')
    fwr3d_t_a,fwr3d_t_sa = fwrpp.pp.fitting_cross_correlation(fwr3d_top,dx_top,'exponential')
    exp_t_a,exp_t_sa = fwrpp.pp.fitting_cross_correlation(exp_top,dx_top,'exponential')
    opt_t,x_t,dne_c_t = dne_ana.density_correlation(ref_pos[top_center],width = ref_pos[top_range[0]]-ref_pos[top_center])
    xgc_t_a,xgc_t_sa = opt_t
    
    xmax_t = 2*np.max((np.abs(fwr_t_a),np.abs(fwr2_t_a),np.abs(exp_t_a)))
    xfit_t = np.linspace(0,xmax_t,500)
    fwr_fit_t = fwrpp.pp.exponential_fit(xfit_t,fwr_t_a)
    fwr2_fit_t = fwrpp.pp.exponential_fit(xfit_t,fwr2_t_a)
    fwr01_fit_t = fwrpp.pp.exponential_fit(xfit_t,fwr01_t_a)
    fwr3d_fit_t = fwrpp.pp.exponential_fit(xfit_t,fwr3d_t_a)
    exp_fit_t = fwrpp.pp.exponential_fit(xfit_t,exp_t_a)
    xgc_fit_t = ana.gaussian_correlation_func(xfit_t,xgc_t_a)

fit_top()

xmax_b = 0
xfit_b = 0
fwr_fit_b = 0
fwr2_fit_b = 0
fwr01_fit_b = 0
fwr3d_fit_b = 0
exp_fit_b = 0
fwr_b_a,fwr_b_sa = 0,0
fwr2_b_a,fwr2_b_sa = 0,0
fwr01_b_a,fwr01_b_sa = 0,0
fwr3d_b_a,fwr3d_b_sa = 0,0
exp_b_a,exp_b_sa = 0,0

xgc_fit_b = 0
xgc_b_a,xgc_b_sa = 0,0
x_b,dne_c_b = 0,0

def fit_bot():
    global fwr_b_a,fwr_b_sa,fwr2_b_a,fwr2_b_sa,fwr01_b_a,fwr01_b_sa,fwr3d_b_a,fwr3d_b_sa,exp_b_a,expt_sa,xmax_b,xfit_b,fwr_fit_b,fwr2_fit_b,exp_fit_b,fwr01_fit_b,fwr3d_fit_b,xgc_fit_b,xgc_b_a,xgc_b_sa,x_b,dne_c_b
    fwr_b_a,fwr_b_sa = fwrpp.pp.fitting_cross_correlation(fwr_bot,dx_bot,'gaussian')
    fwr2_b_a,fwr2_b_sa = fwrpp.pp.fitting_cross_correlation(fwr2_bot,dx_bot,'gaussian')
    fwr01_b_a,fwr01_b_sa = fwrpp.pp.fitting_cross_correlation(fwr01_bot,dx_bot,'gaussian')
    fwr3d_b_a,fwr3d_b_sa = fwrpp.pp.fitting_cross_correlation(fwr3d_bot,dx_bot,'gaussian')
    exp_b_a,exp_b_sa = fwrpp.pp.fitting_cross_correlation(exp_bot,dx_bot,'gaussian')
    
    opt_b,x_b,dne_c_b = dne_ana.density_correlation(ref_pos[bottom_center],width = ref_pos[bottom_range[0]]-ref_pos[bottom_center])
    xgc_b_a,xgc_b_sa = opt_b
    
    xmax_b = 2*np.sqrt(np.max((np.abs(fwr_b_a),np.abs(fwr2_b_a),np.abs(exp_b_a))))
    xfit_b = np.linspace(0,xmax_b,500)
    fwr_fit_b = fwrpp.pp.gaussian_fit(xfit_b,fwr_b_a)
    fwr2_fit_b = fwrpp.pp.gaussian_fit(xfit_b,fwr2_b_a)
    fwr01_fit_b = fwrpp.pp.gaussian_fit(xfit_b,fwr01_b_a)
    fwr3d_fit_b = fwrpp.pp.gaussian_fit(xfit_b,fwr3d_b_a)
    exp_fit_b = fwrpp.pp.gaussian_fit(xfit_b,exp_b_a)
    xgc_fit_b = ana.gaussian_correlation_func(xfit_b,xgc_b_a)
fit_bot()

print 'fitting complete'
print 'fitting curve ready. call plot() to plot. note that the default region is top, pass "bottom" as the argument to plot bottom region. '
#plot the data points and curves

total_plot = 0

#top data
def plot(region = 'top'):
    global total_plot
    #plt.figure()
    #total_plot += 1
    if(region == 'top'):
        plt.title('Cross-Correlation at Upper Pedestal,center_channel at {0:.4}m'.format(ref_pos[top_center]))
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
        plt.legend(labelspacing = 0.2,prop = {'size':12})
        plt.tight_layout()
    elif(region == 'bottom'):
        plt.title('Cross-Correlation at Lower Pedestal,center_channel at {0:.4}m'.format(ref_pos[bottom_center]))
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
        plt.legend(labelspacing = 0.2,prop = {'size':12})
        plt.tight_layout()
    elif(region == '2d/3d_top'):
        plt.title('Cross-Correlation at Upper Pedestal,center_channel at {0:.4}m'.format(ref_pos[top_center]))
        plt.plot(dx_top,exp_top,'bs',label = 'exp data')
        plt.plot(dx_top,fwr_top,'ro',label = 'FWR2D data')
        plt.plot(dx_top,fwr3d_top,'r^',label = 'FWR3D data')
        plt.plot(xfit_t,exp_fit_t,'b-',label = 'exp exponential fit')
        plt.plot(xfit_t,fwr_fit_t,'r--',label = 'FWR2D fit')
        plt.plot(xfit_t,fwr3d_fit_t,'r-.',label = 'FWR3D fit')
        plt.xlabel('distance from center channel reflection($m$)')
        plt.ylabel('cross-correlation')
        plt.legend(labelspacing = 0.2,prop = {'size':12})
        plt.tight_layout()
    elif(region =='2d/3d_bot'):
        #plt.title('Cross-Correlation at Lower Pedestal,center_channel at {0:.4}m'.format(ref_pos[bottom_center]))
        plt.plot(dx_bot,exp_bot,'bs',label = 'exp data')
        plt.plot(dx_bot,fwr_bot,'go',label = 'FWR2D data')
        plt.plot(dx_bot,fwr3d_bot,'r^',label = 'FWR3D data')
        plt.plot(xfit_b,exp_fit_b,'b-')
        plt.plot(xfit_b,fwr_fit_b,'g--')
        plt.plot(xfit_b,fwr3d_fit_b,'r-.')
        plt.xlabel('$distance from center channel(mm)$')
        plt.ylabel('$\gamma$')
        plt.legend(labelspacing = 0.2,prop = {'size':15})
        plt.tight_layout()
    elif(region == '3d_bot'):
        plt.title('2D/3D Cross-Correlation and XGC1 Density Correlation, Lower')
        plt.plot(dx_bot,fwr_bot,'ro',label = '2D')
        plt.plot(dx_bot,fwr3d_bot,'r^',label = '3D')
        plt.plot(x_b,dne_c_b,'bs',label = 'XGC')
        plt.plot(xfit_b,fwr_fit_b,'r-.',label = '2D fit')
        plt.plot(xfit_b,fwr3d_fit_b,'r--',label = '3D fit')
        plt.plot(xfit_b,xgc_fit_b,'b-',label = 'XGC fit')
        plt.xlabel('distance from center channel relfection($m$)')
        plt.ylabel('cross-corelation')
        plt.legend(labelspacing = 0.2,prop = {'size':12})
        plt.tight_layout()
    elif(region == '3d_top'):
        plt.title('2D/3D Cross-Correlation and XGC1 Density Correlation, Upper')
        plt.plot(dx_top,fwr_top,'ro',label = '2D')
        plt.plot(dx_top,fwr3d_top,'r^',label = '3D')
        plt.plot(x_t,dne_c_t,'bs',label = 'XGC')
        plt.plot(xfit_t,fwr_fit_t,'r-.',label = '2D fit')
        plt.plot(xfit_t,fwr3d_fit_t,'r--',label = '3D fit')
        plt.plot(xfit_t,xgc_fit_t,'b-',label = 'XGC fit')
        plt.xlabel('distance from center channel relfection($m$)')
        plt.ylabel('cross-corelation')
        plt.legend(labelspacing = 0.2,prop = {'size':12})
        plt.tight_layout()

def clear_all():
    global total_plot

    for i in range(total_plot):
        plt.close()




# Coherent Signal comparison





