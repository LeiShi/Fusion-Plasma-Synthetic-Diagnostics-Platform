# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:45:59 2015

Script to create all the plots used in Varenna-Lausane paper.

@author: lshi
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import hilbert

import FPSDP.scripts.nstx_reflectometry.load_nstx_exp_ref as nstx_exp
import FPSDP.scripts.FWR_Postprocess.FWR2D_NSTX_139047_Postprocess as fwr_pp
from FPSDP.Diagnostics.Reflectometry.analysis import phase, magnitude, Coherent_Signal, Cross_Correlation, Cross_Correlation_by_fft
from FPSDP.Diagnostics.Reflectometry.NSTX.nstx import band_pass_filter
from FPSDP.Maths.Funcs import band_pass_box, sweeping_correlation
from FPSDP.Diagnostics.Reflectometry.FWR2D.Postprocess import fitting_cross_correlation,gaussian_fit,exponential_fit,remove_average_phase, remove_average_field

import pickle

with open('/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/new_ref_pos.pck','r') as f:
    ref_pos,freqs = pickle.load(f)
Z_mid = (ref_pos.shape[0]-1)/2
ref_pos_mid = ref_pos[Z_mid,:]
    
class Picture:
    """base class for paper pictures, contains abstract methods and components' names
    """
    def __init__(self,title,description):
        self.title = title
        self.description = description
    
    def prepare(self):
        """abstract method for preparing data before plotting
        """
        pass
    def show(self):
        """abstract method for plotting
        """
        pass
    #def close(self):
    #    """abstract method for closing the window
    #    """
    #    pass



class Plot1(Picture):
    """Plot1: 55GHz phase/dphase plot showing chosen time period (0.632-0.633s), overall time is chosen to be 0.4-0.8s
    """
    def __init__(self,channel = 11):
        Picture.__init__(self,'Fulltime Phase/dPhase Plot','Plot1: 55GHz phase/dphase plot showing chosen time period (0.632-0.633s), overall time is chosen to be 0.4-0.8s')
        self.channel = channel
    def prepare(self):   
        """prepare the data for Plot1
        """
        self.tstart = 0.4
        self.tend = 0.8

        self.t_on = 0.632
        self.t_off = 0.633
        
        loader = nstx_exp.loaders[self.channel]
        sig,self.time = loader.signal(self.tstart,self.tend)
        self.ph,self.dph = phase(sig)
        
    def show(self):
        """Make the plot with specified formats.
        """
        self.fig = plt.figure()
        self.subfig1 = self.fig.add_subplot(211)
        self.phase_plot = self.subfig1.plot(self.time,self.ph,'k-',linewidth=0.5)
        ylim1 = self.subfig1.get_ylim()
        self.vline1 = self.subfig1.plot([self.t_on,self.t_off],ylim1,'r-',linewidth=1)
        self.subfig1.set_xbound(self.tstart,self.tend)
        self.subfig1.set_title('(a)',y=-0.2)
        self.subfig1.set_ylabel('$\phi(rad)$')
        
        self.subfig2 = self.fig.add_subplot(212,sharex = self.subfig1)
        self.dph_plot = self.subfig2.plot(self.time[:-1],self.dph,'k-',linewidth = 0.05)
        ylim2 = self.subfig2.get_ylim()
        self.vline2 = self.subfig2.plot([self.t_on,self.t_off],ylim2,'r-',linewidth=1)        
        self.subfig2.set_xbound(self.tstart,self.tend)
        self.subfig2.set_title('(b)',y=-0.3)
        self.subfig2.set_xlabel('time(s)')
        self.subfig2.set_ylabel('$\Delta\phi(rad)$')
        
        self.fig.canvas.draw()


class Plot2(Plot1):
    """Plot2: zoomed in plot for chosen channel
    """
    def __init__(self,channel = 11):
        Plot1.__init__(self,channel)
        Picture.__init__(self,'Zoomed Phase/dPhase Plot','Plot2:chosen channel zoomed in for t1 to t2')

    def show(self):
        """Make the plot with specified formats.
        """
        self.fig = plt.figure()
        self.subfig1 = self.fig.add_subplot(211)
        self.phase_plot = self.subfig1.plot(self.time,self.ph,'k-',linewidth=0.5)
        self.subfig1.set_xbound(self.tstart,self.tend)
        self.subfig1.set_title('(c)',y=-0.2)
        self.subfig1.set_ylabel('$\phi(rad)$')
        
        self.subfig2 = self.fig.add_subplot(212,sharex = self.subfig1)
        self.dph_plot = self.subfig2.plot(self.time[:-1],self.dph,'k-',linewidth = 0.2)
        self.subfig2.set_xbound(self.tstart,self.tend)
        self.subfig2.set_title('(d)',y=-0.3)
        self.subfig2.set_xlabel('time(s)')
        self.subfig2.set_ylabel('$\Delta\phi(rad)$')
        
        self.fig.canvas.draw()        
    
    def zoom(self,t1,t2):
        assert(t1<t2 and t1>self.time[0] and t2 < self.time[-1])
        self.subfig1.set_xlim(t1,t2)
        arg1,arg2 =  np.searchsorted(self.time,[t1,t2])
        self.subfig1.set_ylim(np.min(self.ph[arg1:arg2])-1,np.max(self.ph[arg1:arg2]))
        
        self.fig.canvas.draw()
        
class Plot3(Picture):
    """Plot3: 62.5GHz phase signal frequency domain compared with corresponding FWR result, non-relevant frequencies shaden.
    """
    def __init__(self,channel = 11,channel_freq = 62.5):
        Picture.__init__(self,'Frequency domain comparison','Plot3: 62.5GHz phase signal frequency domain compared with corresponding FWR result, non-relevant frequencies shaden.')
        self.channel = channel
        self.channel_freq = channel_freq
        
    def prepare(self):
        self.tstart = 0.632
        self.tend = 0.633
        self.f_low_norm = 1e4 #lower frequency cutoff for normalization set to be 100 kHz
        self.f_high_norm = 5e5 #high frequency cutoff for normalization set to 500 kHz
        self.f_low_show = 4e4 # lower frequency shown for shading
        self.f_high_show = 5e5# higher frequency shown for shading
        
        
        loader = nstx_exp.loaders[self.channel]        
        sig,time = loader.signal(self.tstart,self.tend)
        ph,dph = phase(sig)
        n = len(time)
        dt = time[1]-time[0]
        
        #get the fft frequency array         
        self.freqs_nstx = np.fft.rfftfreq(n,dt)
        idx_low,idx_high = np.searchsorted(self.freqs_nstx,[self.f_low_norm,self.f_high_norm]) #note that only first half of the frequency array is holding positive frequencies. The rest are negative ones.
        
        #get the spectra of the phase signal, since it's real, only positive frequencies are needed
        spectrum_nstx = np.fft.rfft(ph)
        self.power_density_nstx = np.real(spectrum_nstx*np.conj(spectrum_nstx))
        pd_in_range = self.power_density_nstx[idx_low:idx_high]
        df = self.freqs_nstx[1]-self.freqs_nstx[0]        
        total_power_in_range = 0.5*df*np.sum(pd_in_range[:-1]+pd_in_range[1:]) #trapezoidal formula of integration is used here.
        
        #normalize the spectrum to make the in-range total energy be 1
        self.power_density_nstx /= total_power_in_range
        print('Experimental data ready.')
        
        ref2d = fwr_pp.load_2d([self.channel_freq],np.arange(100,220,1))
        sig_fwr = ref2d.E_out[0,:,0] #note that in reflectometer_output object, E_out is saved in shape (NF,NT,NC). Here we only need the time dimension for the chosen frequency and cross-section.
        ph_fwr,dph_fwr = phase(sig_fwr)        
        dt_fwr = 1e-6
        time_fwr = np.arange(100,220,1)*dt_fwr
        n_fwr = len(time_fwr)
        #similar normalization method for FWR results, temporary variables are reused for fwr quantities
        self.freqs_fwr = np.fft.rfftfreq(n_fwr,dt_fwr)
        idx_low,idx_high = np.searchsorted(self.freqs_fwr,[self.f_low_norm,self.f_high_norm]) #note that only first half of the frequency array is holding positive frequencies. The rest are negative ones.

        spectrum_fwr = np.fft.rfft(ph_fwr)
        self.power_density_fwr = np.real(spectrum_fwr * np.conj(spectrum_fwr))
        pd_in_range = self.power_density_fwr[idx_low:idx_high]
        df = self.freqs_fwr[1]-self.freqs_fwr[0]
        total_power_in_range = 0.5*df*np.sum(pd_in_range[:-1]+pd_in_range[1:]) #trapezoidal formula of integration is used here.
        
        self.power_density_fwr /= total_power_in_range
        print('FWR data ready.')
        
    def show(self,black_white = False):
        if(black_white):
            ls_nstx = 'k.'
            ls_fwr = 'k-'
            
        else:
            ls_nstx = 'b-'
            ls_fwr = 'r-'
        self.fig = plt.figure()
        self.subfig = self.fig.add_subplot(111)
        self.line_nstx = self.subfig.loglog(self.freqs_nstx,self.power_density_nstx,ls_nstx,linewidth = 0.5, label = 'EXP')
        self.line_fwr = self.subfig.loglog(self.freqs_fwr,self.power_density_fwr,ls_fwr,label = 'FWR')
        self.subfig.legend(loc = 'lower left')
        self.subfig.set_xlabel('frequency (Hz)')
        self.subfig.set_ylabel('normalized power density (a.u.)')
        
        #Shade over not used frequency bands
        
        freq_lowerband = np.linspace(1e3,self.f_low_show,10)
        freq_higherband = np.linspace(self.f_high_show,5e6,10)
       
        power_min = np.min([np.min(self.power_density_fwr),np.min(self.power_density_nstx)])
        power_max = np.max([np.max(self.power_density_fwr),np.max(self.power_density_nstx)])
        self.shade_lower = self.subfig.fill_between(freq_lowerband,power_min,power_max,color = 'g',alpha = 0.3)
        self.shade_higher = self.subfig.fill_between(freq_higherband,power_min,power_max,color = 'g',alpha = 0.3)
        
        
class Plot4(Picture):
    """Plot 4: Comparison between filtered and original signals. From experimental data, 55GHz channel.
    """
    def __init__(self,channel=11,channel_freq = 62.5):
        Picture.__init__(self,'Comparison of filtered signals','Plot 4: Comparison between filtered and original signals. From both experimental and simulation')
        self.channel = channel
        self.channel_freq = channel_freq
    def prepare(self,t_exp = 0.001):
        self.tstart = 0.632
        self.tend = self.tstart+t_exp
        self.f_low = 4e4 #lower frequency set to be 40 kHz
        self.f_high = 5e5 #high end set to 500 kHz
        
        loader = nstx_exp.loaders[self.channel]        
        self.sig_nstx,self.time = loader.signal(self.tstart,self.tend)
        self.phase_nstx,self.dph_nstx = phase(self.sig_nstx)
        self.magnitude_nstx = magnitude(self.sig_nstx)
        self.mean_mag_nstx = np.mean(self.magnitude_nstx)
        n = len(self.time)
        dt = self.time[1]-self.time[0]
        
        #get the fft frequency array         
        self.freqs_nstx = np.fft.fftfreq(n,dt)
        idx_low,idx_high = np.searchsorted(self.freqs_nstx[:n/2+1],[self.f_low,self.f_high]) #note that only first half of the frequency array is holding positive frequencies. The rest are negative ones.
        
        #get the fft result for experimental data
        self.phase_spectrum_nstx = np.fft.fft(self.phase_nstx) #Full fft is used here for filtering and inverse fft
        self.filtered_phase_spectrum_nstx = band_pass_box(self.phase_spectrum_nstx,idx_low,idx_high)

        self.mag_spectrum_nstx = np.fft.fft(self.magnitude_nstx)
        self.filtered_mag_spectrum_nstx= band_pass_box(self.mag_spectrum_nstx,idx_low,idx_high)
        
        self.filtered_phase_nstx = np.fft.ifft(self.filtered_phase_spectrum_nstx)
        self.filtered_mag_nstx = np.fft.ifft(self.filtered_mag_spectrum_nstx) + self.mean_mag_nstx # We want to stack the magnitude fluctuation on top of the averaged magnitude
        
        self.filtered_sig_nstx = self.filtered_mag_nstx * np.exp(1j * self.filtered_phase_nstx)
        
    def show(self,black_white = False):
        if(black_white):
            ls_orig = 'k.'
            ls_filt = 'k-'
            
        else:
            ls_orig = 'b-'
            ls_filt = 'r-'
        self.fig,(self.subfig1,self.subfig2,self.subfig3,self.subfig4) = plt.subplots(4,1,sharex=True)
        self.orig_pha_nstx_plot = self.subfig1.plot(self.time,self.phase_nstx,ls_orig,linewidth = 1, label = 'ORIGINAL_PHASE')
        self.filt_pha_nstx_plot= self.subfig2.plot(self.time,self.filtered_phase_nstx,ls_filt,linewidth = 1,label = 'FILTERED_PHASE')
        self.orig_mag_nstx_plot = self.subfig3.plot(self.time,self.magnitude_nstx,ls_orig,linewidth = 1,label = 'ORIGINAL_MAGNITUDE')
        self.filt_mag_nstx_plot= self.subfig4.plot(self.time,self.filtered_mag_nstx,ls_filt,linewidth = 1,label = 'FILTERED_MAGNITUDE')
        mag_low,mag_high = self.subfig3.get_ybound()        
        self.subfig4.set_ybound(mag_low,mag_high)        
        #self.line_fwr = self.subfig.loglog(self.freqs_fwr,self.power_density_fwr,ls_fwr,label = 'FWR')
        self.subfig1.legend(loc = 'best',prop = {'size':10})
        self.subfig2.legend(loc = 'best',prop = {'size':10})
        self.subfig3.legend(loc = 'best',prop = {'size':10})
        self.subfig4.legend(loc = 'best',prop = {'size':10})
        self.subfig4.set_xlabel('time (s)')
        self.subfig1.set_ylabel('$\phi$ (rad)')
        self.subfig2.set_ylabel('$\phi$ (rad)')
        self.subfig3.set_ylabel('magnitude (a.u.)')
        self.subfig4.set_ylabel('magnitude (a.u.)')
        self.subfig4.set_xbound(self.tstart,self.tend)
        
        
class Plot5(Plot4):
    """Plot 5: Experimental g factor as a function of window width, show that filtered signal has indeed a stable g factor
    
    """
    
    def __init__(self,channel=11,channel_freq=62.5):
        Plot4.__init__(self,channel,channel_freq)
        Picture.__init__(self,'Plot5:g factor VS window widths','Plot 5: Experimental g factor as a function of window width, show that filtered signal has indeed a stable g factor')
        
    def prepare(self):
        Plot4.prepare(self)
        
        t_total = self.tend - self.tstart
        self.avg_windows = np.logspace(-5,np.log10(t_total),50)
        
        t_lower = self.time[0]+1e-5
        idx_lower = np.searchsorted(self.time,t_lower)
        t_upper = self.time[idx_lower]+self.avg_windows
        idx_upper = np.searchsorted(self.time,t_upper)
        self.g_orig = np.zeros_like(self.avg_windows)
        self.g_filt = np.zeros_like(self.avg_windows)
        for i in range(len(self.avg_windows)) :
            idx = idx_lower + idx_upper[i]
            sig = self.sig_nstx[idx_lower:idx]
            sig_filt = self.filtered_sig_nstx[idx_lower:idx]
            self.g_orig[i] = np.abs(Coherent_Signal(sig))
            self.g_filt[i] = np.abs(Coherent_Signal(sig_filt))
            
    def show(self,black_white = False):
        if(black_white):
            ls_orig = 'k--'
            ls_filt = 'k-'
            
        else:
            ls_orig = 'b-'
            ls_filt = 'r-'
            
        self.fig = plt.figure()
        self.subfig = self.fig.add_subplot(111)
        self.g_orig_line = self.subfig.semilogx(self.avg_windows,self.g_orig,ls_orig,linewidth = 1,label = 'ORIGINAL')
        self.g_filt_line = self.subfig.semilogx(self.avg_windows,self.g_filt,ls_filt,linewidth = 1,label = 'FILTERED')
        
        self.subfig.legend(loc = 'best', prop = {'size':14})
        
        self.subfig.set_xlabel('average time window(s)')
        self.subfig.set_ylabel('$|g|$')
        
        self.fig.canvas.draw()
        
class Plot6(Picture):
    """ Plot 6: Four channel g factor plot. 55GHz, 57.5GHz, 60GHz, 62.5 GHz and 67.5GHz. 
    """
    def __init__(self):
        Picture.__init__(self,'Plot6:g factors for 5 channels', 'Plot 6: Four channel g factor plot. 55GHz, 57.5GHz, 60GHz, 62.5 GHz. and 67.5GHz')
    
    def prepare(self,t_exp = 0.001):
        
        #prepare the cut-off locations on the mid-plane
        self.x55 = ref_pos_mid[8]
        self.x575 = ref_pos_mid[9]
        self.x60 = ref_pos_mid[10]
        self.x625 = ref_pos_mid[11]
        self.x675 = ref_pos_mid[14]
        self.x65 = ref_pos_mid[12]
        self.x665 = ref_pos_mid[13]
        self.x = [self.x55,self.x575,self.x60,self.x625,self.x675,self.x65,self.x665]
        
        #First, we get experimental g factors ready
        self.t_sections = [[0.632,0.633],[0.6334,0.6351],[0.636,0.6385]]
        self.n_sections = len(self.t_sections)
        
        self.g55 = []
        self.g575 = []
        self.g60 = []
        self.g625 = []
        self.g675 = []
        
        self.f_low = 4e4
        self.f_high = 5e5 #frequency filter range set to 40kHz-500kHz
        
        l55 = nstx_exp.loaders[8]
        l575 = nstx_exp.loaders[9]
        l60 = nstx_exp.loaders[10]
        l625 = nstx_exp.loaders[11]
        l675 = nstx_exp.loaders[12]
        
        for i in range(self.n_sections):
            tstart = self.t_sections[i][0]
            tend = self.t_sections[i][1]
            
            sig55 ,time = l55.signal(tstart,tend)
            sig575 ,time = l575.signal(tstart,tend)                
            sig60 ,time = l60.signal(tstart,tend)
            sig625 ,time = l625.signal(tstart,tend)
            sig675, time = l675.signal(tstart,tend)
            dt = time[1]-time[0]        
            
            # Use band passing filter to filter the signal, so only mid range frequnecy perturbations are kept        
            sig55_filt = band_pass_filter(sig55,dt,self.f_low,self.f_high)
            sig575_filt = band_pass_filter(sig575,dt,self.f_low,self.f_high)
            sig60_filt = band_pass_filter(sig60,dt,self.f_low,self.f_high)
            sig625_filt = band_pass_filter(sig625,dt,self.f_low,self.f_high)
            sig675_filt = band_pass_filter(sig675,dt,self.f_low,self.f_high)
            
            theads = np.arange(tstart,tend-t_exp,t_exp/10) 
            ttails = theads+t_exp/10
            arg_heads = np.searchsorted(time,theads)
            arg_tails = np.searchsorted(time,ttails)
            
            #prepare the g-factors,keep them complex until we draw them
            for j in range(len(arg_heads)):
                self.g55.append(Coherent_Signal(sig55_filt[arg_heads[j]:arg_tails[j]]))
                self.g575.append(Coherent_Signal(sig575_filt[arg_heads[j]:arg_tails[j]]))
                self.g60.append(Coherent_Signal(sig60_filt[arg_heads[j]:arg_tails[j]]))
                self.g625.append(Coherent_Signal(sig625_filt[arg_heads[j]:arg_tails[j]]))
                self.g675.append(Coherent_Signal(sig675_filt[arg_heads[j]:arg_tails[j]]))
                
        self.g_exp = [[self.g55,self.g575,self.g60,self.g625,self.g675],
                      [np.mean(np.abs(self.g55)),np.mean(np.abs(self.g575)),np.mean(np.abs(self.g60)),np.mean(np.abs(self.g625)),np.mean(np.abs(self.g675))],
                      [np.std(np.abs(self.g55)),np.std(np.abs(self.g575)),np.std(np.abs(self.g60)),np.std(np.abs(self.g625)),np.std(np.abs(self.g675))]]
        
        # Now, we prepare the g factors from FWR2D
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_55.sav.npy'
        E55_2d = np.load(E_file)
        E55_2d = remove_average_phase(E55_2d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_57.5.sav.npy'
        E575_2d = np.load(E_file)
        E575_2d = remove_average_phase(E575_2d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_60.sav.npy'
        E60_2d = np.load(E_file)
        E60_2d = remove_average_phase(E60_2d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_62.5.sav.npy'
        E625_2d = np.load(E_file)
        E625_2d = remove_average_phase(E625_2d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_67.5.sav.npy'
        E675_2d = np.load(E_file)
        E675_2d = remove_average_phase(E675_2d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC_add_two_channels/E_out_65.0.sav.npy'
        E65_2d = np.load(E_file)
        E65_2d = remove_average_phase(E65_2d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC_add_two_channels/E_out_66.5.sav.npy'
        E665_2d = np.load(E_file)
        E665_2d = remove_average_phase(E665_2d)
        
        self.g55_2d = Coherent_Signal(E55_2d.flatten())
        self.g575_2d = Coherent_Signal(E575_2d.flatten())
        self.g60_2d = Coherent_Signal(E60_2d.flatten())
        self.g625_2d = Coherent_Signal(E625_2d.flatten())
        self.g675_2d = Coherent_Signal(E675_2d.flatten())
        self.g65_2d = Coherent_Signal(E65_2d.flatten())
        self.g665_2d = Coherent_Signal(E665_2d.flatten())
        self.g_2d = [self.g55_2d,self.g575_2d,self.g60_2d,self.g625_2d,self.g675_2d,self.g65_2d,self.g665_2d]
        
        # And g-factors from FWR3D
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_55GHz/E_out.sav.npy'
        E55_3d = np.load(E_file)
        E55_3d = remove_average_phase(E55_3d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_57.5GHz/E_out.sav.npy'
        E575_3d = np.load(E_file)
        E575_3d = remove_average_phase(E575_3d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_60GHz/E_out.sav.npy'
        E60_3d = np.load(E_file)
        E60_3d = remove_average_phase(E60_3d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_62.5GHz/E_out.sav.npy'
        E625_3d = np.load(E_file)
        E625_3d = remove_average_phase(E625_3d)
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_67.5GHz/E_out.sav.npy'
        E675_3d = np.load(E_file)
        E675_3d = remove_average_phase(E675_3d)
        
        self.g55_3d = Coherent_Signal(E55_3d.flatten())
        self.g575_3d = Coherent_Signal(E575_3d.flatten())
        self.g60_3d = Coherent_Signal(E60_3d.flatten())
        self.g625_3d = Coherent_Signal(E625_3d.flatten())
        self.g675_3d = Coherent_Signal(E675_3d.flatten())
        self.g_3d = [self.g55_3d,self.g575_3d,self.g60_3d,self.g625_3d,self.g675_3d]
        
    def show(self,black_white = False):
        if(black_white):
            color_exp = 'k'
            marker_exp = 's'
            ls_2d = 'k--'
            marker_2d = 'o'
            ls_3d = 'k-.'
            marker_3d = '^'
            
        else:
            color_exp = 'b'
            marker_exp = 's'
            ls_2d = 'g-'
            marker_2d = 'o'
            ls_3d = 'r-'
            marker_3d = '^'
            
        self.fig = plt.figure()
        self.subfig = self.fig.add_subplot(111)
        self.g_exp_line = self.subfig.errorbar(self.x[:5] ,self.g_exp[1],yerr=self.g_exp[2],ecolor = color_exp,linewidth = 1,marker = marker_exp,label = 'EXP')
        self.g_2d_line = self.subfig.errorbar(self.x[:7],np.abs(self.g_2d),yerr = 1./16, fmt = ls_2d,marker = marker_2d,linewidth = 1,label = 'FWR2D')
        self.g_3d_line = self.subfig.errorbar(self.x[:5],np.abs(self.g_3d),yerr = 1./16, fmt = ls_3d,marker = marker_3d,linewidth = 1,label = 'FWR3D')        
        
        self.subfig.legend(loc = 'best', prop = {'size':14})
        
        self.subfig.set_xlabel('R(m)')
        self.subfig.set_ylabel('$|g|$')
        
        self.subfig.set_ylim(0,1)
        xticks = self.subfig.get_xticks()[::2]
        xticklabels = [str(x) for x in xticks]
        self.subfig.set_xticks(xticks)
        self.subfig.set_xticklabels(xticklabels)
        
        self.fig.canvas.draw()
                
        
        
class Plot7(Picture):
    """ Plot 7: Cross-Correlation between 55GHz,57.5GHz, 60GHz, 62.5GHz and 67.5GHz channels. Center channel chosen to be 62.5GHz
    """  
    def __init__(self):
        Picture.__init__(self,'Plot7:Multi channel cross-section plots','Plot 7: Cross-Correlation between 55GHz,57.5GHz, 60GHz, 62.5GHz and 67.5GHz channels.')
    def prepare(self,center = 62.5,t_exp = 0.001): 
        #prepare the cut-off locations on the mid-plane
        if center == 67.5:
            channel_c = 14    
        elif center == 62.5:
            channel_c = 11
        elif center == 60:
            channel_c = 10
        elif center == 55:
            channel_c = 8
        else:
            channel_c = 14
            
        self.x55 = (ref_pos_mid[8]-ref_pos_mid[channel_c])
        self.x575 = (ref_pos_mid[9]-ref_pos_mid[channel_c])
        self.x60 = (ref_pos_mid[10]-ref_pos_mid[channel_c])
        self.x625 = (ref_pos_mid[11]-ref_pos_mid[channel_c])
        self.x675 = (ref_pos_mid[14]-ref_pos_mid[channel_c])
        self.x65 = (ref_pos_mid[12]-ref_pos_mid[channel_c])
        self.x665 = (ref_pos_mid[13]-ref_pos_mid[channel_c])
        self.x = [self.x55,self.x575,self.x60,self.x625,self.x675,self.x65,self.x665]    
        
        #First, we get experimental g factors ready
        
        self.tstart = 0.632
        self.tend = self.tstart + t_exp        
        
        self.f_low = 4e4
        self.f_high = 5e5 #frequency filter range set to 40kHz-500kHz
        
        l55 = nstx_exp.loaders[8]
        l575 = nstx_exp.loaders[9]
        l60 = nstx_exp.loaders[10]
        l625 = nstx_exp.loaders[11]
        l675 = nstx_exp.loaders[12]
        
        sig55 ,time = l55.signal(self.tstart,self.tend)
        sig575 ,time = l575.signal(self.tstart,self.tend)                
        sig60 ,time = l60.signal(self.tstart,self.tend)
        sig625 ,time = l625.signal(self.tstart,self.tend)
        sig675,time = l675.signal(self.tstart,self.tend)
        dt = time[1]-time[0]        
        
        # Use band passing filter to filter the signal, so only mid range frequnecy perturbations are kept        
        sig55_filt = band_pass_filter(sig55,dt,self.f_low,self.f_high)
        sig575_filt = band_pass_filter(sig575,dt,self.f_low,self.f_high)
        sig60_filt = band_pass_filter(sig60,dt,self.f_low,self.f_high)
        sig625_filt = band_pass_filter(sig625,dt,self.f_low,self.f_high)
        sig675_filt = band_pass_filter(sig675,dt,self.f_low,self.f_high)
        
        #prepare the gamma-factors,keep them complex until we draw them
        if center == 67.5:
            sig_c = sig675_filt
        elif center == 62.5:
            sig_c = sig625_filt
        elif center == 60:
            sig_c = sig60_filt
        elif center == 55:
            sig_c = sig55_filt
        else:
            sig_c = sig675_filt
        self.c55 = Cross_Correlation(sig_c,sig55_filt,'NORM')
        self.c575 = Cross_Correlation(sig_c,sig575_filt,'NORM')
        self.c60 = Cross_Correlation(sig_c,sig60_filt,'NORM')
        self.c625 = Cross_Correlation(sig_c,sig625_filt,'NORM')
        self.c675 = Cross_Correlation(sig_c,sig675_filt,'NORM')
        self.c_exp = [self.c55,self.c575,self.c60,self.c625,self.c675]
        
        # Now, we prepare the gamma factors from FWR2D
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_55.sav.npy'
        E55_2d = remove_average_field(remove_average_phase((np.load(E_file))))[:,:120,:].flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_57.5.sav.npy'
        E575_2d = remove_average_field(remove_average_phase((np.load(E_file))))[:,:120,:].flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_60.sav.npy'
        E60_2d = remove_average_field(remove_average_phase((np.load(E_file))))[:,:120,:].flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_62.5.sav.npy'
        E625_2d = remove_average_field(remove_average_phase((np.load(E_file))))[:,:120,:].flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC_add_two_channels/E_out_65.0.sav.npy'
        E650_2d = remove_average_field(remove_average_phase((np.load(E_file))))[:,:120,:].flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC_add_two_channels/E_out_66.5.sav.npy'
        E665_2d = remove_average_field(remove_average_phase((np.load(E_file))))[:,:120,:].flatten()        
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/RUNS/RUN_NSTX_139047_All_Channel_All_Time_MULTIPROC/E_out_67.5.sav.npy'
        E675_2d = remove_average_field(remove_average_phase((np.load(E_file))))[:,:120,:].flatten()
        
        E2d = [E55_2d,E575_2d,E60_2d,E625_2d,E675_2d,E650_2d,E665_2d]
        
        if center == 67.5:
            E2d_c = 4
        elif center == 62.5:
            E2d_c = 3
        elif center == 60:
            E2d_c = 2
        elif center == 55:
            E2d_c = 0
        else:
            E2d_c = 4       
        
        self.c_2d = []
        self.c_2d.append(Cross_Correlation(E2d[E2d_c],E2d[0],'NORM'))
        self.c_2d.append(Cross_Correlation(E2d[E2d_c],E2d[1],'NORM'))
        self.c_2d.append(Cross_Correlation(E2d[E2d_c],E2d[2],'NORM'))
        self.c_2d.append(Cross_Correlation(E2d[E2d_c],E2d[3],'NORM'))
        self.c_2d.append(Cross_Correlation(E2d[E2d_c],E2d[4],'NORM'))
        self.c_2d.append(Cross_Correlation(E2d[E2d_c],E2d[5],'NORM'))
        self.c_2d.append(Cross_Correlation(E2d[E2d_c],E2d[6],'NORM'))
        
        
        
        # And gamma-factors from FWR3D
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_55GHz/E_out.sav.npy'
        E55_3d = remove_average_field(remove_average_phase((np.load(E_file)))).flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_57.5GHz/E_out.sav.npy'
        E575_3d = remove_average_field(remove_average_phase((np.load(E_file)))).flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_60GHz/E_out.sav.npy'
        E60_3d = remove_average_field(remove_average_phase((np.load(E_file)))).flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_62.5GHz/E_out.sav.npy'
        E625_3d = remove_average_field(remove_average_phase((np.load(E_file)))).flatten()
        E_file = '/p/gkp/lshi/XGC1_NSTX_Case/Correlation_Runs/3DRUNS/RUN_NEWAll_16_cross_16_time_67.5GHz/E_out.sav.npy'
        E675_3d = remove_average_field(remove_average_phase((np.load(E_file)))).flatten()
        
        E3d = [E55_3d,E575_3d,E60_3d,E625_3d,E675_3d]
        

        if center == 67.5:
            E3d_c = 4
        elif center == 62.5:
            E3d_c = 3
        elif center == 60:
            E3d_c = 2
        elif center == 55:
            E3d_c = 0
        else:
            E3d_c = 4          
        
        self.c_3d = []
        self.c_3d.append(Cross_Correlation(E3d[E3d_c],E3d[0],'NORM'))
        self.c_3d.append(Cross_Correlation(E3d[E3d_c],E3d[1],'NORM'))
        self.c_3d.append(Cross_Correlation(E3d[E3d_c],E3d[2],'NORM'))
        self.c_3d.append(Cross_Correlation(E3d[E3d_c],E3d[3],'NORM'))
        self.c_3d.append(Cross_Correlation(E3d[E3d_c],E3d[4],'NORM'))
        
               
        # Gaussian fit of the cross-correlations 
        self.a_exp,self.sa_exp = fitting_cross_correlation(np.abs(self.c_exp),self.x[:5],'gaussian')
        self.a_2d,self.sa_2d = fitting_cross_correlation(np.abs(self.c_2d),self.x[:7],'gaussian')
        self.a_3d,self.sa_3d = fitting_cross_correlation(np.abs(self.c_3d),self.x[:5],'gaussian')
        
        self.xmax = 2*np.sqrt(np.max((np.abs(self.a_exp),np.abs(self.a_2d),np.abs(self.a_3d))))
        self.x_fit = np.linspace(-self.xmax,self.xmax,500)
        self.fit_exp = gaussian_fit(self.x_fit,self.a_exp)
        self.fit_2d = gaussian_fit(self.x_fit,self.a_2d)
        self.fit_3d = gaussian_fit(self.x_fit,self.a_3d)
        
        #Exponential fit of the cross-correlations
        self.e_exp,self.se_exp = fitting_cross_correlation(np.abs(self.c_exp),self.x[:5],'exponential')
        self.e_2d,self.se_2d = fitting_cross_correlation(np.abs(self.c_2d),self.x[:7],'exponential')
        self.e_3d,self.se_3d = fitting_cross_correlation(np.abs(self.c_3d),self.x[:5],'exponential')
        
        self.xmax_e = 2*np.max((np.abs(self.e_exp),np.abs(self.e_2d),np.abs(self.e_3d)))
        self.x_fit_e = np.linspace(-self.xmax,self.xmax_e,500)
        self.fit_exp_e = exponential_fit(self.x_fit_e,self.e_exp)
        self.fit_2d_e = exponential_fit(self.x_fit_e,self.e_2d)
        self.fit_3d_e = exponential_fit(self.x_fit_e,self.e_3d)        
        
        
        
    def show(self,black_white = False,Gaussian_fit = True):
        if(black_white):
            ls_exp = 'k-'
            c_exp = 'k'
            marker_exp = 's'
            ls_2d = 'k--'
            c_2d = 'k'
            marker_2d = 'o'
            ls_3d = 'k-.'
            c_3d = 'k'
            marker_3d = '^'
            
        else:
            ls_exp = 'b-'
            c_exp = 'b'
            marker_exp = 's'
            ls_2d = 'g-'
            c_2d = 'g'
            marker_2d = 'o'
            ls_3d = 'r-'
            c_3d = 'r'
            marker_3d = '^'
            
        self.fig = plt.figure()
        self.subfig = self.fig.add_subplot(111)
        self.c_exp_dots = self.subfig.scatter(self.x[:5] ,np.abs(self.c_exp),c = c_exp,marker = marker_exp,label = 'EXP')
        self.c_2d_dots = self.subfig.errorbar(self.x[:7],np.abs(self.c_2d),yerr = 1./16, mfc=c_2d, fmt = marker_2d,label = 'FWR2D')
        self.c_3d_dots = self.subfig.errorbar(self.x[:5],np.abs(self.c_3d),yerr = 1./16, mfc = c_3d, fmt = marker_3d,label = 'FWR3D')        
        
        if(Gaussian_fit):
            self.c_exp_fit_line = self.subfig.plot(self.x_fit,self.fit_exp,ls_exp,label = 'EXP FIT')
            self.c_2d_fit_line = self.subfig.plot(self.x_fit,self.fit_2d,ls_2d,label = 'FWR2D FIT')       
            self.c_3d_fit_line = self.subfig.plot(self.x_fit,self.fit_3d,ls_3d,label = 'FWR3D FIT')
            self.subfig.hlines(1/np.e,-self.xmax,self.xmax)
            self.subfig.set_xlim(-self.xmax,self.xmax) 
        else:
            self.c_exp_fit_line = self.subfig.plot(self.x_fit_e,self.fit_exp_e,ls_exp,label = 'EXP FIT')
            self.c_2d_fit_line = self.subfig.plot(self.x_fit_e,self.fit_2d_e,ls_2d,label = 'FWR2D FIT')       
            self.c_3d_fit_line = self.subfig.plot(self.x_fit_e,self.fit_3d_e,ls_3d,label = 'FWR3D FIT')
            self.subfig.hlines(1/np.e,-self.xmax_e,self.xmax_e)
            self.subfig.set_xlim(-self.xmax_e,self.xmax_e) 
                
        
        self.subfig.legend(loc = 'best', prop = {'size':14})
        
        self.subfig.set_xlabel('\Delta R(m)')
        self.subfig.set_ylabel('$|\gamma|$')
        self.subfig.set_ylim(0,1.1)
               
        
        self.fig.canvas.draw()
class Plot8(Picture):
    """Plot 8: Cross-Correlation of density perturbations between different cross-sections (try hilbert transform)
    Note: The 
    """
    def __init__(self):
        Picture.__init__(self,'Plot8: Density Cross-Correlation','Plot 8: Cross-Correlation of density perturbations between different cross-sections')
        
    def prepare(self):
        pass
    
        
            
                    
