# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:45:59 2015

Script to create all the plots used in Varenna-Lausane paper.

@author: lshi
"""

import numpy as np
import scipy.io.netcdf as nc
import matplotlib.pyplot as plt
from math import sqrt

import FPSDP.scripts.nstx_reflectometry.load_nstx_exp_ref as nstx_exp
import FPSDP.scripts.FWR_Postprocess.FWR2D_NSTX_139047_Postprocess as fwr_pp
from FPSDP.Diagnostics.Reflectometry.analysis import phase,magnitude,Coherent_Signal,Cross_Correlation,Cross_Correlation_by_fft
from FPSDP.Maths.Funcs import band_pass_box

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
    def __init__(self):
        Picture.__init__(self,'Fulltime Phase/dPhase Plot','Plot1: 55GHz phase/dphase plot showing chosen time period (0.632-0.633s), overall time is chosen to be 0.4-0.8s')
    def prepare(self):   
        """prepare the data for Plot1
        """
        self.tstart = 0.4
        self.tend = 0.8
        self.channel = 8 # channel 8 is 55GHz
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
    """Plot2: 55GHz zoomed in for 0.632-0.633s
    """
    def __init__(self):
        Picture.__init__(self,'Zoomed Phase/dPhase Plot','Plot2:55GHz zoomed in for 0.632-0.633s')

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
    
    def zoom(self):
        self.subfig1.set_xlim(0.632,0.633)
        self.subfig1.set_ylim(2510,2560)
        
        self.fig.canvas.draw()
        
class Plot3(Picture):
    """Plot3: 55GHz I component frequency domain compared with corresponding FWR result, non-relevant frequencies shaden.
    """
    def __init__(self,black_white=False):
        Picture.__init__(self,'Frequency domain comparison','Plot3: 55GHz I component frequency domain compared with corresponding FWR result, non-relevant frequencies shaden.')
        self.black_white = black_white
        
    def prepare(self):
        self.tstart = 0.632
        self.tend = 0.633
        self.f_low_norm = 1e4 #lower frequency cutoff for normalization set to be 100 kHz
        self.f_high_norm = 5e5 #high frequency cutoff for normalization set to 500 kHz
        self.f_low_show = 5e4 # lower frequency shown for shading
        self.f_high_show = 4e5# higher frequency shown for shading
        
        loader = nstx_exp.loaders[8]        
        sig,time = loader.signal(self.tstart,self.tend)
        ph,dph = phase(sig)
        n = len(time)
        dt = time[1]-time[0]
        
        #get the fft frequency array         
        self.freqs_nstx = np.fft.fftfreq(n,dt)[:n/2+1]
        idx_low,idx_high = np.searchsorted(self.freqs_nstx,[self.f_low_norm,self.f_high_norm]) #note that only first half of the frequency array is holding positive frequencies. The rest are negative ones.
        
        #get the spectra of the In-phase component, since it's real, only positive frequencies are needed
        spectrum_nstx = np.fft.rfft(ph)
        self.power_density_nstx = np.real(spectrum_nstx*np.conj(spectrum_nstx))
        pd_in_range = self.power_density_nstx[idx_low:idx_high]
        df = self.freqs_nstx[1]-self.freqs_nstx[0]        
        total_power_in_range = 0.5*df*np.sum(pd_in_range[:-1]+pd_in_range[1:]) #trapezoidal formula of integration is used here.
        
        #normalize the spectrum to make the in-range total energy be 1
        self.power_density_nstx /= total_power_in_range
        print('Experimental data ready.')
        
        ref2d = fwr_pp.load_2d([55],np.arange(100,220,1))
        sig_fwr = ref2d.E_out[0,:,0] #note that in reflectometer_output object, E_out is saved in shape (NF,NT,NC). Here we only need the time dimension for the chosen frequency and cross-section.
        ph_fwr,dph_fwr = phase(sig_fwr)        
        dt_fwr = 1e-6
        time_fwr = np.arange(100,220,1)*dt_fwr
        n_fwr = len(time_fwr)
        #similar normalization method for FWR results, temporary variables are reused for fwr quantities
        self.freqs_fwr = np.fft.fftfreq(n_fwr,dt_fwr)[:n_fwr/2+1]
        idx_low,idx_high = np.searchsorted(self.freqs_fwr,[self.f_low_norm,self.f_high_norm]) #note that only first half of the frequency array is holding positive frequencies. The rest are negative ones.

        spectrum_fwr = np.fft.rfft(ph_fwr)
        self.power_density_fwr = np.real(spectrum_fwr * np.conj(spectrum_fwr))
        pd_in_range = self.power_density_fwr[idx_low:idx_high]
        df = self.freqs_fwr[1]-self.freqs_fwr[0]
        total_power_in_range = 0.5*df*np.sum(pd_in_range[:-1]+pd_in_range[1:]) #trapezoidal formula of integration is used here.
        
        self.power_density_fwr /= total_power_in_range
        print('FWR data ready.')
        
    def show(self):
        if(self.black_white):
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
    def __init__(self,black_white = False):
        Picture.__init__(self,'Comparison of filtered signals','Plot 4: Comparison between filtered and original signals. From both experimental and simulation')
        self.black_white = black_white
        
    def prepare(self):
        self.tstart = 0.632
        self.tend = 0.633
        self.f_low = 5e4 #lower frequency set to be 50 kHz
        self.f_high = 5e5 #high end set to 500 kHz
        
        loader = nstx_exp.loaders[8]        
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
        
    def show(self):
        if(self.black_white):
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
        self.subfig1.legend(loc = 'upper left',prop = {'size':12})
        self.subfig2.legend(loc = 'upper left',prop = {'size':12})
        self.subfig3.legend(loc = 'upper left',prop = {'size':12})
        self.subfig4.legend(loc = 'upper left',prop = {'size':12})
        self.subfig4.set_xlabel('time (s)')
        self.subfig1.set_ylabel('$\phi$ (rad)')
        self.subfig2.set_ylabel('$\phi$ (rad)')
        self.subfig3.set_ylabel('magnitude (a.u.)')
        self.subfig4.set_ylabel('magnitude (a.u.)')
        
        
class Plot5(Plot4):
    """Plot 5: Experimental g factor as a function of window width, show that filtered signal has indeed a stable g factor
    
    """
    
    def __init__(self,black_white=False):
        Picture.__init__(self,'Plot5:g factor VS window widths','Plot 5: Experimental g factor as a function of window width, show that filtered signal has indeed a stable g factor')
        self.black_white = black_white
    def prepare(self):
        Plot4.prepare(self)
        
        t_total = self.tend - self.tstart
        self.avg_windows = np.logspace(-5,np.log10(t_total),50)
        
        t_upper = self.time[0]+self.avg_windows
        idx_upper = np.searchsorted(self.time,t_upper)
        self.g_orig = np.zeros_like(self.avg_windows)
        self.g_filt = np.zeros_like(self.avg_windows)
        for i in range(len(self.avg_windows)) :
            idx = idx_upper[i]
            sig = self.sig_nstx[:idx]
            sig_filt = self.filtered_sig_nstx[:idx]
            self.g_orig[i] = np.abs(Coherent_Signal(sig))
            self.g_filt[i] = np.abs(Coherent_Signal(sig_filt))
            
    def show(self):
        if(self.black_white):
            ls_orig = 'k--'
            ls_filt = 'k-'
            
        else:
            ls_orig = 'b-'
            ls_filt = 'r-'
            
        self.fig = plt.figure()
        self.subfig = self.fig.add_subplot(111)
        self.g_orig_line = self.subfig.semilogx(self.avg_windows,self.g_orig,ls_orig,linewidth = 1,label = 'ORIGINAL')
        self.g_filt_line = self.subfig.semilogx(self.avg_windows,self.g_filt,ls_filt,linewidth = 1,label = 'FILTERED')
        
        self.subfig.legend(loc = 'middle right', prop = {'size':14})
        
        self.subfig.set_xlabel('average time window(s)')
        self.subfig.set_ylabel('$|g|$')
        
        self.fig.canvas.draw()
        
            
                    