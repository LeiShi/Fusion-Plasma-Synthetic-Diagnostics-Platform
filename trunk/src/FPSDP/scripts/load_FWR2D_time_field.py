

"""script to collect all the time series FWR2D results 
"""

from scipy.io.netcdf import netcdf_file
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import Image

run_path = '/p/gkp/lshi/GTS_ALCATOR_Case/L_mode/RUNS/test_RUNS/RUN_TEST_MULTIPROC_batch/140/'

t_arr = range(1,200,30)

freq = 140

def fwr2d_output_fname(path,freq):

    return path+'out_{0:0>6.2f}_equ.cdf'.format(freq)

class FWR2D_time_data:

    def __init__(self,run_path = run_path,t_arr = t_arr,freq=freq,Full_Load = True):
        if(Full_Load):
            self.run_path = run_path
            self.t_arr = t_arr
            self.nt = len(t_arr)
            self.freq = freq
            self.initialize_arrays()
            self.load_data()
        else:
            print 'Full_Load = False. Nothing done. Please load data from saved files.'
        
            

    def initialize_arrays(self):
        """Read in necessary parameters and setup empty arrays
        """
        fname = fwr2d_output_fname(self.run_path+'1/0/',self.freq)
        f0 = netcdf_file(fname,'r')

        #get coordinates and dimensions
        self.p_x = np.copy(f0.variables['p_x'].data)
        self.s_x = np.copy(f0.variables['s_x'].data)
        self.y = np.copy(f0.variables['p_y'].data)
        self.p_nx = np.copy(f0.dimensions['p_nx'])
        self.s_nx = np.copy(f0.dimensions['s_nx'])
        self.ny = np.copy(f0.dimensions['p_ny'])

        f0.close()
        # setup ndarrays for field data
        self.p_E = np.empty((self.nt,self.ny,self.p_nx),dtype = 'complex')
        self.s_E = np.empty((self.nt,self.ny,self.s_nx),dtype = 'complex')


    def load_data(self):
        """Load all the field data from FWR2D output files
        """
        for i in range(self.nt):
            t = self.t_arr[i]
            fname = fwr2d_output_fname('{0}{1}/0/'.format(self.run_path,t),self.freq)
            f = netcdf_file(fname,'r')

            self.s_E[i,:,:] = f.variables['s_Er'][:,:] + 1j*f.variables['s_Ei'][:,:]
            self.p_E[i,:,:] = f.variables['p_Er'][1,:,:] + 1j*f.variables['p_Ei'][1,:,:]

            f.close()

    def save(self,info_fname = './time_info.sav',data_fname = './time_data.sav'):
        with open(info_fname,'w') as f:
            info_dic = dict(run_path = self.run_path, t_arr = self.t_arr, nt = self.nt, freq = self.freq, p_nx = self.p_nx,s_nx = self.s_nx,ny = self.ny)
            pickle.dump(info_dic,f)

        np.savez(data_fname, p_Er=np.real(self.p_E),p_Ei = np.imag(self.p_E), s_Er=np.real(self.s_E),s_Ei = np.imag(self.s_E))

    def load(self,info_fname = './time_info.sav',data_fname = './time_data.sav'):
        with open(info_fname, 'r') as f:
            info_dic = pickle.load(f)
            for (key,value) in info_dic.items():
                setattr(self,key,value)
        if not '.npz' in data_fname:
            data_fname += '.npz'
        f = np.load(data_fname)
        for (key,value) in f.items():
            setattr(self,key,value)
        f.close()

        self.p_E = self.p_Er + 1j * self.p_Ei
        self.s_E = self.s_Er + 1j * self.s_Ei
                
        
        
        

class Animator:
    """A class that make simple 2D movies
    
    """

    def __init__(self,fullwave_data,para_data,phase_data,plasma_mode='L',t_arr = None):
        """Initialize with fullwave region field data, paraxial region field data, and received phase data.
        Arguments:
            fullwave_data: ndarray (nt,ny,s_nx), electric field in fullwave region
            para_data: ndarray (nt,ny,p_nx), electric field in paraxial region
            phase_data: ndarray(nt), phase value over time
            t_arr: Optional, ndarray (nt), exact time array
        """
        self.fullwave_data = fullwave_data
        self.para_data = para_data
        self.phase_data = phase_data
        self.plasma_mode = plasma_mode
        self.nt = self.fullwave_data.shape[0]
        if(t_arr != None):
            self.t_arr = t_arr

        else:
            self.t_arr = np.arange(self.nt)
        

    def setup_fig(self,**Params):
        self.fig = plt.figure(**Params)

    def setup_frame(self,extent_fw=None,extent_para=None,**Params):
        gs = GridSpec(1,3,width_ratios = [1,2,1])

        self.sp1 = plt.subplot(gs[0])
        self.im_fw = plt.imshow(self.fullwave_data[0,...],extent = extent_fw,**Params)
        self.sp1.set_ylabel('Z(cm)')
        
        self.sp2 = plt.subplot(gs[1])
        self.im_para = plt.imshow(self.para_data[0,...],extent = extent_para,**Params)
        self.sp2.set_yticks([])
        self.sp2.set_xlabel('R(cm)')

        self.sp3 = plt.subplot(gs[2])
        self.plot3 = plt.plot(self.t_arr,self.phase_data)
        xticks = self.sp3.get_xticks()
        new_xticks = np.linspace(xticks[0],xticks[-1],5)
        new_xtick_labels = ['{:.1f}'.format(x) for x in new_xticks]
        self.sp3.set_xticks(new_xticks)
        self.sp3.set_xticklabels(new_xtick_labels)
        self.sp3.set_xlabel('time(micro second)')
        self.sp3.set_ylabel('phase(rad)')
        self.time_line = plt.vlines(0,np.min(self.phase_data),np.max(self.phase_data))

        self.t_tag = plt.figtext(0.9,0.9,'t={0:.3f}'.format(self.t_arr[0]),ha = 'right')
        self.title = plt.figtext(0.5,0.95,'Reflected Wave Field and Phase in {}-mode Case'.format(self.plasma_mode),ha = 'center',va = 'bottom')
        plt.show()

    def update_frame(self,t):
        self.im_fw.set_data(self.fullwave_data[t,...])
        self.im_para.set_data(self.para_data[t,...])
        self.t_tag.set_text('t={0:.3f}'.format(self.t_arr[t]))
        self.time_line.set_segments([[[self.t_arr[t],np.min(self.phase_data)],[self.t_arr[t],np.max(self.phase_data)]]])

    def start_animation(self,**Params):
        """useful animiation parameters:
            repeat:bool, if True, animation automatically repeat after end of frames
            interval: int, draw new frame every 'interval' miliseconds
        """
        self.anim = FuncAnimation(self.fig,self.update_frame, frames = self.nt,**Params)
        plt.show()

    def save_frames(self,fname='movie'):
        self.setup_fig(figsize = (16,10))
        self.setup_frame([80,90,-15,15],[90,110,-15,15],aspect = 1,origin = 'lower')
        for t in range(self.nt):
            self.update_frame(t)
            self.fig.canvas.draw()
            wholename = './{0}{1:0>5}'.format(fname,t)
            self.fig.savefig(wholename+'.png')
            Image.open(wholename+'.png').save(wholename+'.jpg','JPEG')

        
        
    
            
        
        
        
        
        
