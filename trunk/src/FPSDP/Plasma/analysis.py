"""Analysis modules for Plasma equilibrium and fluctuations

including:
    loading modules: load GTS, XGC, and other modules' saving files
    Spectrum analysis:use fft to analyze fluctuations
"""

import numpy as np
import scipy as sp

from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

from ..Geometry.Grid import Cartesian2D, Cartesian3D


class XGC_Density_Loader:
    """ class contains XGC analysis modules, now has only density fluctuation analysis. To be completed in the future.
    """

    def __init__(this,file_name):

        this.load_density(file_name)
        

    def load_density(this,file_name):
        """load density data from saving file given by XGC_Loader.save_dne()

        Arguments:
            file_name: string, the file name of the saving file, INCLUDING the full path, '.npz' extension is not necessary.

        Constructs:
        Depending on the dimension of XGC_loader spatial grid, and having non-adiabatic electrons or not, following arrays are constructed:
            X1D: 1d float array,1D coordinates along R (major radius) direction
            Y1D: 1d float array, coordinates along Z (vertical) direction
            dne_ad: nd float array, with shape (NC,NT,[NZ],NY,NX), adiabatic electron density perturbation on grid
            ne0: nd float array, with shape ([NZ],NY,NX), equilibrium electron density on grid
            
        (if 3D)    Z1D: 1D coordinates along R cross Z direction
        (if HaveElectron) nane: nd float array, same shape as dne_ad, non-adiabatci electron density perturbation on grid

        following quantities are constructed:
            NC: int, the total number of cross-sections loaded in XGC
            NT: int, the total time steps loaded in XGC
            dimension: int, the dimension of space used in XGC_loader, either 3 or 2
            HaveElectron: bool, the flag showing that non-adiabatic electrons included or not in XGC simulation
            
        """

        dne_file = np.load(file_name)

        if 'Z1D' in dne_file.keys():
            this.dimension = 3
        else:
            this.dimension = 2

        this.X1D = dne_file['X1D']
        this.Y1D = dne_file['Y1D']
        this.NX = this.X1D.shape[0]
        this.NY = this.Y1D.shape[0]
        if(this.dimension == 3):
            this.Z1D = dne_file['Z1D']
            this.NZ = this.Z1D.shape[0]
           
        this.dne_ad = dne_file['dne_ad']
        this.dne = this.dne_ad
        if 'nane' in dne_file.keys():
            this.HaveElection = True
            this.nane = dne_file['nane']
            this.dne += this.nane
        this.ne0 = dne_file['ne0']

        this.NC = this.dne_ad.shape[0]
        this.NT = this.dne_ad.shape[1]

    def interpolator_setup(this,cross_num,time_step):
        """ prepare the interpolators for the given time and cross section data
        returns:
            dne_sp
        """
        if(cross_num < this.NC and time_step < this.NT):
            if this.dimension == 2:
                dne_sp = RectBivariateSpline(this.Y1D,this.X1D,this.dne[cross_num,time_step,:,:])
            else:
                z_mid = (this.NZ-1)/2
                dne_sp = RectBivariateSpline(this.Y1D,this.X1D,this.dne[cross_num,time_step,z_mid,:,:])
            ne0_sp = RectBivariateSpline(this.Y1D,this.X1D,this.ne0[:,:])
        return dne_sp,ne0_sp

    def density_correlation(this,x0,direction = 'r',width = 0.01, z_center = 0,r_center = 1.46):
        """ fit the density fluctuation correlation funciton with gaussian form:
            <dn(x),dn(x0)> = dn^2 exp(-(x-x0)^2/lambda_n^2) cos(k_fl(x-x0))
            where dn, lambda_n, k_fl are parameters to be fitted. and x0 is chosen as the center location respect to which the correlation is considered.
            <...> is the ensemble average. 
        """

        n_x = 101
        if direction == 'r': 
            dx = np.linspace(0,width,n_x)
            x = x0+dx
            dne_cross = np.zeros((n_x))
            dne_x0 = np.zeros((this.NC,this.NT))
            dne_x = np.zeros((this.NC,this.NT,n_x))
            for i in range(this.NC):
                for j in range(this.NT):
                    dne_sp,ne0_sp = this.interpolator_setup(i,j)
                    ne0_x0 = ne0_sp(z_center,x0)[0,0]
                    dne_x0[i,j] = dne_sp(z_center,x0)[0,0]/ne0_x0
                    dne_x[i,j,:] = dne_sp(z_center,x)[0,:]/ne0_x0
            for i in range(n_x): 
                dne_cross[i] = np.average(dne_x0[:,:] * dne_x[:,:,i])/np.sqrt(np.average(dne_x0**2)*np.average(dne_x[:,:,i]**2))
           
        elif direction == 'z':
            dx = np.linspace(0,width,n_x)
            x = x0+dx
            dne_cross = np.zeros((n_x))
            dne_x0 = np.zeros((this.NC,this.NT))
            dne_x = np.zeros((this.NC,this.NT,n_x))
            for i in range(this.NC):
                for j in range(this.NT):
                    dne_sp,ne0_sp = this.interpolator_setup(i,j)
                    ne0_x0 = ne0_sp(x0,r_center)[0,0]
                    dne_x0[i,j] = dne_sp(x0,r_center)[0,0]/ne0_x0
                    dne_x[i,j,:] = dne_sp(x,r_center)[:,0]/ne0_x0
                    
            for i in range(n_x):        
                dne_cross[i] = np.average(dne_x0[:,:] * dne_x[:,:,i])/np.sqrt(np.average(dne_x0**2)*np.average(dne_x[:,:,i]**2))

        return curve_fit(gaussian_correlation_func,dx,dne_cross),dx,dne_cross    

    def averaged_fluctuation_level(this):
        """The amplitude of fluctuations at every spatial point is averaged over the ensemble of time and cross-sections, then divided by the equilibrium density
        """
        this.dne_bar = np.average(np.average(this.dne,axis = 1),axis = 0)
        this.dne_fluc = this.dne-this.dne_bar[np.newaxis,np.newaxis,:]
        dne_fluc_amp = np.abs(this.dne_fluc)
        #assuming sinoidal shape perturbation and uniformly distributed sampling, the maximum amplitude of the perturbation will be pi/2 times the averaged amplitude 
        this.dne_amp_bar = np.pi/2 *np.average(np.average(dne_fluc_amp,axis = 1),axis = 0) #average over time and cross section dimensions

        this.dn_over_n_raw = np.zeros(this.dne_amp_bar.shape)

        #pick up the non-zero density locations
        valid_idx = np.nonzero(this.ne0 > 0)

        this.dn_over_n_raw[valid_idx] = this.dne_amp_bar[valid_idx]/this.ne0[valid_idx]

        #screen out the large noise coming from very low ne0 places
        meaningful_idx = np.nonzero(this.dn_over_n_raw < 1)

        this.dn_over_n = np.zeros(this.dne_amp_bar.shape)
        this.dn_over_n[meaningful_idx] += this.dn_over_n_raw[meaningful_idx]

        return this.dn_over_n

    def get_frequencies(self):
        """calculate the relevant frequencies along the plasma midplane,return them as a dictionary
    
        arguments:
            time_eval: boolean, a flag for time evolution. Default to be False
    
        return: dictionary contains all the frequency arrays
            keywords: 'f_pi','f_pe','f_ci','f_ce','f_uh','f_lh','f_lx','f_ux'
    
        using formula in Page 28, NRL_Formulary, 2011 and Section 2-3, Eqn.(7-9), Waves in Plasmas, Stix.
        """
        if(isinstance(self.xgc_loader.grid,Cartesian2D) ):#2D mesh
            Zmid = (self.xgc_loader.grid.NZ-1)/2
            ne = self.xgc_loader.ne_on_grid[:,:,Zmid,:]*1e-6 #convert into cgs unit
            ni = ne #D plasma assumed,ignore the impurity. 
            mu = 2 # mu=m_i/m_p, in D plasma, it's 2
            B = self.xgc_loader.B_on_grid[Zmid,:]*1e4 #convert to cgs unit
        else:#3D mesh
            Ymid = (self.xgc_loader.grid.NY-1)/2
            Zmid = (self.xgc_loader.grid.NZ-1)/2
            ne = self.xgc_loader.ne_on_grid[:,:,Zmid,Ymid,:]*1e-6
            ni = ne
            mu = 2
            if(self.xgc_loader.equilibrium_mesh == '3D'):
                B = self.xgc_loader.B_on_grid[Zmid,Ymid,:]*1e4
            else:
                B = self.xgc_loader.B_on_grid[Ymid,:]*1e4
                
                f_pi = 2.1e2*mu**(-0.5)*np.sqrt(ni)
                f_pe = 8.98e3*np.sqrt(ne)
                f_ci = 1.52e3/mu*B
                f_ce = 2.8e6*B
                
                f_uh = np.sqrt(f_ce**2+f_pe**2)
                f_lh = np.sqrt( 1/ ( 1/(f_ci**2+f_pi**2) + 1/(f_ci*f_ce) ) )

                f_ux = 0.5*(f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))
                f_lx = 0.5*(-f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))

            return {'f_pi':f_pi,
                    'f_pe':f_pe,
                    'f_ci':f_ci,
                    'f_ce':f_ce,
                    'f_uh':f_uh,
                    'f_lh':f_lh,
                    'f_ux':f_ux,
                    'f_lx':f_lx}
        
    def get_ref_pos(self,freqs,mode = 'O'):
        """estimates the O-mode or X-mode reflection position in R direction for given frequencies.

        Input:

            freqs:sequence of floats, all the probing frequencies in GHz.
            mode: 'O' or 'X', indication of the wave polarization.
        return:
            2D array of floats,in the shape of (time_steps,freqs). R coordinates of all the estimated reflection position on mid plane, for each timestep and frequency.
        """
        if(isinstance(self.xgc_loader.grid,Cartesian2D)):
            R = self.xgc_loader.grid.R1D
        else:
            R = self.xgc_loader.grid.X1D

        plasma_freqs = self.get_frequencies()
    
        if(mode == 'O'):
            cutoff = plasma_freqs['f_pe']*1e-9
        elif(mode == 'X'):
            cutoff = plasma_freqs['f_ux']*1e-9 #convert into GHz
        else:
            print 'mode should be either O or X!'
            raise
        ref_idx = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))
        ref_pos = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))

        for j in range(cutoff.shape[0]):
            for k in range(cutoff.shape[1]):
                for i in range(len(freqs)):
    
                    ref_idx[j,k,i] = np.max(np.where(cutoff[j,k,:] > freqs[i])[0])#The right most index where the wave has been cutoff
                    
                    #linearly interpolate the wave frequency to the cutoff frequency curve, to find the reflected location
                    f1 = cutoff[j,k,ref_idx[j,k,i]+1]
                    f2 = cutoff[j,k,ref_idx[j,k,i]]
                    f3 = freqs[i]
                    
                    R1 = R[ref_idx[j,k,i]+1]
                    R2 = R[ref_idx[j,k,i]]
                    
                    ref_pos[j,k,i] = R2 + (f2-f3)/(f2-f1)*(R1-R2)
                    
                    
        return ref_pos

        

def gaussian_correlation_func(dx,lambda_n):
    """ fitting function used by curve_fit, dx is the variable, dn,lambda_n,k_fl are fitting parameters
    """
    return np.exp(-dx**2/lambda_n**2)
        

#General property calculations
def get_frequencies(prof_loader):
    """calculate the relevant frequencies along the plasma midplane,return them as a dictionary
    
    arguments:
        time_eval: boolean, a flag for time evolution. Default to be False
    
    return: dictionary contains all the frequency arrays
        keywords: 'f_pi','f_pe','f_ci','f_ce','f_uh','f_lh','f_lx','f_ux'
    
    using formula in Page 28, NRL_Formulary, 2011 and Section 2-3, Eqn.(7-9), Waves in Plasmas, Stix.
    """
    if(isinstance(prof_loader.grid,Cartesian2D) ):#2D mesh
        Zmid = (prof_loader.grid.NZ-1)/2
        ne = prof_loader.ne_on_grid[:,:,Zmid,:]*1e-6 #convert into cgs unit
        ni = ne #D plasma assumed,ignore the impurity. 
        mu = 2 # mu=m_i/m_p, in D plasma, it's 2
        B = prof_loader.B_on_grid[Zmid,:]*1e4 #convert to cgs unit
    else:#3D mesh
        Ymid = (prof_loader.grid.NY-1)/2
        Zmid = (prof_loader.grid.NZ-1)/2
        ne = prof_loader.ne_on_grid[:,:,Zmid,Ymid,:]*1e-6
        ni = ne
        mu = 2
        if(prof_loader.equilibrium_mesh == '3D'):
            B = prof_loader.B_on_grid[Zmid,Ymid,:]*1e4
        else:
            B = prof_loader.B_on_grid[Ymid,:]*1e4
            
    f_pi = 2.1e2*mu**(-0.5)*np.sqrt(ni)
    f_pe = 8.98e3*np.sqrt(ne)
    f_ci = 1.52e3/mu*B
    f_ce = 2.8e6*B
    
    f_uh = np.sqrt(f_ce**2+f_pe**2)
    f_lh = np.sqrt( 1/ ( 1/(f_ci**2+f_pi**2) + 1/(f_ci*f_ce) ) )

    f_ux = 0.5*(f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))
    f_lx = 0.5*(-f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))

    return {'f_pi':f_pi,
            'f_pe':f_pe,
            'f_ci':f_ci,
            'f_ce':f_ce,
            'f_uh':f_uh,
            'f_lh':f_lh,
            'f_ux':f_ux,
            'f_lx':f_lx}

def get_ref_pos(prof_loader,freqs,mode = 'O'):
    """estimates the O-mode reflection position in R direction for given frequencies.

    Input:
        prof_loader:XGC_loader object containing the profile and fluctuation information.
        freqs:sequence of floats, all the probing frequencies in GHz.
        mode: 'O' or 'X', indication of the wave polarization.
    return:
        2D array of floats,in the shape of (time_steps,freqs). R coordinates of all the estimated reflection position on mid plane, for each timestep and frequency.
    """
    if(isinstance(prof_loader.grid,Cartesian2D)):
        R = prof_loader.grid.R1D
    else:
        R = prof_loader.grid.X1D

    plasma_freqs = get_frequencies(prof_loader)
    
    if(mode == 'O'):
        cutoff = plasma_freqs['f_pe']*1e-9
    elif(mode == 'X'):
        cutoff = plasma_freqs['f_ux']*1e-9 #convert into GHz
    else:
        print 'mode should be either O or X!'
        raise
    ref_idx = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))
    ref_pos = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))

    for j in range(cutoff.shape[0]):
        for k in range(cutoff.shape[1]):
            for i in range(len(freqs)):
    
                ref_idx[j,k,i] = np.max(np.where(cutoff[j,k,:] > freqs[i])[0])#The right most index where the wave has been cutoff

                #linearly interpolate the wave frequency to the cutoff frequency curve, to find the reflected location
                f1 = cutoff[j,k,ref_idx[j,k,i]+1]
                f2 = cutoff[j,k,ref_idx[j,k,i]]
                f3 = freqs[i]
            
                R1 = R[ref_idx[j,k,i]+1]
                R2 = R[ref_idx[j,k,i]]
            
                ref_pos[j,k,i] = R2 + (f2-f3)/(f2-f1)*(R1-R2)
        
    
    return ref_pos


def get_gts_frequencies(gts_loader):
    """calculate the relevant frequencies along the plasma midplane,return them as a dictionary
    
    arguments:
        time_eval: boolean, a flag for time evolution. Default to be False
    
    return: dictionary contains all the frequency arrays
        keywords: 'f_pi','f_pe','f_ci','f_ce','f_uh','f_lh','f_lx','f_ux'
    
    using formula in Page 28, NRL_Formulary, 2011 and Section 2-3, Eqn.(7-9), Waves in Plasmas, Stix.
    """
    if(isinstance(gts_loader.grid,Cartesian2D) ):#2D mesh
        Zmid = (gts_loader.grid.NZ-1)/2
        ne = gts_loader.ne_on_grid[:,:,0,Zmid,:]*1e-6 #convert into cgs unit
        ni = ne #D plasma assumed,ignore the impurity. 
        mu = 2 # mu=m_i/m_p, in D plasma, it's 2
        B = gts_loader.B_on_grid[0,Zmid,:]*1e4 #convert to cgs unit
    else:#3D mesh , NOT FINISHED!!
        Ymid = (gts_loader.grid.NY-1)/2
        Zmid = (gts_loader.grid.NZ-1)/2
        ne = gts_loader.ne0_on_grid[Zmid,Ymid,:]*1e-6
        ni = ne
        mu = 2
        #if(gts_loader.equilibrium_mesh == '3D'):
        #    B = gts_loader.B_on_grid[Zmid,Ymid,:]*1e5
        #else:
        B = gts_loader.Bt_2d[0,Ymid,:]*1e4
            
    f_pi = 2.1e2*mu**(-0.5)*np.sqrt(ni)
    f_pe = 8.98e3*np.sqrt(ne)
    f_ci = 1.52e3/mu*B
    f_ce = 2.8e6*B
    
    f_uh = np.sqrt(f_ce**2+f_pe**2)
    f_lh = np.sqrt( 1/ ( 1/(f_ci**2+f_pi**2) + 1/(f_ci*f_ce) ) )

    f_ux = 0.5*(f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))
    f_lx = 0.5*(-f_ce + np.sqrt(f_ce**2+ 4*(f_pe**2 + f_ci*f_ce)))

    return {'f_pi':f_pi,
            'f_pe':f_pe,
            'f_ci':f_ci,
            'f_ce':f_ce,
            'f_uh':f_uh,
            'f_lh':f_lh,
            'f_ux':f_ux,
            'f_lx':f_lx}

def get_gts_ref_pos(gts_loader,freqs,mode = 'O'):
    """estimates the O-mode reflection position in R direction for given frequencies.

    Input:
        gts_loader:XGC_loader object containing the profile and fluctuation information.
        freqs:sequence of floats, all the probing frequencies in GHz.
        mode: 'O' or 'X', indication of the wave polarization.
    return:
        2D array of floats,in the shape of (time_steps,freqs). R coordinates of all the estimated reflection position on mid plane, for each timestep and frequency.
    """
    if(isinstance(gts_loader.grid,Cartesian2D)):
        R = gts_loader.grid.R1D
    else:
        R = gts_loader.grid.X1D

    plasma_freqs = get_gts_frequencies(gts_loader)
    
    if(mode == 'O'):
        cutoff = plasma_freqs['f_pe']*1e-9
    elif(mode == 'X'):
        cutoff = plasma_freqs['f_ux']*1e-9 #convert into GHz
    else:
        print 'mode should be either O or X!'
        raise
    ref_idx = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))
    ref_pos = np.zeros((cutoff.shape[0],cutoff.shape[1],freqs.shape[0]))

    for j in range(cutoff.shape[0]):
        for k in range(cutoff.shape[1]):
            for i in range(len(freqs)):
    
                ref_idx[j,k,i] = np.max(np.where(cutoff[j,k,:] > freqs[i])[0])#The right most index where the wave has been cutoff

                #linearly interpolate the wave frequency to the cutoff frequency curve, to find the reflected location
                f1 = cutoff[j,k,ref_idx[j,k,i]+1]
                f2 = cutoff[j,k,ref_idx[j,k,i]]
                f3 = freqs[i]
            
                R1 = R[ref_idx[j,k,i]+1]
                R2 = R[ref_idx[j,k,i]]
            
                ref_pos[j,k,i] = R2 + (f2-f3)/(f2-f1)*(R1-R2)
        
    
    return ref_pos
    
    
