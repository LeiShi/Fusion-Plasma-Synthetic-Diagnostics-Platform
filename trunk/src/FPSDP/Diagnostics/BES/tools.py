import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, splev, splrep
from scipy.fftpack import fft2, ifft2
import FPSDP.Diagnostics.Beam.beam as beam_
import FPSDP.Diagnostics.BES.bes as bes_
import FPSDP.Plasma.XGC_Profile.load_XGC_local as xgc_

# command for using pdflatex for the graph in pgf
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)

class Tools:
    """ Defines a few tools for doing some computations on the BES image
    
    Two different methods of initalization have been made:
    The first one is the usual __init__ function that take as input the photon radiance, the R,Z coordinates
    and the psin value, and the second one read a npz file with a standard order and calls the __init__ method.

    :param np.array[Nt,Nfib] I: Photon radiance
    :param np.array[Nfib] R: R coordinate
    :param np.array[Nfib] Z: Z coordinate
    :param np.array[Nfib] psin: :math:`\Psi_n` value
    :param str name_id: Name defining the data (used for the titles))
    :param bool I_fluc: Input are fluctuation or total intensity
    """

    def __init__(self,I,R,Z,psin,name_id,I_fluc=False):
        if I_fluc == False:
            self.Itot = I
            Iav = np.mean(I,axis=0)
            self.I = (I-Iav)/Iav
        else:
            self.I = I

        self.name_id = name_id
        self.R = R
        self.Z = Z
        self.psin = psin
        self.Nfoc = Z.shape[0]
        self.Nt = I.shape[0]

    @classmethod
    def init_from_file(cls,filename,name_id,fluc=False):
        """ Load of file in the numpy format (npz)

        The data inside it should be in the following order: [I,psin,pos_foc,...]

        :param str filename: Name of the file
        :param bool fluc: The intensity inside the file is the total (False) or the fluctuation (True)
        :return: New instance variable
        :rtype: Tools
        
        """
        data = np.load(filename)
        I = data['arr_0']
        psin = data['arr_1']
        pos = data['arr_2']
        R = np.sqrt(np.sum(pos[:,0:2]**2,axis=1))
        Z = pos[:,2]
        return cls(I,R,Z,psin,name_id,fluc)

    def interpolate(self,Nr,Nz,I,timestep=None,kind='linear'):
        """ Interpolate all the data on a spatial mesh and create this mesh.
        The interpolation is done for each timestep

        :param int Nr: Number of points for the discretization in R
        :param int Nz: Number of points for the discretization in Z 
        :param np.array[Ntime,R,Z] I: Picture to interpolate
        :param int timestep: Time step wanted (None compute all of them)

        :return: r,z of the mesh and I on the mesh
        :rtype: tuple(np.array[Nr],np.array[Nz],np.array[Ntime,Nr,Nz])
        """
        r = np.linspace(np.min(self.R),np.max(self.R),Nr)
        z = np.linspace(np.min(self.Z),np.max(self.Z),Nz)

        if timestep is None:
            Igrid = np.zeros((self.Nt,Nr,Nz))
            for i in range(self.Nt):
                temp = interp2d(self.R,self.Z,I[i,:],kind)
                Igrid[i,:,:] = temp(r,z).T
        else:
            Igrid = np.zeros((Nr,Nz))
            temp = interp2d(self.R,self.Z,I[timestep,:],kind)
            Igrid = temp(r,z).T
        return r,z,Igrid

    def fluctuations_picture(self,timestep,v=40,total=False):
        """ Plot a graph of the fluctuation
        :param int timestep: Index of the timestep
        :param int v: Number of ticks for the colorbar
        :param bool total: Choice between total intensity or only fluctuation
        """
        if total:
            I = self.Itot[timestep,:]
            v = np.linspace(np.min(self.Itot),np.max(self.Itot))
        else:
            I = self.I[timestep,:]
            v = np.linspace(np.min(self.I),np.max(self.I))

        plt.figure()
        plt.tricontourf(self.R,self.Z,I,v)
        plt.colorbar()
        plt.plot(self.R,self.Z,'x')
        plt.tricontour(self.R,self.Z,self.psin,[1])
        plt.xlabel('R[m]')
        plt.ylabel('Z[m]')
        plt.show()

    def fluctuations_movie(self,v=40,name_movie='movie_fl.mp4'):
        """ Make a movie from the data and save it in movie_fl.mp4
        :param int v: Number of ticks for the colorbar
        :param str name_movie: Name of the movie that will be saved
        """
        from matplotlib import animation
        fig = plt.figure()
        ax = plt.gca()

        plt.xlabel('R[m]')
        plt.ylabel('Z[m]')

        v = np.linspace(np.min(self.I),np.max(self.I),v)
        
        def animate(i):
            ax.cla()
            print 'Timestep: ', i
            plt.title('Timestep : {}'.format(i))
            plt.tricontourf(self.R,self.Z,self.I[i,:],v)
            plt.plot(self.R,self.Z,'x')
            plt.tricontour(self.R,self.Z,self.psin,[1])
            fig.canvas.draw()
            return None

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, frames=self.I.shape[0],repeat=False)
        
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html

        FFwriter = animation.FFMpegWriter()
        anim.save(name_movie, writer=FFwriter,fps=15, extra_args=['-vcodec', 'libx264'])

        
    def comparison_picture(self,tool2,timestep,v=40,total=False):
        """ Make a picture from the data.

        :param Tools tool2: Second instance of the Tools class
        :param int timestep: Index of the timestep
        :param int v: Number of ticks for the colorbar
        :param bool total: Total intensity or only fluctuation
        """
        if total:
            I = self.Itot
            I2 = tool2.Itot
            v1 = np.linspace(0,np.max(I),v)
            v2 = np.linspace(0,np.max(I2),v)
        else:
            I = self.I
            I2 = tool2.I
            lim1 = np.min([np.max(I), -np.min(I)])
            lim2 = np.min([np.max(I2), -np.min(I2)])
            v1 = np.linspace(np.min(I),np.max(I),v)
            v2 = np.linspace(np.min(I),np.max(I2),v)

        fig, axarr = plt.subplots(1,2)
        
        axarr[0].set_xlabel('R[m]')
        axarr[0].set_ylabel('Z[m]')
        axarr[1].set_xlabel('R[m]')

        axarr[0].locator_params(axis = 'x',nbins=5)
        
        axarr[1].locator_params(axis = 'x',nbins=5)
        axarr[0].set_title(tool2.name_id)
        axarr[1].set_title(self.name_id)
        
        axarr[1].set_yticklabels([])
        plt.suptitle('Timestep : {}'.format(timestep))
        
        axarr[0].plot(tool2.R,tool2.Z,'x')
        tri0 = axarr[0].tricontourf(tool2.R,tool2.Z,I2[timestep,:],v2,extend='both')
        axarr[0].tricontour(self.R,self.Z,self.psin,[1])
        
        axarr[1].plot(self.R,self.Z,'x')
        tri1 = axarr[1].tricontourf(self.R,self.Z,I[timestep,:],v1,extend='both')
        axarr[1].tricontour(self.R,self.Z,self.psin,[1])
        

        cb = plt.colorbar(tri0,ax=axarr[0])
        cb = plt.colorbar(tri1,ax=axarr[1])
        
        plt.show()


    def comparison_movie(self,tool2,v=40,name_movie='movie_comp.mp4',interpolation=False):
        """ Make a movie from the data and save it in movie_comp.mp4
        self should be the density fluctuation
        :param Tools tool2: BES images
        :param int v: Number of ticks for the colorbar
        :param str name_movie: Name of the output movie
        :param bool interpolation: Choice of making an interpolation on a grid
        """
        Nr = 100
        Nz = 120
        from matplotlib import animation
        if interpolation:
            r,z,I_id = self.interpolate(Nr,Nz,self.I)
            r,z,I = self.interpolate(Nr,Nz,tool2.I)

        lim1 = np.min([np.max(self.I), -np.min(self.I)])
        lim2 = np.min([np.max(tool2.I), -np.min(tool2.I)])
        v1 = np.linspace(-lim1,lim1,v)
        v2 = np.linspace(-lim2,lim2,v)
        
        #v1 = np.linspace(np.min(self.I),np.max(self.I),v)
        #v2 = np.linspace(np.min(tool2.I),np.max(tool2.I),v)
        fig, axarr = plt.subplots(1,2)
        
        axarr[0].set_xlabel('R[m]')
        axarr[0].set_ylabel('Z[m]')
        axarr[1].set_xlabel('R[m]')

        
        tri0 = axarr[1].tricontourf(self.R,self.Z,self.I[0,:],v1)
        tri1 = axarr[0].tricontourf(tool2.R,tool2.Z,tool2.I[0,:],v2)

        cb = plt.colorbar(tri0,ax=axarr[0])
        cb = plt.colorbar(tri1,ax=axarr[1])


        def animate(i):
            print 'Timestep: ', i
            axarr[0].cla()
            axarr[0].locator_params(axis = 'x',nbins=5)
            axarr[1].cla()
            axarr[1].locator_params(axis = 'x',nbins=5)
            axarr[1].set_title(tool2.name_id)
            axarr[0].set_title(self.name_id)
  
            axarr[1].set_yticklabels([])
            plt.suptitle('Timestep : {}'.format(i))
            
            axarr[0].plot(tool2.R,tool2.Z,'x')
            if interpolation:
                tri0 = axarr[1].contourf(r,z,I[i,...].T,v2,extend='both')
            else:
                tri0 = axarr[1].tricontourf(tool2.R,tool2.Z,tool2.I[i,:],v2,extend='both')
            axarr[1].tricontour(tool2.R,tool2.Z,tool2.psin,[1])

            axarr[1].plot(self.R,self.Z,'x')
            if interpolation:
                tri1 = axarr[0].contourf(r,z,I_id[i,...].T,v1,extend='both')
            else:
                tri1 = axarr[0].tricontourf(self.R,self.Z,self.I[i,:],v1,extend='both')
            axarr[0].tricontour(self.R,self.Z,self.psin,[1])

            fig.canvas.draw()
            return None

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, frames=self.I.shape[0],repeat=False)
        
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html

        FFwriter = animation.FFMpegWriter()
        anim.save(name_movie, writer=FFwriter,fps=15, extra_args=['-vcodec', 'libx264'])


    def crosscorrelation(self, Nr=60, Nz=70, dr_max=0.1, dz_max=0.03, dkr_max=300, dkz_max=300, graph='3d',figure=True):
        """ Plot or just compute the shape of the crosscorrelation from the point R[0],Z[0]

        :param int Nr: Number of points for the discretization
        :param int Nz: Number of points for the discretization
        :param float dr_max: Distance max to show in the figure for the real space crosscorrelation
        :param float dz_max: Distance max to show in the figure for the real space crosscorrelation
        :param float dkr_max: Distance max to show in the figure for the Fourier space crosscorrelation
        :param float dkz_max: Distance max to show in the figure for the Fourier space crosscorrelation
        :param str graph: Choice between surface or contourf graph ('2d')
        :param bool figure: Choice between plot or computing

        :return: If (figure == False), the correlation and its fourier transform are returned.\
        The size of the two arrays is defined by the cutoff limits (dr_max,dkr_max,...))
        :rtype: (np.array[R,Z],np.array[R,Z])
        """
        from scipy.signal import correlate2d
        from mpl_toolkits.mplot3d import Axes3D

        corr = np.zeros(2*np.array([Nr,Nz])-1)
        
        r,z,Igrid = self.interpolate(Nr,Nz,self.I)
            
        for i in range(self.Nt):
            if np.isfinite(Igrid[i,...]).all():
                corr += correlate2d(Igrid[i,...],Igrid[i,...])
            else:
                print 'miss'


        corr /= np.max(corr)
        temp = np.zeros(2*Nr-1)
        r = r-r[0]
        z = z-z[0]
        temp[:Nr-1] = -r[Nr-1:0:-1]
        temp[Nr-1:] = r
        r = temp
        temp = np.zeros(2*Nz-1)
        temp[:Nz-1] = -z[Nz-1:0:-1]
        temp[Nz-1:] = z
        z = temp

        indr_ = (r >= 0) & (r<dr_max)
        indz_ = (z >= 0) & (z<dz_max)
        ind = np.einsum('i,j->ij',indr_,indz_)
        
        rm, zm = np.meshgrid(r[indr_],z[indz_])
        fft_corr = np.abs(np.fft.fft2(corr))/np.sqrt(Nr*Nz)
        corr = corr[ind]
        corr = np.reshape(corr,[np.sum(indr_),np.sum(indz_)])
        
        krfft = np.fft.fftfreq(Nr,r[2]-r[1])
        kzfft = np.fft.fftfreq(Nz,z[2]-z[1])
        indrfft = (krfft >= 0) & (krfft < dkr_max)
        indzfft = (kzfft >= 0) & (kzfft < dkz_max)
        krfft,kzfft = np.meshgrid(krfft[indrfft],kzfft[indzfft])

        if figure:
            fig = plt.figure()
            plt.title('Correlation')
            if graph == '3d':
                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(rm,zm,corr.T, cmap=matplotlib.cm.coolwarm, rstride=2, linewidth=0, cstride=2)
                fig.colorbar(surf)
            else:
                plt.contourf(r[indr_],z[indz_],corr.T,30)
                plt.colorbar()
            plt.xlabel('R')
            plt.ylabel('Z')

            fft_ = fft_corr[indrfft,:]
            fft_ = fft_[:,indzfft]
            
            fig = plt.figure()
            plt.title('FFT of the Correlation')
            if graph == '3d':
                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(krfft,kzfft,fft_.T, cmap=matplotlib.cm.coolwarm,rstride=1,linewidth=0, cstride=1)
                fig.colorbar(surf)
            else:
                plt.contourf(krfft,kzfft,fft_.T,40)
                plt.colorbar()
            plt.xlabel('k_r')
            plt.ylabel('k_z')
            
            plt.figure()
            plt.plot(r[indr_],corr[:,0],label='R')
            plt.plot(z[indz_],corr[0,:],label='Z')
            plt.legend()
            
            plt.show()
        else:
            return corr,fft_

    def radial_dep_correlation(self,Nr=40,Zref=0.01,eps=0.4,figure=True):
        """ Show the characteristic length of the radial correlation length as a function
        of the radial position

        :param int Nr: Number of points for the discretization
        :param fload Zref: Horizontal plane wanted
        :param float eps: Relative interval accepted for Zref
        :param bool figure: Choice between plot or computing

        :return: If (figure == False), the radius and the correlation are returned.\
        :rtype: (np.array[Nr],np.array[Nr])
        
        """
        ind = np.abs((self.Z - Zref)/Zref) < eps
        N = np.sum(ind)
        
        corr = np.zeros(Nr)
        r = np.linspace(np.min(self.R[ind]),np.max(self.R[ind]),Nr)

        R_temp = self.R[ind]
        Igrid = np.zeros((self.Nt,Nr))
        for i in range(self.Nt):
            temp = splrep(R_temp,self.I[i,ind])
            Igrid[i,:] = splev(r,temp)
        
        for i in range(Nr):
            temp = np.zeros(Nr)
            for j in range(Nr):
                temp[j] = np.corrcoef(Igrid[:,i],Igrid[:,j])[0,1]
            temp = r[(temp<np.exp(-1)) & (r>r[i])]-r[i]
            if temp.shape[0] > 0:
                corr[i] = temp[0]
            else:
                corr[i] = np.nan
        if not figure:
            return r,corr,ind
        else:

            fig = plt.figure()
            plt.plot(r,corr)
            plt.xlabel('R [m]')
            plt.ylabel('Correlation length')
            plt.show()

    def comparison_radial_correlation_lenght(self,tool2,Nr=40,Zref=0.01,eps=0.4):
        """ Show the characteristic length of the radial correlation length as a function
        of the radial position for two different diagnostics.

        :param Tools tool2: Second instance of the Tools class
        :param int Nr: Number of points for the discretization
        :param fload Zref: Horizontal plane wanted
        :param float eps: Relative interval accepted for Zref
        :param bool figure: Choice between plot or computing
        """

        r0, corr0,ind0 = self.radial_dep_correlation(Nr,Zref,eps,figure=False)
        print 'First computation done'
        r1, corr1,ind1 = tool2.radial_dep_correlation(Nr,Zref,eps,figure=False)

        plt.figure()
        plt.plot(r0,100*corr0,label=self.name_id)
        plt.plot(r1,100*corr1,label=tool2.name_id)
        plt.xlabel('Radius [m]')
        plt.ylabel('Correlation length [cm]')

        psi_n = splrep(self.R[ind0],self.psin[ind0])
        psi_n = splev(r0,psi_n)

        ind_sep = np.abs(1.0-psi_n) < 1e-3
        if np.sum(ind_sep) > 0:
            ind_sep = np.argmax(ind_sep)
        xa,xb,ya,yb = plt.axis()
        plt.plot([r0[ind_sep],r0[ind_sep]],[ya,yb])
        plt.legend()
        plt.show()



def put_two_files_together(name1,name2,outputname):
    """ When doing two simulations on a different time intervals,
    this function can be used to put the output in one file.
    """
    data1 = np.load(name1)
    I1 = data1['arr_0']
    psin1 = data1['arr_1']
    pos1 = data1['arr_2']

    data2 = np.load(name2)
    I2 = data2['arr_0']
    psin2 = data2['arr_1']
    pos2 = data2['arr_2']

    check = True
    if (pos1 != pos2).any():
        check = False
    if (psin1 != psin2).any():
        check = False

    if not check:
        raise NameError('Not the same simulation')

    else:
        I = np.zeros((I1.shape[0]+I2.shape[0],I1.shape[1]))
        I[:I1.shape[0],:] = I1
        I[I1.shape[0]:,:] = I2
        np.savez(outputname,I,psin1,pos1,data1['arr_3'])
        
    
""" Define a few test for checking the data given by the code
It contains all the code used for the figure in my report
"""
name = 'FPSDP/Diagnostics/BES/bes.in'



def beam_density(t=150):
    """ Compare the beam density of the equilibrium case and of the equilibrium+fluctuations case

    Three plots will be done: a first one with the relative difference between the equilibrium and the fluctuation cases,
    a second one with the absolute value of both and the last one shows the ratio of each components as a function of the distance

    """
    bes = bes_.BES(name)
    bes.beam.eq = True
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.data.current = t
    bes.beam.data.load_next_time_step(increase=False)
    bes.beam.compute_beam_on_mesh()
    nb_eq = bes.beam.density_beam
    ne_eq = bes.beam.get_quantities(bes.beam.mesh,t,['ne'],True,check=False)[0]
    bes.beam.eq = False
    
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.compute_beam_on_mesh()
    nb_fl = bes.beam.density_beam
    ne_fl = bes.beam.get_quantities(bes.beam.mesh,t,['ne'],False,check=False)[0]
    dl = np.sqrt(np.sum((bes.beam.mesh-bes.beam.pos[np.newaxis,:])**2,axis=1))
    
    
    fig, axarr = plt.subplots(2,sharex=True)
        
    axarr[1].plot(dl,((nb_eq-nb_fl)/nb_eq).T)
    plt.xlabel('Distance [m]')
    axarr[1].set_ylabel('Error')
    axarr[1].legend(['1st component','2nd component','3rd component'],loc=3)
    axarr[1].grid(True)
    
    axarr[0].plot(dl,nb_eq.T)
    axarr[0].plot(dl,nb_fl.T)
    axarr[0].grid(True)
    axarr[0].set_ylabel('Beam density [m$^{-3}$]')
    axarr[0].legend(['1st Eq','2nd Eq','3rd Eq'],loc=3)
    
    
    fig, axarr = plt.subplots(2,sharex=True)
        
    axarr[1].plot(dl,((ne_eq-ne_fl)/ne_eq))
    plt.xlabel('Distance [m]')
    axarr[1].set_ylabel('Error')
    axarr[1].grid(True)
    
    axarr[0].plot(dl,ne_eq)
    axarr[0].plot(dl,ne_fl)
    axarr[0].grid(True)
    axarr[0].legend(['Equilibrium','Fluctuations'],loc=2)
    axarr[0].set_ylabel('Electron Density [m$^{-3}$]')
        
    plt.figure()
    tot = np.sum(nb_eq,axis=0)
    plt.plot(dl,(nb_eq/tot[np.newaxis,:]).T)
    tot = np.sum(nb_fl,axis=0)
    plt.plot(dl,(nb_fl/tot[np.newaxis,:]).T)
    plt.legend(['1st Eq','2nd Eq','3rd Eq'],loc=2)
    plt.xlabel('Distance [m]')
    plt.ylabel('Ratio')
    plt.grid(True)
    
    plt.show()

def beam_emission(Nr=40,t=81):
    """ Shows the effect of the beam density on the emission and the effect of the lifetime.

    Three plots will be done: a first one with the emission without the lifetime, a second one with
    lifetime and a last one that show the relative difference between the two

    :param int Nr: Number of points along the radial direction of the beam
    :param int t: Timestep wanted
    """
    bes = bes_.BES(name)
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.data.current = t
    bes.beam.data.load_next_time_step(increase=False)
    bes.beam.compute_beam_on_mesh()

    r_max = 0.25
    dl = np.sqrt(np.sum((bes.beam.mesh-bes.beam.pos[np.newaxis,:])**2,axis=1))
    r = np.linspace(-r_max,r_max,Nr)
    R,L = np.meshgrid(r,dl)
    R = R.flatten()
    L = L.flatten()
    perp = np.copy(bes.beam.direc)
    perp[0] = perp[1]
    perp[1] = -bes.beam.direc[0]
    pos = bes.beam.pos[np.newaxis,:] + L[:,np.newaxis]*bes.beam.direc + R[:,np.newaxis]*perp
    emis = bes.beam.get_emis(pos,t)[0,:]
    emis_l = bes.beam.get_emis_lifetime(pos,t)[0,:]

    emis = np.reshape(emis, (-1,Nr))
    emis_l = np.reshape(emis_l, (-1,Nr))
    R = np.reshape(R, (-1,Nr))
    L = np.reshape(L, (-1,Nr))

    v = 50
    plt.figure()
    plt.title('Photon radiance')
    plt.contourf(R,L,emis,v)
    width = bes.beam.stddev_h
    plt.plot([-width, -width],[0, dl[-1]],'--k')
    plt.plot([width, width],[0, dl[-1]],'--k')
    plt.colorbar()
    plt.xlabel('Distance from the central line [m]')
    plt.ylabel('Distance from the source [m]')

    plt.figure()
    plt.title('Photon radiance')
    plt.contourf(R,L,emis_l,v)
    plt.plot([-width, -width],[0, dl[-1]],'--k')
    plt.plot([width, width],[0, dl[-1]],'--k')
    plt.colorbar()
    plt.xlabel('Distance from the central line [m]')
    plt.ylabel('Distance from the source [m]')

    v = np.linspace(-1,1,v)
    plt.figure()
    plt.title('Relative Error')
    plt.contourf(R,L,(emis_l-emis)/emis,v)
    plt.plot([-width, -width],[0, dl[-1]],'--k')
    plt.plot([width, width],[0, dl[-1]],'--k')
    plt.colorbar()
    plt.xlabel('Distance from the central line [m]')
    plt.ylabel('Distance from the source [m]')

    plt.show()

def check_convergence_beam_density(t=140,eq=False):
    """ Plot the error of the beam density at the last computed point as a function of
    the number of intervals.

    :param int t: Time step wanted
    :param bool eq: If equilibrium is wanted
    """
    bes = bes_.BES(name)
    
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.data.current = t
    bes.beam.data.load_next_time_step(increase=False)
    bes.beam.data.dphi = 0.01

    Nsample = 20
    N = np.logspace(2,3,Nsample)
    nb = np.zeros((bes.beam.beam_comp.shape[0],Nsample))
    Nref = 3000

    bes.beam.eq = eq
    for i in range(Nsample):
        print N[i]
        bes.beam.Nz = np.round(N[i])
        bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
        bes.beam.create_mesh()
        bes.beam.compute_beam_on_mesh()
        nb[:,i] = bes.beam.density_beam[:,-1]

    bes.beam.Nz = Nref
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.create_mesh()
    bes.beam.compute_beam_on_mesh()
    nbref = bes.beam.density_beam[:,-1]

    #np.save('test_beam_conv',np.abs(nb-nbref[:,np.newaxis])/nbref[:,np.newaxis])
    nb_test = np.load('test_beam_conv.npy')
    for i in range(nb.shape[0]):
        plt.loglog(N,np.abs((nb[i,:]-nbref[i])/nbref[i]),label='Beam Component {}'.format(i+1))

    plt.loglog(N,nb_test[0,:],label='Test Function')
    plt.title('Beam Density')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Number of intervals')
    plt.ylabel('Error')
    plt.show()


def check_convergence_lifetime(t=81,fib=4,beam_comp=0):
    """ Plot the error as a function of the number of interval for the computation
    of the emission with lifetime.

    :param int t: Timestep wanted
    :param int fib: Index of the fiber to use for the computation
    """

    bes = bes_.BES(name)
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.data.current = t
    bes.beam.data.dphi = 2*np.pi/(16*200)
    bes.beam.data.load_next_time_step(increase=False)
    bes.beam.compute_beam_on_mesh()
    bes.beam.t_max = 8.0

    N = 10*np.round(np.logspace(2,3,30))
    Nref = 40000

    #bes.pos_foc[fib,:] += 0.4*bes.beam.direc
    emis = np.zeros(N.shape)
    #test = np.zeros(N.shape)
    for i,Nlt in enumerate(N):
        print i
        bes.beam.Nlt = Nlt
        emis[i] = bes.beam.get_emis_lifetime(bes.pos_foc[fib,:],t)[beam_comp,:]
        #test[i] = bes.beam.get_emis_lifetime(bes.pos_foc[fib,:],t,test=True)[beam_comp,:]
    bes.beam.Nlt = Nref
    emis_ref = bes.beam.get_emis_lifetime(bes.pos_foc[fib,:],t)[beam_comp,:]
    #test_ref = bes.beam.get_emis_lifetime(bes.pos_foc[fib,:],t,test=True)[beam_comp,:]
    
    plt.figure()
    plt.title('Lifetime convergence')
    plt.loglog(N,np.abs((emis-emis_ref)/emis_ref))
    #plt.loglog(N,np.abs((test-test_ref)/test_ref))
    plt.grid(True)
    plt.ylabel('Error')
    plt.xlabel('Number of interval')
    plt.show()


def check_convergence_field_line(fib=4,phi=0.2,nber_plane=16,fwd=True):
    """ Plot the error of the field line following integration as a function of the integration step
    at the position given by the focus point of the fiber and by phi

    :param int fib: Index of the fiber
    :param float phi: Angle to use for the position
    :param int nber_plane: Number of planes in XGC1
    :param bool fwd: Choice of the forward or backward direction
    """

    bes = bes_.BES(name)
    foc = bes.pos_foc[fib,:]
    r = np.sqrt(np.sum(foc[0:2]**2))
    z = foc[2]
    
    N = np.logspace(1,3,20)
    Nref =5000

    dphi = 2*np.pi/(nber_plane*N)
    dphi_ref = 2*np.pi/(nber_plane*Nref)

    if fwd:
        ind = 1
    else:
        ind = 0
        
    pos = np.zeros((N.shape[0],3))
    pos_ref = np.zeros(3)

    phi = np.atleast_1d(phi)
    r = np.atleast_1d(r)
    z = np.atleast_1d(z)

    prev,nex = xgc_.get_interp_planes_local(bes.beam.data,phi)
    for i,dphi_ in enumerate(dphi):
        bes.beam.data.dphi = dphi_
        temp = bes.beam.data.find_interp_positions(r,z,phi,prev,nex)
        pos[i,0] = temp[ind,1] # R
        pos[i,1] = temp[ind,0] # Z
        pos[i,2] = temp[ind,2] # s
        
    bes.beam.data.dphi = dphi_ref
    temp = bes.beam.data.find_interp_positions(r,z,phi,prev,nex)
    pos_ref[0] = temp[ind,1] # R
    pos_ref[1] = temp[ind,0] # Z
    pos_ref[2] = temp[ind,2] # s

    print temp[ind,:]
    
    plt.figure()
    plt.loglog(dphi,np.abs((pos[:,0]-pos_ref[0])/pos_ref[0]),label='R')
    plt.loglog(dphi,np.abs((pos[:,1]-pos_ref[1])/pos_ref[1]),label='Z')
    plt.loglog(dphi,np.abs((pos[:,2]-pos_ref[2])/pos_ref[2]),label='s')
    plt.grid(True)
    plt.legend(loc=2)
    
    plt.ylabel('Error')
    plt.xlabel('$\Delta\phi [rad]$')
    plt.show()


def check_convergence_interpolation_data(t=140,fib=4,phi=0.2,nber_plane=16,eq=False):
    """ Plot the error of the field line following interpolation as a function of the
    step size of the field line integration at the position given by 
    the focus point of the fiber and by phi
    
    :param int t: Timestep wanted
    :param int fib: Index of the fiber
    :param float phi: Angle to use for the position
    :param int nber_plane: Number of planes in XGC1
    :param bool eq: Choice of the equilibrium or not
    """


    bes = bes_.BES(name)
    bes.beam.data.current = t
    bes.beam.data.load_next_time_step(increase=False)
    foc = bes.pos_foc[fib,:]
    r = np.sqrt(np.sum(foc[0:2]**2))
    x = np.cos(phi)*r
    y = np.sin(phi)*r
    z = foc[2]
    
    N = np.logspace(0.5,2,20)
    Nref = 1000

    dphi = 2*np.pi/(nber_plane*N)
    dphi_ref = 2*np.pi/(nber_plane*Nref)

        
    ne = np.zeros((N.shape[0]))

    phi = np.atleast_1d(phi)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    pos = np.array([x,y,z])
    pos = np.atleast_2d(pos).T
    for i,dphi_ in enumerate(dphi):
        bes.beam.data.dphi = dphi_
        ne[i] = bes.beam.data.interpolate_data(pos,t,['ne'],eq,check=True)[0]
                
    bes.beam.data.dphi = dphi_ref
    ne_ref = bes.beam.data.interpolate_data(pos,t,['ne'],eq,check=True)[0]
    
    plt.figure()
    plt.title('Interpolation of data')
    plt.loglog(N,np.abs((ne-ne_ref)/ne_ref))
    plt.grid(True)    
    plt.ylabel('Error')
    plt.xlabel('Number of interval')
    plt.show()


def check_convergence_optic_int(t=81,fib=4):
    """ Plot the error of the image captured by a fiber as a function of the number
    of interval for the integration along the optical axis

    :param int t: Time step wanted
    :param int fib: Index of the fiber
    """

    bes = bes_.BES(name)
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.data.current = t
    bes.beam.data.load_next_time_step(increase=False)
    bes.beam.compute_beam_on_mesh()
    bes.pos_foc = np.atleast_2d(bes.pos_foc[fib,:])
    
    Nsample = 30
    N_ref = 300
    N = np.logspace(1,2,Nsample)
    I1D = np.zeros(Nsample)
    I2D = np.zeros(Nsample)
    #Itest = np.zeros(Nsample)

    for i,N_ in enumerate(N):
        print i
        bes.Nint = N_
        bes.type_int = '1D'
        bes.solid = np.zeros((1,bes.Nint-1,2,21) )
        I1D[i] = bes.intensity(t,0,comp_eps=True)
        bes.type_int = '2D'
        I2D[i] = bes.intensity(t,0,comp_eps=True)
        #Itest[i] = bes.intensity(t,0,comp_eps=True,test=True)
        

    bes.Nint = N_ref
    bes.type_int = '1D'
    bes.solid = np.zeros((1,bes.Nint-1,2,21) )
    I1D_ref = bes.intensity(t,0,comp_eps=True)
    bes.type_int = '2D'
    I2D_ref = bes.intensity(t,0,comp_eps=True)
    #Itest_ref = bes.intensity(t,0,comp_eps=True,test=True)
    

    
    plt.figure()
    plt.title('Optical integral')
    plt.loglog(N,np.abs((I1D-I1D_ref)/I1D_ref),label='1D integral')
    plt.loglog(N,np.abs((I2D-I2D_ref)/I2D_ref),label='2D integral')
    #plt.loglog(N,np.abs((Itest-Itest_ref)/Itest_ref),label='Test')
    plt.grid(True)
    plt.ylabel('Error')
    plt.xlabel('Number of interval')
    plt.legend()
    plt.show()

def check_convergence_solid_angle_to_analy(R=0.5,Z=0.1,radius=0.4,Nth=100,Nr=10):
    """ Is used for checking the convergence to the analytical formula
    in the case where the intersection tends to a single point

    :param float R: Distance from the central axis of the emission position
    :param float Z: Z-coordinate (along the central line)
    :param float radius: Radius of the obstacle
    """
    from FPSDP.Maths.Funcs import solid_angle_disk, solid_angle_seg
    Nsample = 100
    phi = np.linspace(np.pi/4.0,np.pi/2.0,Nsample)
    y = -radius*np.sin(phi)
    x = radius*np.cos(phi)

    x1 = np.array([x,y]).T
    x2 = np.array([-x,y]).T
    pos = np.array(np.ones((Nsample,1))*np.array([[0,R,Z]]))
    
    solid = solid_angle_seg(pos,[x1,x2],radius,0,Nth,Nr)
    analytical = solid_angle_disk(pos[0,:],radius)

    phi = np.cos(phi)
    plt.figure()
    plt.title('Verification of the Solid Angle Computation')
    plt.grid(True)
    plt.plot(phi,np.abs((solid-analytical)/analytical),'x')
    plt.xlabel('Cosine of the Intersection Angle')
    plt.ylabel('Error')
    plt.show()

def check_convergence_solid_angle():
    """ Plot the error of the solid angle as a function of the number of interval for R and :math:`\Theta`
    """
    import FPSDP.Maths.Funcs as F

    pos = np.array([[0.1,0.012,0.5]])
    x = [np.array([[-0.08,0.1268857]]),np.array([[-0.047708,0.1422107]])]
    r = 0.15

    Nsample = 30
    N = np.logspace(0.5,2,Nsample)

    N_ref = 300
    R = np.zeros(Nsample)
    Th = np.zeros(Nsample)

    for i,N_ in enumerate(N):
        print i
        N_ = np.round(N_)
        R[i] = F.solid_angle_seg(pos,x,r,False,N_ref,N_)
        Th[i] = F.solid_angle_seg(pos,x,r,False,N_,N_ref)

    ref = F.solid_angle_seg(pos,x,r,False,N_ref,N_ref)
    
    plt.figure()
    plt.title('Solid Angle')
    plt.loglog(N,np.abs((R-ref)/ref),label='R')
    plt.loglog(N,np.abs((Th-ref)/ref),label='$\Theta$')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Number of interval')
    plt.show()


def check_field_line_integration(R=2.23,Z=0.01,phi=0.1,Nrot=10,fwd=True,data_name='/project/projectdirs/m499/jlang/particle_pinch/'):
    """ Use the coordinate of a fiber for making a field line integration
    on Nrot tour and plot the value of :math:`\psi_n` at each toroidal planes.

    :param float R: Initial position
    :param float Z: Initial position
    :param int Nrot: Number of toroidal tour
    :param bool fwd: Choice between forward or backward rotation
    """
    Rinit = R
    Zinit = Z
    # the value of the limits are random, we do not care about them in this function
    data = xgc_.XGC_Loader_local(data_name,1,182,1,np.array([[0.1,1],[0.1,1]]),0.001)
    # take all the planes into account
    n_plane = data.n_plane
    
    Rcur = np.atleast_1d(R)
    Zcur = np.atleast_1d(Z)
    phicur = np.atleast_1d(phi)
    R = np.zeros(n_plane*Nrot)
    Z = np.zeros(n_plane*Nrot)

    dPhi = 2*np.pi/n_plane
    phi_planes = np.arange(n_plane)*dPhi+data.shift

    sign = 1.0
    if not data.CO_DIR:
        sign = -1.0
    for i in range(n_plane*Nrot):
        prevplane,nextplane = xgc_.get_interp_planes_local(data,phicur)
        # MAYBE NEED TO CHANGE IT IF data.CO_DIR (I do not have data with it for checking)
        # this condition is due to the fact that phicur == phi_planes and that I want to
        # go to a next plane
        if phicur == phi_planes[nextplane]:
            nextplane -= 1
            nextplane = nextplane % n_plane

        interp = data.find_interp_positions(Rcur,Zcur,phicur,prevplane,nextplane)
        if fwd:
            Rcur = interp[1,0,:]
            Zcur = interp[1,1,:]
            phicur = phi_planes[nextplane]
        else:
            Rcur = interp[0,0,:]
            Zcur = interp[0,1,:]
            phicur = phi_planes[prevplane]

        R[i] = Rcur
        Z[i] = Zcur

    psi = data.psi_interp(np.array([R,Z]).T)/data.psi_x
    psi_ref = data.psi_interp(np.array([Rinit,Zinit]).T)/data.psi_x
    N = np.arange(n_plane*Nrot)+1
    plt.figure()
    plt.plot(Rinit,Zinit,'rd')
    plt.plot(R,Z,'-x')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')

    plt.figure()
    plt.loglog(N,np.abs(psi-psi_ref)/psi_ref)
    plt.xlabel('Number of step')
    plt.ylabel('Relative error in $\psi$')
    plt.show()

    
def check_geometry(minorR=0.67,majorR=1.67):
    """ Plot the geometry readed by the synthetic diagnostics
    The default Tokamak is the D3D (only a very simple model of the tokamak
    is considered)

    :param float minorR: Minor radius of the tokamak (distance between the center\
    of the plasma to the wall)
    :param float majorR: Major radius of the tokamak (distance between the symmetry axis\
    and the center of the plasma))
    """
    print 'Default values assume D3D'

    bes = bes_.BES(name)
    foc = bes.pos_foc
    r = np.sqrt(np.sum(foc[:,:2]**2,axis=1))
    
    plt.figure()
    plt.title('Top view')
    perp = np.copy(bes.beam.direc)
    perp[1] = perp[0]
    perp[0] = -bes.beam.direc[1]
    stdp = bes.beam.mesh[:,:2]+perp[:2]*bes.beam.stddev_h
    stdm = bes.beam.mesh[:,:2]-perp[:2]*bes.beam.stddev_h
    plt.plot(bes.beam.mesh[:,0],bes.beam.mesh[:,1],'-x',label='Beam')
    plt.plot(stdp[:,0],stdp[:,1],stdm[:,0],stdm[:,1],label='Standard deviation')
    plt.plot(foc[:,0],foc[:,1],'x',label='Focus Points')

    rad = np.linspace(0,2*np.pi,1000)
    xmin_tok = (majorR-minorR)*np.cos(rad)
    ymin_tok = (majorR-minorR)*np.sin(rad)

    xmax_tok = (majorR+minorR)*np.cos(rad)
    ymax_tok = (majorR+minorR)*np.sin(rad)

    plt.plot(xmin_tok,ymin_tok,xmax_tok,ymax_tok,label='Tokamak')
    lim = bes.limits
    plt.plot(bes.pos_lens[0],bes.pos_lens[1],'x',label='lens')
    plt.plot([lim[0,0],lim[0,1],lim[0,1],lim[0,0],lim[0,0]],[lim[1,0],lim[1,0],lim[1,1],lim[1,1],lim[1,0]],label='Limits')
    plt.legend()


    plt.figure()
    plt.title('Torroidal plane')

    stdp = np.copy(bes.beam.mesh)
    stdm = np.copy(bes.beam.mesh)
    perp = np.copy(bes.beam.direc)
    perp[2] = -np.sqrt(np.sum(bes.beam.direc[:2]**2))
    perp[0] = bes.beam.direc[2]
    stdp[:,2] = bes.beam.mesh[:,2]+perp[2]*bes.beam.stddev_v
    stdm[:,2] = bes.beam.mesh[:,2]-perp[2]*bes.beam.stddev_v
    stdp[:,0] = np.sqrt(np.sum(bes.beam.mesh[:,:2]**2,axis=1))
    stdm[:,0] = np.sqrt(np.sum(bes.beam.mesh[:,:2]**2,axis=1))

    plt.plot(np.sqrt(np.sum(bes.beam.mesh[:,:2]**2,axis=1)),bes.beam.mesh[:,2],'-x',label='Beam')
    plt.plot(stdp[:,0],stdp[:,2],stdm[:,0],stdm[:,2],label='Standard deviation')
    plt.plot(np.sqrt(np.sum(foc[:,:2]**2,axis=1)),foc[:,2],'x',label='Focus Points')
    plt.plot(np.sqrt(np.sum(bes.pos_lens[:2]**2)),bes.pos_lens[2],'x',label='lens')

    rlim = np.sqrt(np.sum(lim[:2,:]**2,axis=0))
    plt.plot([rlim[0],rlim[1],rlim[1],rlim[0],rlim[0]],[lim[2,0],lim[2,0],lim[2,1],lim[2,1],lim[2,0]],label='Limits')

    rad = np.linspace(0,2*np.pi,1000)
    x_tok = majorR+(majorR-minorR)*np.cos(rad)
    y_tok = (majorR-minorR)*np.sin(rad)


    plt.plot(x_tok,y_tok,label='Tokamak')
    plt.legend()
    
    plt.show()
    


def compute_beam_config(Rsource,phisource, Rtan,R=np.array([])):
    """ Help to compute the value to put inside the config file

    :param float Rsource: R-coordinate of the beam source
    :param float phisource: :math:`\phi`-coordinate of the beam source
    :param float Rtan: Radius where the beam will be tangent
    :param np.array[N] R: R-coordinate of the fibers (if given, will print the\
    phi value for being in the beam)
    """
    side = np.sqrt(Rsource**2-Rtan**2)
    alpha = np.arccos((Rsource**2 + Rtan**2 - side**2)/(2*Rsource*Rtan))
    phitan = phisource - alpha
    xtan = Rtan*np.cos(-phitan)
    ytan = Rtan*np.sin(-phitan)

    direc = np.array([xtan,ytan])
    possource = np.array([np.cos(-phisource),np.sin(-phisource)])
    possource = Rsource*possource

    direc = direc - possource
    direc = direc/np.sqrt(np.sum(direc**2))
    print 'The position of the beam is :', possource
    print 'The direction of the beam is :', direc
    if R.shape[0] != 0:
        phifoc = np.arccos(Rtan/R) + phitan
        print 'If you want the fibers on the central line of the beam, Phi = ',-phifoc



def interpolation_toroidal_plane(phi=-2.58,t=130,Nr=1000,Nz=1000,R=[1.82,2.3],Z=[-0.25,0.25]):
    """ Create a R-Z mesh and interpolate the data on it.
    Is usefull for checking if the data are well interpolated

    :param float phi: Toroidal plane to compute
    :param int t: Time step
    :param int Nr: Number of radial points
    :param int Nz: Number of Z points
    :param list[] R: R min and max
    :param list[] Z: Z min and max
    """
    
    bes = bes_.BES(name)
    
    bes.beam.t_ = t-1 # will be increase in compute_beam_on_mesh
    bes.beam.data.current = t
    bes.beam.data.load_next_time_step(increase=False)

    
    r = np.linspace(R[0],R[1],Nr)
    z = np.linspace(Z[0],Z[1],Nz)
    R,Z = np.meshgrid(r,z)
    R = R.flatten()
    Z = Z.flatten()
    
    x = R*np.cos(phi)
    y = R*np.sin(phi)
    
    ne = bes.beam.data.interpolate_data(np.array([x,y,Z]).T,t,['ne'],False,True)[0]
    ne = np.reshape(ne,(Nr,Nz))
    R = np.reshape(R,(Nr,Nz))
    Z = np.reshape(Z,(Nr,Nz))


    plt.figure()
    plt.title('Electron Density')
    plt.contourf(R,Z,ne)
    plt.xlabel('R-coordinate')
    plt.ylabel('Z-coordinate')
    plt.colorbar()

    plt.show()


def solid_angle_evolution(Rmax=2,Zmax=0.1,Nr=80,Nz=100,fib=4,v=40):
    """ Plot the value of the solid angle as a function of R-Z (optical system)
    (Every parameter are in the unit of the focus point radius/distance to the lens)
    """

    bes = bes_.BES(name)
    r = np.linspace(-Rmax,Rmax,Nr)*bes.rad_foc[fib]
    z = np.linspace(1-Zmax,1+Zmax,Nz)*bes.dist[fib]

    Z,R = np.meshgrid(z,r,indexing='ij')

    eps = np.zeros(R.shape)
    for i in range(Nz):
        pos = np.array([r,np.zeros(r.shape),z[i]*np.ones(r.shape)]).T
        eps[i,:] = bes.get_solid_angle(pos,fib)
        
    Z -= bes.dist[fib]

    plt.figure()
    #v = np.linspace(0,0.1,v)
    plt.contourf(Z,R,eps,v)
    plt.plot([0, 0],np.array([-1,1])*bes.rad_foc[fib],'-k')
    plt.xlabel('Distance from the central line')
    plt.ylabel('Distance from the focus point along the central line')
    plt.colorbar()
    plt.title('Solid Angle')

    plt.figure()
    plt.plot(R[0:Nz:10,:].T,eps[0:Nz:10,:].T)
    plt.xlabel('Distance from the central line')
    plt.ylabel('Solid Angle')

    
    plt.show()
