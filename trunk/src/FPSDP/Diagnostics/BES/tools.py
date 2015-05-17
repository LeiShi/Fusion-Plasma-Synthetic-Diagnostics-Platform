import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.fftpack import fft2, ifft2
import FPSDP.Diagnostics.Beam.beam as beam_
import FPSDP.Diagnostics.BES.bes as bes_

class Tools:
    """ Defines a few tools for doing some computations on the BES image

    """

    def __init__(self,I,R,Z,psin,I_fluc=False):
        if I_fluc == False:
            Iav = np.mean(I,axis=0)
            self.I = (I-Iav)/Iav
        else:
            self.I = I

        self.R = R
        self.Z = Z
        self.psin = psin
        self.Nfoc = Z.shape[0]
        self.Nt = I.shape[0]

    @classmethod
    def init_from_file(cls,filename):
        """ Load of file in the numpy format (npz)

        The data inside it should be in the following order: [I,psin,pos_foc,...]
        """
        data = np.load(filename)
        I = data['arr_0']
        psin = data['arr_1']
        pos = data['arr_2']
        R = np.sqrt(np.sum(pos[:,0:2]**2,axis=1))
        Z = pos[:,2]
        return cls(I,R,Z,psin)

    def interpolate(self,Nr,Nz,I):
        """ Interpolate all the data on a mesh
        """
        if not hasattr(self,'r'):
            self.r = np.linspace(np.min(self.R),np.max(self.R),Nr)
            self.z = np.linspace(np.min(self.Z),np.max(self.Z),Nz)
        else:
            if (self.r.shape[0] != Nr) or (self.z.shape[0] != Nz):
                raise NameError('Not the same size of mesh')

        Igrid = np.zeros((self.Nt,Nr,Nz))
        for i in range(self.Nt):
            temp = interp2d(self.R,self.Z,I[i,:])
            Igrid[i,:,:] = temp(self.r,self.z).T
            
        return Igrid
            
            
    def computePSF(self, ne, Nr=20, Nz=30, ne_fluc=False):
        """ Compute the PSF
        """
        if isinstance(ne,str):
            data = np.load(ne)
            ne = data['arr_0']
        if ne_fluc == False:
            neav = np.mean(ne,axis=0)
            ne = (ne-neav)/neav

        negrid = self.interpolate(Nr,Nz,ne)
        Igrid = self.interpolate(Nr,Nz,self.I)
        
        self.psf = np.zeros((Nr,Nz))

        for i in range(self.Nt):
            ne_tilde = fft2(negrid[i,...])
            I_tilde = fft2(Igrid[i,...])
            I_tilde /= ne_tilde
            self.psf += np.real(ifft2(I_tilde))

        self.psf /= self.Nt

        plt.figure()
        plt.contourf(self.r,self.z,self.psf.T)
        plt.xlabel('R[m]')
        plt.ylabel('Z[m]')
        plt.colorbar()
        plt.show()

        
""" Define a few test for checking the data given by the code
It contains all the code used for the figure in my report
"""
name = 'FPSDP/Diagnostics/BES/bes.in'
t = 150


def beam_density():
    """ Compare the beam density of the equilibrium case and of the equilibrium+fluctuations case

    """
    bes = bes_.BES(name)
    bes.beam.eq = True
    bes.beam.t_ = t
    bes.beam.data.current = t
    bes.beam.compute_beam_on_mesh()
    nb_eq = bes.beam.density_beam
    ne_eq = bes.beam.get_quantities(bes.beam.mesh,self.t_,['ne'],True,check=False)[0]
    bes.beam.eq = False
    bes.beam.compute_beam_on_mesh()
    nb_fl = bes.beam.density_beam
    ne_fl = bes.beam.get_quantities(bes.beam.mesh,self.t_,['ne'],False,check=False)[0]
    dl = np.sqrt(np.sum((bes.beam.mesh-bes.beam.pos[np.newaxis,:])**2,axis=1))
    
    
    fig, axarr = plt.subplots(2,sharex=True)
        
    axarr[1].plot(dl,((nb_eq-nb_fl)/nb_eq).T)
    plt.xlabel('Distance [m]')
    axarr[1].set_ylabel('Error')
    axarr[1].legend(['1st component','2nd component','3rd component'],loc=4)
    axarr[1].grid(True)
    
    axarr[0].plot(dl,nb_eq.T)
    axarr[0].plot(dl,nb_fl.T)
    axarr[0].grid(True)
    axarr[0].set_ylabel('Beam density [m$^{-3}$]')
    axarr[0].legend(['1st Eq','2nd Eq','3rd Eq'])
    
    
    fig, axarr = plt.subplots(2,sharex=True)
        
    axarr[1].plot(dl,((ne_eq-ne_fl)/ne_eq))
    plt.xlabel('Distance [m]')
    axarr[1].set_ylabel('Error')
    axarr[1].grid(True)
    
    axarr[0].plot(dl,ne_eq)
    axarr[0].plot(dl,ne_fl)
    axarr[0].grid(True)
    axarr[0].legend(['Equilibrium','Fluctuations'],loc=4)
    axarr[0].set_ylabel('Electron Density [m$^{-3}$]')
        
    plt.figure()
    tot = np.sum(nb_eq,axis=0)
    plt.plot(dl,(nb_eq/tot[np.newaxis,:]).T)
    tot = np.sum(nb_fl,axis=0)
    plt.plot(dl,(nb_fl/tot[np.newaxis,:]).T)
    plt.legend(['1st Eq','2nd Eq','3rd Eq'],loc=2)
    plt.grid(True)
    
    plt.show()

def beam_emission():
    """ Shows the effect of the beam density on the emission and the effect of the lifetime.
    """
    bes = bes_.BES(name)
    bes.beam.t_ = t
    bes.beam.data.current = t
    bes.beam.compute_beam_on_mesh()

    r_max = 0.5
    dl = np.sqrt(np.sum((bes.beam.mesh-bes.beam.pos[np.newaxis,:])**2,axis=1))
    r = np.linspace(-r_max,r_max,20)
    R,L = plt.meshgrid(r,dl)
    R = R.flatten()
    L = L.flatten()
    perp = bes.beam.direc
    perp[0] = perp[1]
    perp[1] = -bes.beam.direc[0]
    pos = bes.beam.pos[np.newaxis,:] + L[:,np.newaxis]*bes.beam.direc + R[:,np.newaxis]*perp

    emis = bes.beam.get_emis(pos,t)
    emis_l = bes.beam.get_emis_lifetime(pos,t)

    
    plt.figure()
    plt.title('Photon radiance with an instantaneous emission')
    plt.contourf(R,L,emis)
    plt.colorbar()
    plt.xlabel('Distance from the central line [m]')
    plt.ylabel('Distance from the source [m]')

    plt.figure()
    plt.title('Photon radiance')
    plt.contourf(R,L,emis_l)
    plt.colorbar()
    plt.xlabel('Distance from the central line [m]')
    plt.ylabel('Distance from the source [m]')

    plt.figure()
    plt.title('Difference of emission between the instantaneous and non-instantaneous emission')
    plt.contourf(R,L,emis_l-emis)
    plt.colorbar()
    plt.xlabel('Distance from the central line [m]')
    plt.ylabel('Distance from the source [m]')

    plt.show()


