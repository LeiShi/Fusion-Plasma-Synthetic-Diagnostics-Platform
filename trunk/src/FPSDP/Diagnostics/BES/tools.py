import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.fftpack import fft2, ifft2

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
