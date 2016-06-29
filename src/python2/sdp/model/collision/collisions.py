import numpy as np
import ADAS_file as adas
from scipy import interpolate
from sdp.settings.unitsystem import SI


class Collisions:
    r""" Class containing all the physics about the collisions
        
    Read the files from ADAS database, compute the lifetime, and the cross-sections (cubic spline interpolation is used).
    
    For computing the coefficients, two interpolations are done.
    A first one in 2D (beam energy and density) and a second one in temperature.
    The final result is given by:

    .. math::
       C = \frac{\text{Interp}(E_b,\rho)\cdot \text{Interp}(T)}{C_\text{ref}}

    where :math:`C_\text{ref}` is the coefficient at the reference temperature, density and beam energy.
    
    
    :param list[str] files_atte: List of names for ADAS21 files (beam stopping coefficient)
    :param list[str] files_emis: List of names for ADAS22 files (emission coefficient)
    :param list[int] states: Quantum number of the lower (states[0]) and the higher(states[1]) states of the hydrogen atom
    :param float lifetime: Lifetime of the excited state

    :var list[str] self.files_atte: List of names for ADAS21 files (beam stopping coefficient)
    :var list[str] self.files_emis: List of names for ADAS22 files (emission coefficient)
    :var list[] self.beam_atte: List of :class:`ADAS21 <sdp.plasma.collision.ADAS_file.ADAS21>` instance variable (beam stopping coefficient)
    :var list[] self.beam_emis: List of :class:`ADAS22 <sdp.plasma.collision.ADAS_file.ADAS22>` instance variable (emission coefficient)
    :var list[tck_interp] self.atte_tck_dens: List of interpolant computed with cubic spline for the beam stopping coefficient as a function of the density and the beam energy
    :var list[tck_interp] self.emis_tck_dens: List of interpolant computed with cubic spline for the emission coefficient as a function of the density and the beam energy
    :var list[tck_interp] self.atte_tck_temp: List of interpolant computed with cubic spline for the beam stopping coefficient as a function of the temperature
    :var list[tck_interp] self.emis_tck_temp: List of interpolant computed with cubic spline for the emission coefficient as a function of the temperature
    :var float self.n_low: Quantum number of the lower state for the hydrogen atom
    :var float self.n_high: Quantum number of the higher state for the hydrogen atom
    :var float self.E0: Energy of the ground state (in eV)    
    :var float self.lifetime: Lifetime of the excited state
    """

    def __init__(self,files_atte,files_emis,states,lifetime):
        """ Copy the input inside the instance
            
        :param list[str] files_atte: List of names for ADAS21 files (beam stopping coefficient)
        :param list[str] files_emis: List of names for ADAS22 files (emission coefficient)
        :param list[int] states: Quantum number of the lower (states[0]) and the higher(states[1]) states of the hydrogen atom
        """
        self.files_atte = files_atte                                         #!
        self.files_emis = files_emis                                         #!
        self.beam_emis = []                                                  #!
        self.beam_atte = []                                                  #!
        print 'The wavelength assumes an hydrogen atom'
        self.n_low = states[0]                                               #!
        self.n_high = states[1]                                              #!
        self.E0 = -13.6
        self.lifetime = lifetime
        self.read_adas()

        # compute the interpolant
        self.atte_tck_dens = []                                              #!
        self.emis_tck_dens = []                                              #!
        self.atte_tck_temp = []                                              #!
        self.emis_tck_temp = []                                              #!
        for i in range(len(self.beam_atte)):
            # get data
            ldensities = np.log(self.get_list_density('atte',i))
            lbeams = np.log(self.get_list_beams('atte',i))
            coef_dens = self.get_coef_density('atte',i)
            lbeams, ldens = np.meshgrid(lbeams, ldensities)
            
            # interpolation over beam and density
            self.atte_tck_dens.append(interpolate.bisplrep(
                lbeams,ldens,coef_dens,kx=1,ky=1))
            
            # get data for the interpolation in temperature
            T = np.log(self.get_list_temperature('atte',i))
            coef_T = self.get_coef_T('atte',i)
            Tref = np.log(self.get_Tref('atte',i))
            index = abs((Tref-T)/Tref) < 1e-4
            
            #interpolation over the temperature
            self.atte_tck_temp.append(interpolate.splrep(
                T,coef_T/coef_T[index],k=1))

        for i in range(len(self.beam_emis)):
            # get data
            ldensities = np.log(self.get_list_density('emis',i))
            lbeams = np.log(self.get_list_beams('emis',i))
            coef_dens = self.get_coef_density('emis',i)
            lbeams, ldens = np.meshgrid(lbeams, ldensities)
            
            # interpolation over beam and density
            self.emis_tck_dens.append(interpolate.bisplrep(
                lbeams,ldens,coef_dens,kx=1,ky=1))

            # Get data for the interpolation in temperature
            T = np.log(self.get_list_temperature('emis',i))
            coef_T = self.get_coef_T('emis',i)
            Tref = np.log(self.get_Tref('emis',i))
            index = abs((Tref-T)/Tref) < 1e-4
            
            #interpolation over the temperature
            self.emis_tck_temp.append(interpolate.splrep(
                T,coef_T/coef_T[index],k=1))

        
    def read_adas(self):
        """ Read the ADAS files and stores them as attributes (used during the initialization)
        """
        for name in self.files_atte:
            self.beam_atte.append(adas.ADAS21(name))
        for name in self.files_emis:
            self.beam_emis.append(adas.ADAS22(name))

    def get_attenutation(self,beam,ne,mass_b,Ti,file_number):
        """ Get the beam stopping coefficient for a given density, beam energy, and temperature.

        The ADAS database store the data as two array, for putting them together, we do a first
        interpolation for the 2D array (as a function of density and beam energy) and after 
        we do a scaling with the temperature.

        :param float beam: Beam energy (eV)
        :param float or np.array[N] ne: Electron density density
        :param float mass_b: mass of a neutral particle in the beam (amu)
        :param float or np.array[N] Ti: Ion temperature (should be of the same lenght than ne) 
        :param int file_number: File number wanted (choosen by beam.py)

        :returns: Beam stopping coefficient
        :rtype: np.array[ne.shape]
        """
        beam = np.log(beam/mass_b)
        ne = np.log(ne)
        Ti = np.log(Ti)
        if len(ne.shape) == 1:
            coef = np.zeros(ne.shape)
            for i,n in enumerate(ne):
                coef[i] = interpolate.bisplev(beam,n,self.atte_tck_dens[file_number])
        else:
            coef = interpolate.bisplev(beam,ne,self.atte_tck_dens[file_number])

        coef = coef * interpolate.splev(Ti,self.atte_tck_temp[file_number])
        return coef

    def get_emission(self,beam,ne,mass_b,Ti,file_number):
        """ Get the emission coefficient for a given density, beam energy, and temperature.

        The ADAS database store the data as two array, for putting them together, we do a first
        interpolation for the 2D array (as a function of density and beam energy) and after 
        we do a scaling with the temperature.

        :param float beam: Beam energy (eV)
        :param float or np.array[N] ne: Electron density density
        :param float mass_b: mass of a neutral particle in the beam (amu)
        :param float or np.array[N] Ti: Ion temperature (should be of the same lenght than ne) 
        :param int file_number: File number wanted (choosen by beam.py)

        :returns: Emission coefficient
        :rtype: np.array[ne.shape]
        """

        beam = np.log(beam/mass_b)
        ne = np.log(ne)
        Ti = np.log(Ti)
        if not isinstance(ne,float):
            coef = np.zeros(len(ne))
            for i in range(len(ne)):
                coef[i] = interpolate.bisplev(beam,ne[i],self.emis_tck_dens[file_number])
        else:
            coef =  interpolate.bisplev(beam,ne,self.emis_tck_dens[file_number])

        coef = coef * interpolate.splev(Ti,self.emis_tck_temp[file_number])
        return coef

    def get_Tref(self,typ,file_number):
        """ Return the reference temperature of the attenuation[beam stopping\
        coefficient]/emission file

        :param str typ: Choice of the type of file ('emis' or 'atte')
        :param int file_number: File number (choosen in beam.py)

        :returns: Reference temperature
        :rtype: float
        """
        if typ == 'emis':
            return self.beam_emis[file_number].T_ref
        elif typ == 'atte':
            return self.beam_atte[file_number].T_ref
        else:
            raise NameError('No list with this name: {0}'.format(typ))
    
    def get_coef_density(self,typ,file_number):
        """ Return the coefficient as a function of the density and the beam energy\
        of the attenuation[beam stopping coefficient]/emission file

        :param str typ: Choice of the type of file ('emis' or 'atte')
        :param int file_number: File number (choosen in beam.py)

        :returns: Coefficient as a function of the density and the beam energy
        :rtype: np.array[Ndens,Nbeam]
        """
        if typ == 'emis':
            return self.beam_emis[file_number].coef_dens
        elif typ == 'atte':
            return self.beam_atte[file_number].coef_dens
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    def get_coef_T(self,typ,file_number):
        """ Return the coefficient as a function of the temperature\
        of the attenuation[beam stopping coefficient]/emission file

        :param str typ: Choice of the type of file ('emis' or 'atte')
        :param int file_number: File number (choosen in beam.py)

        :returns: Coefficient as a function of the temperature
        :rtype: np.array[N]
        """
        if typ == 'emis':
            return self.beam_emis[file_number].coef_T
        elif typ == 'atte':
            return self.beam_atte[file_number].coef_T
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    
    def get_list_temperature(self,typ,file_number):
        """ Return the temperatures used in the ADAS file for\
        the attenuation[beam stopping coefficient]/emission file

        :param str typ: Choice of the type of file ('emis' or 'atte')
        :param int file_number: File number (choosen in beam.py)

        :returns: Temperatures computed in the ADAS file
        :rtype: np.array[N]
        """
        if typ == 'emis':
            return self.beam_emis[file_number].temperature
        elif typ == 'atte':
            return self.beam_atte[file_number].temperature
        else:
            raise NameError('No list with this name: {0}'.format(typ))
    
    def get_list_density(self,typ,file_number):
        """ Return the densities used in the ADAS file for\
        the attenuation[beam stopping coefficient]/emission file

        :param str typ: Choice of the type of file ('emis' or 'atte')
        :param int file_number: File number (choosen in beam.py)

        :returns: Densities computed in the ADAS file
        :rtype: np.array[N]
        """
        if typ == 'emis':
            return self.beam_emis[file_number].densities
        elif typ == 'atte':
            return self.beam_atte[file_number].densities
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    def get_list_beams(self,typ,file_number):
        """ Return the beam energies used in the ADAS file for\
        the attenuation[beam stopping coefficient]/emission file

        :param str typ: Choice of the type of file ('emis' or 'atte')
        :param int file_number: File number (choosen in beam.py)

        :returns: Beam energies computed in the ADAS file
        :rtype: np.array[N]

        """
        if typ == 'emis':
            # multiply by the mass due to ADAS
            return self.beam_emis[file_number].adas_beam
        elif typ == 'atte':
            return self.beam_atte[file_number].adas_beam
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    def get_lifetime(self,ne,Te,beam,mass_b,file_number):
        """ Compute the lifetime of the excited state.
        This version is using the lifetime given in the input but can be upgraded in order
        to have a plasma dependant lifetime.

        :todo: Upgrading in order to take into account the plasma density
        
        :param float beam: Beam energy (eV)
        :param np.array[N] ne: Electron density (m :sup:`-3`)
        :param float mass_b: Mass of a neutral particle in the beam (amu)
        :param np.array[N] Te: Electron temperature (eV)
        :param int file_number: File number (choosen in Beam.py)

        :returns: Lifetime of the excited atom
        :rtype: np.array[ne.shape]

        """

        return self.lifetime*np.ones(ne.shape)

    def get_wavelength(self):
        """ Compute the wavelength of the emitted photons in the particles 
        reference frame (assume an hydrogen atom).
        
        :returns: Wavelength emitted in reference frame (nm)
        :rtype: float
        """
        E = -self.E0*(1.0/self.n_low**2 - 1.0/self.n_high**2)
        return SI['hc']*1e12/(E*SI['keV'])


if __name__ == '__main__':
    'small test and example'
    import matplotlib.pyplot as plt

    # check the computation of the lifetime
    path = 'sdp/diagnostic/bes/'
    col = Collisions([path+'bms10#h_h1.dat'],[path+'bme10#h_h1.dat'],[2, 3])
    Ebeam = col.beam_atte[0].E_ref
    dens = col.beam_atte[0].densities
    T = col.beam_atte[0].temperature
    lifetime = np.zeros((T.shape[0],dens.shape[0]))
    for i,t in enumerate(T):
        lifetime[i,:] = col.get_lifetime(dens,t,t,Ebeam,1.0,0)
    plt.figure()
    plt.contourf(dens,T,lifetime)
    plt.colorbar()
    plt.show()


    'small check for the emission and the beam stopping coefficient'
    ne = np.linspace(np.min(dens),np.max(dens),1000)
    t_ref = T[4]
    at = col.get_attenutation(Ebeam,ne,1.0,t_ref,0)
    em = col.get_emission(Ebeam,ne,1.0,t_ref,0)
    if ((at < 0) | (em < 0)).any():
        print np.sum(at<0),at[at<0]
        print np.sum(em<0),em[em<0]
        plt.figure()
        plt.semilogx(ne,em<0,label='emission')
        plt.semilogx(ne,at<0,label='attenuation')
        plt.semilogx(dens,dens>0,'x')
        plt.legend()
        
        plt.figure()
        plt.semilogx(ne,em)
        plt.show()
        raise NameError('Error in the interpolation or in the ADAS file')
