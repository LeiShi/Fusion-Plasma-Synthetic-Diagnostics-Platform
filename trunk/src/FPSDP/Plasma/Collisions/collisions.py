import numpy as np
import ADAS_file as adas
from scipy import interpolate


class Collisions:
    """ Class containing all the physics about the collisions
        
    Read the files from ADAS database, compute the lifetime, and compute
    the cross-sections (cubic spline interpolation is used).
    

    :param [str] files_atte: List of names for ADAS21 files (beam stopping coefficient)
    :param [str] files_emis: List of names for ADAS22 files (emission coefficient)
    :param [int] states: Quantum number of the lower (states[0]) and the higher(states[1]) states of the hydrogen atom
    
    :ivar [str] files_atte: List of names for ADAS21 files (beam stopping coefficient)
    :ivar [str] files_emis: List of names for ADAS22 files (emission coefficient)
    :ivar [:class:`FPSDP.Plasma.Collisions.ADAS21`] beam_atte: List of ADAS file for the beam stopping coefficient
    :ivar [:class:`FPSDP.Plasma.Collisions.ADAS22`] beam_emis: List of ADAS file for the emission coefficient

    :ivar [tck_interp] atte_tck_dens: List of interpolant computed with bisplrep for\
the beam stopping coefficient as a function of the density and the beam energy
    :ivar [tck_interp] emis_tck_dens: List of interpolant computed with bisplrep for\
the emission coefficient as a function of the density and the beam energy
    :ivar [tck_interp] atte_tck_temp: List of interpolant computed with bisplrep for\
the beam stopping coefficient as a function of the temperature
    :ivar [tck_interp] emis_tck_temp: List of interpolant computed with bisplrep for\
the emission coefficient as a function of the temperature

    :ivar float n_low: Quantum number of the lower state for the hydrogen atom
    :ivar float n_high: Quantum number of the higher state for the hydrogen atom
    :ivar float E0: Energy of the ground state (in eV)    
        
    """

    def __init__(self,files_atte,files_emis,states):
        """ Copy the input inside the instance
            
            Arguments:
            files_atte  -- list containing the name of the attenuation files
            files_emis  -- list containing the name of the emission files
            beams       -- list of the different energy beam (should be in the 
                           same order than the files)
            lifetime    -- list of the lifetime of the excited states (same 
                           order than the other lists)
        """
        self.files_atte = files_atte                                         #!
        self.files_emis = files_emis                                         #!
        self.beam_emis = []                                                  #!
        self.beam_atte = []                                                  #!
        print 'The lifetime assume an hydrogen atom'
        self.n_low = states[0]                                               #!
        self.n_high = states[1]                                              #!
        self.E0 = -13.6
        self.read_adas()

        # compute the interpolant
        self.atte_tck_dens = []                                              #!
        self.emis_tck_dens = []                                              #!
        self.atte_tck_temp = []                                              #!
        self.emis_tck_temp = []                                              #!
        for i in range(len(self.beam_atte)):
            # get data
            ldensities = self.get_list_density('atte',i)
            lbeams = self.get_list_beams('atte',i)
            coef_dens = self.get_coef_density('atte',i)
            lbeams, ldens = np.meshgrid(lbeams, ldensities)
            
            # interpolation over beam and density
            self.atte_tck_dens.append(interpolate.bisplrep(lbeams,ldens,coef_dens))
            
            # get data for the interpolation in temperature
            T = self.get_list_temperature('atte',i)
            coef_T = self.get_coef_T('atte',i)
            Tref = self.get_Tref('atte',i)
            index = abs((Tref-T)/Tref) < 1e-4
            
            #interpolation over the temperature
            self.atte_tck_temp.append(interpolate.splrep(T,coef_T/coef_T[index]))

        for i in range(len(self.beam_emis)):
            # get data
            ldensities = self.get_list_density('emis',i)
            lbeams = self.get_list_beams('emis',i)
            coef_dens = self.get_coef_density('emis',i)
            lbeams, ldens = np.meshgrid(lbeams, ldensities)
            
            # interpolation over beam and density
            self.emis_tck_dens.append(interpolate.bisplrep(lbeams,ldens,coef_dens))

            # Get data for the interpolation in temperature
            T = self.get_list_temperature('emis',i)
            coef_T = self.get_coef_T('emis',i)
            Tref = self.get_Tref('emis',i)
            index = abs((Tref-T)/Tref) < 1e-4
            
            #interpolation over the temperature
            self.emis_tck_temp.append(interpolate.splrep(T,coef_T/coef_T[index]))

        
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
        """
        beam /= mass_b
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
        """

        beam /= mass_b
        if not isinstance(ne,float):
            coef = np.zeros(len(ne))
            for i in range(len(ne)):
                coef[i] = interpolate.bisplev(beam,ne[i],
                                              self.emis_tck_dens[file_number])
        else:
            coef =  interpolate.bisplev(beam,ne,self.emis_tck_dens[file_number])

        coef = coef * interpolate.splev(Ti,self.emis_tck_temp[file_number])
        return coef

    def get_Tref(self,typ,file_number):
        """ Return the reference temperature of the attenuation[beam stopping\
        coefficient]/emission file

        :param str typ: Choice of the type of file ('emis' or 'atte')
        :param int file_number: File number (choosen in beam.py)
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
        """
        if typ == 'emis':
            # multiply by the mass due to ADAS
            return self.beam_emis[file_number].adas_beam
        elif typ == 'atte':
            return self.beam_atte[file_number].adas_beam
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    def get_lifetime(self,ne,Te,Ti,beam,mass_b,file_number):
        """ Compute the lifetime of the excited state following the formula
        given by I. H. Hutchinson, Plasma Phys. Controlled Fusion 44,71 (2002)
        assuming an hydrogen atom.

        :param float beam: Beam energy
        :param np.array[N] ne: Electron density
        :param float mass_b: Mass of a neutral particle in the beam (amu)
        :param np.array[N] Te: Electron temperature
        :param np.array[N] Ti: Ion temperature
        :param int file_number: File number (choosen in Beam.py)
        """
        #:todo:
        l = np.ones(ne.shape)
        return 1.68e-9*l
        #frac = (float(self.n_low)/float(self.n_high))**2
        #E = -self.E0*(1.0/self.n_low**2 - 1.0/self.n_high**2)
        #sigv = frac*np.exp(E/Te)*self.get_emission(beam,ne,mass_b,Ti,file_number)
        #return 1.0/(sigv*ne)
