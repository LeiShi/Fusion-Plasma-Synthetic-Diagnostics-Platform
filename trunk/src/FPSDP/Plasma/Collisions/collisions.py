import numpy as np
import ADAS_file as adas
from scipy import interpolate


class Collisions:
    """ Class containing all the physics about the collisions
        Read the files from ADAS database, compute the lifetiime, and compute
        the cross-sections (cubic spline interpolation is used)

        Methods:

          read_adas()     --  load the data from the ADAS file
          get_attenutation(beam,ne,Ti,file_number)
                          --  gives the attenuation coefficient
          get_emission(beam,ne,mass_b,Ti,file_number):
                          --  gives <sigma*v> for the exitation
          get_lifetime(ne,Te,Ti,beam,mass_b,file_number):
                          --  gives the lifetime of the excited state
    
          Access data in ADAS file:
            get_Tref(file_number)             --  reference temperature
            get_coef_density(file_number)     --  first table
            get_coef_T(file_number)           --  second table  
            get_list_temperature(file_number) --  list temperatures
            get_list_density(file_number)     --  list densities
            get_list_beams(file_number)       --  list beams

        Attributes:
    
        files_atte -- list of ADAS files for the attenuation
        files_emis -- list of ADAS files for the emission
        beam_data  -- list of the beams
        n_low      -- quantum number of the lower state
        n_high     -- quantum number of the higher state
        
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
        """ Read the ADAS files and stores them as attributes
        """
        for name in self.files_atte:
            self.beam_atte.append(adas.ADAS21(name))
        for name in self.files_emis:
            self.beam_emis.append(adas.ADAS22(name))

    def get_attenutation(self,beam,ne,mass_b,Ti,file_number):
        """ get the attenuation value for a given density, beam energy, and
            temperature

            Arguments:
            beam        -- beam energy wanted
            ne          -- electron density (in plasma) density wanted
            mass_b      -- mass of an atom (of the beam in amu)
            Ti          -- ion temperature 
            file_number -- file number wanted (should be simplify in Beam.py)
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
        """ get the emission value for a given density, beam energy, and
            temperature

            Arguments:
            beam        -- beam energy wanted
            ne          -- electron (in plasma) density wanted
            mass_b      -- mass of an atom (of the beam in amu)
            Ti          -- ion temperature
            file_number -- file number wanted (should be simplify in Beam.py)
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
        """ Return the reference temperature for coef_density
            Argument:
            typ         --  'emis' or 'atte' (specify in which list)
            file_number --  file number in the list
        """
        if typ == 'emis':
            return self.beam_emis[file_number].T_ref
        elif typ == 'atte':
            return self.beam_atte[file_number].T_ref
        else:
            raise NameError('No list with this name: {0}'.format(typ))
    
    def get_coef_density(self,typ,file_number):
        """ return the list of attenuation coefficient given by ADAS
            as a function of the energy beam and the density
            Argument:
            typ         --  'emis' or 'atte' (specify in which list)
            file_number --  file number in the list
        """
        if typ == 'emis':
            return self.beam_emis[file_number].coef_dens
        elif typ == 'atte':
            return self.beam_atte[file_number].coef_dens
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    def get_coef_T(self,typ,file_number):
        """ return the list of attenuation coefficient given by ADAS
            as a function of the temperature
            Argument:
            typ         --  'emis' or 'atte' (specify in which list)
            file_number --  file number in the list
        """
        if typ == 'emis':
            return self.beam_emis[file_number].coef_T
        elif typ == 'atte':
            return self.beam_atte[file_number].coef_T
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    
    def get_list_temperature(self,typ,file_number):
        """ return the list of temperature given by ADAS
            Argument:
            typ         --  'emis' or 'atte' (specify in which list)
            file_number --  file number in the list
        """
        if typ == 'emis':
            return self.beam_emis[file_number].temperature
        elif typ == 'atte':
            return self.beam_atte[file_number].temperature
        else:
            raise NameError('No list with this name: {0}'.format(typ))
    
    def get_list_density(self,typ,file_number):
        """ return the list of densities given by ADAS
            Argument:
            typ         --  'emis' or 'atte' (specify in which list)
            file_number --  file number in the list
        """
        if typ == 'emis':
            return self.beam_emis[file_number].densities
        elif typ == 'atte':
            return self.beam_atte[file_number].densities
        else:
            raise NameError('No list with this name: {0}'.format(typ))

    def get_list_beams(self,typ,file_number):
        """ return the list of beams given by ADAS
            multiply by the mass of the beam atoms due to ADAS
            Argument:
            typ         --  'emis' or 'atte' (specify in which list)
            file_number --  file number in the list

            WARNING THE BEAM ENERGY IS BY ATOMIC MASS UNIT
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
            assuming an hydrogen atom

            Arguments:
            beam        -- beam energy wanted
            ne          -- electron (in plasma) density wanted
            mass_b      -- mass of an atom (of the beam in amu)
            Te          -- electron temperature
            Ti          -- ion temperature
            file_number -- file number wanted (should be simplify in Beam.py)
        """
        l = np.ones(ne.shape)
        return 1.68e-9*l
        #frac = (float(self.n_low)/float(self.n_high))**2
        #E = -self.E0*(1.0/self.n_low**2 - 1.0/self.n_high**2)
        #sigv = frac*np.exp(E/Te)*self.get_emission(beam,ne,mass_b,Ti,file_number)
        #return 1.0/(sigv*ne)
