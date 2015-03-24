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
        # get data
        ldensities = self.get_list_density('atte',file_number)
        lbeams = self.get_list_beams(mass_b,'atte',file_number)
        coef_dens = self.get_coef_density('atte',file_number)
        lbeams, ldens = np.meshgrid(lbeams, ldensities)
        
        # interpolation over beam and density
        tck = interpolate.bisplrep(lbeams,ldens,coef_dens)
        if len(ne.shape) == 1:
            coef = np.zeros(ne.shape)
            for i,n in enumerate(ne):
                coef[i] = interpolate.bisplev(beam,n,tck)
        else:
            coef = interpolate.bisplev(beam,ne,tck)
        # get data for the interpolation in temperature
        T = self.get_list_temperature('atte',file_number)
        coef_T = self.get_coef_T('atte',file_number)
        Tref = self.get_Tref('atte',file_number)
        index = np.where(abs((Tref-T)/Tref) < 1e-4)[0][0]

        #interpolation over the temperature
        tck = interpolate.splrep(T,coef_T/coef_T[index])
        coef = coef * interpolate.splev(Ti,tck)
        if (coef <= 0).any():
            print coef
            raise NameError('Attenuation coefficient smaller than 0')
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
        # get data
        ldensities = self.get_list_density('emis',file_number)
        lbeams = self.get_list_beams(mass_b,'emis',file_number)
        coef_dens = self.get_coef_density('emis',file_number)
        lbeams, ldens = np.meshgrid(lbeams, ldensities)
        
        # interpolation over beam and density
        tck = interpolate.bisplrep(lbeams,ldens,coef_dens)
        if not isinstance(ne,float):
            be = beam*np.ones(len(ne))
            coef = np.zeros(len(be))
            for i in range(len(be)):
                coef[i] = interpolate.bisplev(be[i],ne[i],tck)
                if coef[i] < 0:
                    coef[i] = 0
        else:
            be = beam
            coef =  interpolate.bisplev(be,ne,tck)
            if coef < 0:
                coef = 0

        # Get data for the interpolation in temperature
        T = self.get_list_temperature('emis',file_number)
        coef_T = self.get_coef_T('emis',file_number)
        Tref = self.get_Tref('emis',file_number)
        index = np.where(abs((Tref-T)/Tref) < 1e-4)[0][0]

        #interpolation over the temperature
        tck = interpolate.splrep(T,coef_T/coef_T[index])
        coef = coef * interpolate.splev(Ti,tck)
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

    def get_list_beams(self,mass_b,typ,file_number):
        """ return the list of beams given by ADAS
            multiply by the mass of the beam atoms due to ADAS
            Argument:
            typ         --  'emis' or 'atte' (specify in which list)
            file_number --  file number in the list
        """
        if typ == 'emis':
            # multiply by the mass due to ADAS
            return mass_b*self.beam_emis[file_number].adas_beam
        elif typ == 'atte':
            return mass_b*self.beam_atte[file_number].adas_beam
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
        return 1.68e-9
        #frac = (float(self.n_low)/float(self.n_high))**2
        #E = -self.E0*(1.0/self.n_low**2 - 1.0/self.n_high**2)
        #sigv = frac*np.exp(E/Te)*self.get_emission(beam,ne,mass_b,Ti,file_number)
        #return 1.0/(sigv*ne)
