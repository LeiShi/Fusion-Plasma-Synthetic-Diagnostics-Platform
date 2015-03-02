import numpy as np
import ADAS_file as adas
from scipy import interpolate


class Collisions:
    """ Class containing all the physics about the collisions
        Read the files from ADAS database, compute the lifetiime, and compute
        the cross-sections (cubic spline interpolation is used)

        Methods:

          read_adas()     --  load the data from the ADAS file
          get_attenutation(beam,density,temperature,file_number)
                          --  gives the attenuation coefficient
          
    
          Access data in ADAS file:
            get_Tref(file_number)             --  reference temperature
            get_coef_density(file_number)     --  first table
            get_coef_T(file_number)           --  second table  
            get_list_temperature(file_number) --  list temperatures
            get_list_density(file_number)     --  list densities
            get_list_beams(self,file_number)  --  list beams

        Attributes:
    
        files      -- list of ADAS files
        lifetimes  -- list with the lifetime of the excited states
        beam_data  -- list of the beams
        
      
    """

    def __init__(self,files,lifetimes):
        """ Copy the input inside the instance
            
            Arguments:
            files       -- list containing the name of all files
            beams       -- list of the different energy beam (should be in the 
                           same order than the files)
            lifetime    -- list of the lifetime of the excited states (same 
                           order than the other lists)
        """
        self.files = files                                                   #!
        self.lifetimes = lifetimes                                           #!
        self.beam_data = []                                                  #!
        self.read_adas()
        
    def read_adas(self):
        """ Read the ADAS files and stores them as attributes
        """
        for name in self.files:
            self.beam_data.append(adas.Beam_ADAS_File(name))

    def get_attenutation(self,beam,density,temperature,file_number):
        """ get the attenuation value for a given density, beam energy, and
            temperature

            Arguments:
            beam        -- beam energy wanted
            density     -- ion (in plasma) density wanted
            temperature -- temperature wanted (default = -1)
                           if value == -1, the temperature of the ADAS file
                           is used
            file_number -- file number wanted (should be simplify in Beam.py)
        """
        # get data
        ldensities = self.get_list_density(file_number)
        lbeams = self.get_list_beams(file_number)
        coef_dens = self.get_coef_density(file_number)
        lbeams, ldens = np.meshgrid(lbeams, ldensities)
        
        # interpolation over beam and density
        tck = interpolate.bisplrep(lbeams,ldens,coef_dens)
        coef = interpolate.bisplev(beam,density,tck)

        # get data for the interpolation in temperature
        T = self.get_list_temperature(file_number)
        coef_T = self.get_coef_T(file_number)
        Tref = self.get_Tref(file_number)
        index = np.where(abs((Tref-T)/Tref) < 1e-4)[0][0]

        #interpolation over the temperature
        tck = interpolate.splrep(T,coef_T/coef_T[index])
        coef = coef * interpolate.splev(temperature,tck)
        if coef <= 0:
            raise NameError('Attenuation coefficient smaller than 0')
        return coef

    def get_Tref(self,file_number):
        """ Return the reference temperature for coef_density """
        return self.beam_data[file_number].T_ref
    
    def get_coef_density(self,file_number):
        """ return the list of attenuation coefficient given by ADAS
            as a function of the energy beam and the density
        """
        return self.beam_data[file_number].coef_dens

    def get_coef_T(self,file_number):
        """ return the list of attenuation coefficient given by ADAS
            as a function of the temperature
        """
        return self.beam_data[file_number].coef_T

    
    def get_list_temperature(self,file_number):
        """ return the list of temperature given by ADAS"""
        return self.beam_data[file_number].temperature
    
    def get_list_density(self,file_number):
        """ return the list of densities given by ADAS"""
        return self.beam_data[file_number].densities

    def get_list_beams(self,file_number):
        """ return the list of beams given by ADAS"""
        return self.beam_data[file_number].adas_beam
