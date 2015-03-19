""" class and subclass reading different kind of adas file
"""

import numpy as np
import scipy.interpolate as ip


class ADAS_file:
    """ Global class for defining the different kind of database
        DO NOT USE
    """
    def __init__(self,name):
        self.name = name

    def read_block(self,data,i,array,n):
        """ Read a block of the ADAS file (Beam stopping rate) and return the 
        number of the final line
        
        Arguments:
        data -- list of string containing the file (see read_adas for an 
                example)
        i    -- first line to look at
        list -- list where to add the data
        n    -- number of item contains in data
        """
        # loop over all the data block
        index = 0
        while index is not n:
            temp = data[i].split()
            # loop over one line for taking all the datas
            # inside this line
            for j in range(len(temp)):
                array[index] = temp[j]
                index += 1
            i += 1
        return i



class ADAS21(ADAS_file):
    """ Class containing all the data from one ADAS file (adf21)
        Use a cubic spline for the interpolation
        Is used for the collisions

       Attributs:
    
       n_b         -- number of beam energies in the first table
       n_density   -- number of densities in the first table
       T_ref       -- reference temperature of the first table
       adas_beam   -- beam energies of the first table (1D array)
       densities   -- densities of the first table (1D array)
       coef_dens   -- first table (2D array)
       n_T         -- number of temperature for the second table
       E_ref       -- reference beam energy for the second table
       dens_ref    -- reference density for the second table
       temperature -- temperatures for the second table (1D array)
       coef_T      -- second table (1D array)
    """
    def __init__(self,name):
        """ Read the file and store everything as attributes
        
            Arguments:
            name  -- name of the file
        """
        ADAS_file.__init__(self,name)
        # open the file and store all the text in data
        f = open(self.name,'r')
        data = f.read().split('\n')
        
        temp = data[2].split()
        # number of different beams in the ADAS file
        self.n_b = int(temp[0])           #! (this sign indicates an attribute)
        # change of unit
        # number of different densities for the target
        self.n_density = int(temp[1])                                   #!
        # reference temperature
        self.T_ref = float(temp[2].split('=')[1])                       #!
        
        # list of all beams computed by ADAS
        self.adas_beam = np.zeros(self.n_b)                             #!
        # line number in the ADAS file
        i = 4
        # read all the beam energies taken in account in the
        # ADAS file
        i = self.read_block(data,i,self.adas_beam,self.n_b)

        # same as before but with the densities
        self.densities = np.zeros(self.n_density)                       #!
        i = self.read_block(data,i,self.densities,self.n_density)
        # change of unit cm-3 -> m-3
        self.densities *= 100.0**3
        i += 1 # remove line with ----
        
        # contains the coefficients as a function of densities and the beam
        # energies
        self.coef_dens = np.zeros((self.n_density,self.n_b))
        # coef_dens[i,j] -- i for beam, j for densities                #!

        for j in range(self.n_density):
            i = self.read_block(data,i,self.coef_dens[j],self.n_b)
            
            
        i += 1 # remove line with ----
        # change of unit cm -> m
        self.coef_dens /= 100.0**3
        
        temp = data[i].split()
        self.n_T = int(temp[0]) # number of different temperature       #!
        # reference energy
        self.E_ref = float(temp[1].split('=')[1])                       #!
        # reference density
        self.dens_ref = float(temp[2].split('=')[1])*100**3             #!
        
        i += 2 # goes to next line, and remove line with ----
        
        # list of temperature
        self.temperature = np.zeros(self.n_T)                           #!
        i = self.read_block(data,i,self.temperature,self.n_T)

        i += 1 # remove line with ----
        
        # read the coefficients as a function of the temperature
        self.coef_T = np.zeros(self.n_T)                                #!
        i = self.read_block(data,i,self.coef_T,self.n_T)

        # change of unit
        self.coef_T /= 100.0**3
        # END OF READING


class ADAS22(ADAS_file):
    """ Class containing all the data from one ADAS file (adf22))
        Use a cubic spline for the interpolation
        Is used for the collisions

       Attributs:
    
       n_b         -- number of beam energies in the first table
       n_density   -- number of densities in the first table
       T_ref       -- reference temperature of the first table
       adas_beam   -- beam energies of the first table (1D array)
       densities   -- densities of the first table (1D array)
       coef_dens   -- first table (2D array) [the choice of the name
                      coef is for having the same naming convention
                      in all the ADAS file, the user should know the
                      class of the object]]
       n_T         -- number of temperature for the second table
       E_ref       -- reference beam energy for the second table
       dens_ref    -- reference density for the second table
       temperature -- temperatures for the second table (1D array)
       coef_T      -- second table (1D array)
    """
    def __init__(self,name):
        """ Read the file and store everything as attributes
        
            Arguments:
            name  -- name of the file
        """
        ADAS_file.__init__(self,name)
        # open the file and store all the text in data
        f = open(self.name,'r')
        data = f.read().split('\n')
        
        temp = data[2].split()
        # number of different beams in the ADAS file
        self.n_b = int(temp[0])           #! (this sign indicates an attribute)
        # change of unit
        # number of different densities for the target
        self.n_density = int(temp[1])                                   #!
        # reference temperature
        self.T_ref = float(temp[2].split('=')[1])                       #!
        
        # list of all beams computed by ADAS
        self.adas_beam = np.zeros(self.n_b)                             #!
        # line number in the ADAS file
        i = 4
        # read all the beam energies taken in account in the
        # ADAS file
        i = self.read_block(data,i,self.adas_beam,self.n_b)

        # same as before but with the densities
        self.densities = np.zeros(self.n_density)                       #!
        i = self.read_block(data,i,self.densities,self.n_density)
        # change of unit cm-3 -> m-3
        self.densities *= 100.0**3
        i += 1 # remove line with ----
        
        # contains the coefficients as a function of densities and the beam
        # energies
        self.coef_dens = np.zeros((self.n_density,self.n_b))
        # coef_dens[i,j] -- i for beam, j for densities                #!

        for j in range(self.n_density):
            i = self.read_block(data,i,self.coef_dens[j],self.n_b)
            
            
        i += 1 # remove line with ----
        # change of unit cm -> m
        self.coef_dens /= 100.0**3
        
        temp = data[i].split()
        self.n_T = int(temp[0]) # number of different temperature       #!
        # reference energy
        self.E_ref = float(temp[1].split('=')[1])                       #!
        # reference density
        self.dens_ref = float(temp[2].split('=')[1])*100**3             #!
        
        i += 2 # goes to next line, and remove line with ----
        
        # list of temperature
        self.temperature = np.zeros(self.n_T)                           #!
        i = self.read_block(data,i,self.temperature,self.n_T)

        i += 1 # remove line with ----
        
        # read the coefficients as a function of the temperature
        self.coef_T = np.zeros(self.n_T)                                #!
        i = self.read_block(data,i,self.coef_T,self.n_T)

        # change of unit
        self.coef_T /= 100.0**3
        # END OF READING
    
