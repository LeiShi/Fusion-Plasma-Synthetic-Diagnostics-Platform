""" class and subclass reading different kind of adas file
"""

import numpy as np
import scipy.interpolate as ip


class ADAS_file:
    """ Parent class for defining the different kind of database readers.

    This class is for inheritence purpose only. It will be inherited
    by all the ADAS readers.
    It defines how to read a block 
    of data (:func:`read_block`) [often used in 2D data of the ADAS database].
    The beam energy is divided by the atomic mass of the beam particles (eV/amu).

    
    :param str name: Name of the ADAS file

    :ivar str self.name: Name of the ADAS file
    
    """
    def __init__(self,name):
        """ Save the name of the file
        
        :param name: Name of the ADAS file
        :type name: str

        :ivar str name: Name of the ADAS file
        """
        
        self.name = name

    def read_block(self,data,i,array,n):
        """ Read one bloc in an ADAS file

        The coefficient depending on two coefficients are written in a block form,
        thus this function read the block at the line i and return the 
        number of the final line
        

        :param data: file currently readed (each index is for a line)
        :type data: list[str]
        :param i: first line to look at (index of data)
        :type i: int
        :param array: array where to add the data from the file (should be of the 
        good size)
        :type array: np.array
        :param n: Number of item contained inside the data
        :type n: int

        :returns: index of the index of the final line
        :rtype: int
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

    The data contained in this kind of file is the beam stopping coefficient.
    The beam energy is divided by the atomic mass of the beam particles (eV/amu).

    :param name: Name of the ADAS file
    :type name: str
    

    :ivar int n_b: Size of the beam energy dimension
    :ivar int n_density: Size of the density dimension
    :ivar float T_ref: Reference temperature (eV)
    :ivar np.array[n_b] adas_beam: Beam energies considered (eV/amu)
    :ivar np.array[n_density] densities: Densities considered (m :sup:`-3`)
    :ivar np.array[n_density,n_b] coef_dens: Beam stopping coefficient as a function\
 of the density and the beam energy (m :sup:`3`/s)
    :ivar int n_T: Size of the temperature dimension
    :ivar float E_ref: Reference beam energy (eV/amu)
    :ivar float dens_ref: Reference density (m :sup:`-3`)
    :ivar np.array[n_T] temperature: Temperatures considered (eV)
    :ivar np.array[n_T] coef_T: Beam stopping coefficient as a function of the temperature (m :sup:`3`/s)
    """
    def __init__(self,name):
        """ Read the file and store all the values.
        
        :param name: Name of the ADAS file
        :type name: str

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
    """ Class containing all the data from one ADAS file (adf22)

    The data contained in this kind of file is the emission coefficient.
    The beam energy is divided by the atomic mass of the beam particles (eV/amu).

    :param name: Name of the ADAS file
    :type name: str
    

    :ivar int n_b: Size of the beam energy dimension
    :ivar int n_density: Size of the density dimension
    :ivar float T_ref: Reference temperature (eV)
    :ivar np.array[n_b] adas_beam: Beam energies considered (eV/amu)
    :ivar np.array[n_density] densities: Densities considered (m :sup:`-3`)
    :ivar np.array[n_density,n_b] coef_dens: Emission coefficient as a function\
 of the density and the beam energy (m :sup:`3`/s)
    :ivar int n_T: Size of the temperature dimension
    :ivar float E_ref: Reference beam energy (eV/amu)
    :ivar float dens_ref: Reference density (m :sup:`-3`)
    :ivar np.array[n_T] temperature: Temperatures considered (eV)
    :ivar np.array[n_T] coef_T: Emission coefficient as a function of the temperature (m :sup:`3`/s)
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
    
