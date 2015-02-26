import scipy.interpolate as ip


def read_block(data,i,list,n):
    """ Read a block of the ADAS file and return the number of the final line

        Arguments:
        data -- list of string containing the file (see read_adas for an 
                example)
        i    -- first line to look at
        list -- list where to add the data
        n    -- number of item contains in data
    """
    # loop over all the data block 
    while len(list) is not n:
        temp = data[i].split()
        # loop over one line for taking all the datas
        # inside this line
        for j in range(len(temp)):
            list.append(float(temp[j]))
        i += 1
    return i



class Beam_ADAS_File:
    """ Class containing all the data from one ADAS file
        Use a cubic spline for the interpolation
        Is used for the collisions
    """
    def __init__(self,name, beams = []):
        """ Read the file and store everything as attributes
        
            Arguments:
            name  -- name of the file
            beams -- list of the wanted beams (default keep all the file)
        """
        
        self.beams = beams                                              #!
        # open the file and store all the text in data
        f = open(name,'r')
        data = f.read().split('\n')
        
        temp = data[2].split()
        # number of different beams in the ADAS file
        self.n_b = int(temp[0])           #! (this sign indicates an attribute)
        # number of different densities for the target
        self.n_rho = int(temp[1])                                       #!
        # reference temperature
        self.T_ref = float(temp[2].split('=')[1])                       #!
        
        # list of all beams computed by ADAS
        self.adas_beam = []                                             #!
        # line number in the ADAS file
        i = 4
        # read all the beam energies taken in account in the
        # ADAS file
        i = read_block(data,i,self.adas_beam,self.n_b)
        
        # same as before but with the densities
        self.densities = []                                             #!
        i = read_block(data,i,self.densities,self.n_rho)
        i += 1 # remove line with ----
        
        # contains the coefficients as a function of densities and the beam
        # energies
        self.coef_rho = [] # coeff_rho[i,j] -- i for beam, j for densities
                                                                        #!
        for j in range(self.n_b):
            self.coef_rho.append([])
            i = read_block(data,i,self.coef_rho[j],self.n_rho)
            
            
        i += 1 # remove line with ----
            
        temp = data[i].split()
        self.n_T = int(temp[0]) # number of different temperature       #!
        # reference energy
        self.E_ref = float(temp[1].split('=')[1])                       #!
        # reference density
        self.rho_ref = float(temp[2].split('=')[1])                     #!
        
        i += 2 # goes to next line, and remove line with ----
        
        # list of temperature
        self.temperature = []                                           #!
        i = read_block(data,i,self.temperature,self.n_T)

        i += 1 # remove line with ----
        
        # read the coefficients as a function of the temperature
        self.coef_T = []                                               #!
        i = read_block(data,i,self.coef_T,self.n_T)

        self.keep_only_wanted()
        # END OF READING

    def keep_only_wanted(self):
        # list of all coeff [i,j]: i is for the beam, j for the density
        self.coef = []                                                  #!
        for i in range(len(self.beams)):
            self.coef.append([])
            for j in range(len(self.densities)):
                tck = ip.splrep(self.adas_beam,self.coef_rho[j], s=0)
                self.coef[i].append(float(ip.splev(self.beams[i],tck, der=0)))
