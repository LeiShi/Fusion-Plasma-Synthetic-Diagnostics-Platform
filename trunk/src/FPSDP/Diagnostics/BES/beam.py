import json
import numpy as np
import collisions as col
import ConfigParser as psr


class Beam1D:
    """ Simulate a 1D beam with the help of datas from simulation
    """
    
    def __init__(self,config_file,timesteps,data):
        """ Load everything from the config file"""
        self.cfg_file = config_file                                          #!
        config = psr.ConfigParser()
        config.read(self.cfg_file)

        # load data for collisions
        adas_files = json.loads(config.get('Collisions','adas_file'))
        lifetimes = json.loads(config.get('Collisions','lifetimes'))
        self.collisions = col.Collisions(adas_files,lifetimes)               #!
        self.list_col = json.loads(config.get('Collisions','collisions'))

        # load data about the beam energy
        self.beams = json.loads(config.get('Beam energy','E'))
        self.power = float(config.get('Beam energy','power'))
        self.frac = json.loads(config.get('Beam energy','f'))
        if sum(self.frac) > 100:
            raise NameError('Sum of f is greater than 100')

        # load data about the geometry of the beam
        self.pos = json.loads(config.get('Beam geometry','position'))
        self.direc = json.loads(config.get('Beam geometry','direction'))
        self.direc = np.array(self.direc)/sum(self.direc)
        self.std_dev = json.loads(config.get('Beam geometry','std_dev'))
        self.Nz = int(config.get('Beam geometry','Nz'))

    def get_width(self,pos):
        """ Return the width of the beam at the position "pos" (projected 
            against the direction of the beam)
            Used for simplification in the case that someone want to add the
            beam divergence
        """
        return self.std_dev

    def get_attenuation(self,pos):
        return 1
