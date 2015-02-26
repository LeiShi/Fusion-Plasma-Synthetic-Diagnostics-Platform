import ADAS_file as adas

class Collisions:
    """ Class containing all the physics about the collisions
        Read the files from ADAS database, compute the lifetiime, and compute
        the cross-sections
    """

    def __init__(self,files,beams,lifetimes):
        """ Copy the input inside the instance
            
            Arguments:
            files    -- list containing the name of all files
            beams    -- list of the different energy beam (should be in the 
                        same order than the files)
            lifetime -- list of the lifetime of the excited states (same order
                        than the other lists)
        """
        self.files = files
        self.lifetimes = lifetimes
        self.beam_data = []
        self.read_adas(beams)
        
    def read_adas(self,beams):
        """ Read the ADAS files and stores them as attributes
        """
        for name in self.files:
            self.beam_data.append(adas.Beam_ADAS_File(name,beams))

    def get_beams(self):
        """ just for simplifying the writting"""
        return self.beam_data[0].beams

    def get_coef(self,beam,file_number):
        for i in range(len(self.get_beams())):
            if abs(self.get_beams()[i] - beam)/abs(beam) < 1e-4:
                return self.beam_data[file_number].coef[i]
        raise NameError('Beam wanted not in the computed beam')

    def get_density(self):
        return self.beam_data[0].densities
