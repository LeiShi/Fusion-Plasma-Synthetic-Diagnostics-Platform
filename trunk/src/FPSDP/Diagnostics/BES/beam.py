import json
import numpy as np
import collisions as col
import ConfigParser as psr
import os,sys,inspect
print 'REMOVE THIS'
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import Maths.Interpolation as Fint


class Beam1D:
    """ Simulate a 1D beam with the help of datas from simulation
    """
    
    def __init__(self,config_file,timesteps,data):
        """ Load everything from the config file"""
        self.cfg_file = config_file                                          #!
        self.data = data                                                     #!
        config = psr.ConfigParser()
        config.read(self.cfg_file)

        # load data for collisions
        adas_files = json.loads(config.get('Collisions','adas_file'))
        lifetimes = json.loads(config.get('Collisions','lifetimes'))
        self.collisions = col.Collisions(adas_files,lifetimes)               #!
        self.list_col = json.loads(config.get('Collisions','collisions'))    #!

        # load data about the beam energy
        self.beams = json.loads(config.get('Beam energy','E'))               #!
        self.power = float(config.get('Beam energy','power'))                #!
        self.frac = json.loads(config.get('Beam energy','f'))                #!
        if sum(self.frac) > 100:
            raise NameError('Sum of f is greater than 100')

        # load data about the geometry of the beam
        self.pos = json.loads(config.get('Beam geometry','position'))        #!
        self.pos = np.array(self.pos)
        self.direc = json.loads(config.get('Beam geometry','direction'))     #!
        self.direc = np.array(self.direc)
        self.direc = self.direc/np.sqrt(sum(self.direc**2))
        self.std_dev = json.loads(config.get('Beam geometry','std_dev'))     #!
        self.Nz = int(config.get('Beam geometry','Nz'))                      #!
        
        self.create_mesh()

    def get_width(self,pos):
        """ Return the width of the beam at the position "pos" (projected 
            against the direction of the beam)
            Used for simplification in the case that someone want to add the
            beam divergence
        """
        return self.std_dev

    def create_mesh(self):
        """ create the 1D mesh between the source of the beam and the end 
            of the mesh
        """
        # intersection between end of mesh and beam
        self.inters = self.find_wall()
        length = np.sqrt(sum((self.pos-self.inters)**2))
        dl = np.linspace(0,length,self.Nz)
        self.mesh = np.zeros((self.Nz,3))
        self.mesh[:,0] = self.pos[0] + dl*self.inters[0]/sum(self.inters**2)
        self.mesh[:,1] = self.pos[1] + dl*self.inters[1]/sum(self.inters**2)
        self.mesh[:,2] = self.pos[2] + dl*self.inters[2]/sum(self.inters**2)
                
    def find_wall(self):
        """ find the wall (of the mesh) that will stop the beam and return
            the coordinate of the intersection with the beam
        """
        # X-direction
        # choose the right wall
        if self.direc[0] > 0:
            tx = abs(self.data.grid.Xmax-self.pos[0])/self.direc[0]
        elif self.direc[0] < 0:
            tx = -abs(self.data.grid.Xmin-self.pos[0])/self.direc[0]
        else:
            tx = float('Inf')

        # Y-direction
        # choose the right wall
        if self.direc[1] > 0:
            ty = abs(self.data.grid.Ymax-self.pos[1])/self.direc[1]
        elif self.direc[1] < 0:
            ty = -abs(self.data.grid.Ymin-self.pos[1])/self.direc[1]
        else:
            ty = float('Inf')

        # Z-direction
        # choose the right wall
        if self.direc[2] > 0:
            tz = abs(self.data.grid.Zmax-self.pos[2])/self.direc[2]
        elif self.direc[2] < 0:
            tz = -abs(self.data.grid.Zmin-self.pos[2])/self.direc[2]
        else:
            tz = float('Inf')

        t = min([tx,ty,tz])
        ind = [i for i, j in enumerate([tx,ty,tz]) if j == t]

        self.get_density(np.array([1.324,0.1234,0.12434]))
        return self.pos + self.direc*t

    def get_density(self,pos):
        """ get the density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        
        return Fint.trilinear_interp_1pt(self.data.grid.X3D,self.data.grid.Y3D,
                                         self.data.grid.Z3D,self.data.ne,pos)
    
    def compute_beam_on_mesh(self):
        """ compute the beam intensity at each position of the mesh 
        """
        for i in range(self.Nz):
            alpha = 0
            for l_col in range(len(self.collisions)):
                alpha = 1
