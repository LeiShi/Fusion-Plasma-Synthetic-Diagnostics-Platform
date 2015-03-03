import math
import json
import numpy as np
import collisions as col
import ConfigParser as psr
import os,sys,inspect
import FPSDP.Maths.Interpolation as Fint


class Beam1D:
    """ Simulate a 1D beam with the help of datas from simulation
    """
    
    def __init__(self,config_file,timesteps,data):
        """ Load everything from the config file"""
        self.cfg_file = config_file                                          #!
        self.data = data                                                     #!
        self.timesteps = timesteps                                           #!
        config = psr.ConfigParser()
        config.read(self.cfg_file)
        
        # load data for collisions
        adas_files = json.loads(config.get('Collisions','adas_file'))
        lifetimes = json.loads(config.get('Collisions','lifetimes'))
        self.collisions = col.Collisions(adas_files,lifetimes)               #!
        self.list_col = json.loads(config.get('Collisions','collisions'))    #!

        # load data about the beam energy
        self.mass_b = json.loads(config.get('Beam energy','mass_b'))         #!
        self.mass_b = np.array(self.mass_b)
        self.beam_comp = json.loads(config.get('Beam energy','E'))           #!
        self.beam_comp = np.array(self.beam_comp)
        self.power = float(config.get('Beam energy','power'))                #!
        self.power = np.array(self.power)
        self.frac = json.loads(config.get('Beam energy','f'))                #!
        self.frac = np.array(self.frac)
        if sum(self.frac) > 100: # allow the possibility to simulate only
            # a part of the beam
            raise NameError('Sum of f is greater than 100')

        # load data about the geometry of the beam
        self.pos = json.loads(config.get('Beam geometry','position'))        #!
        self.pos = np.array(self.pos)
        self.direc = json.loads(config.get('Beam geometry','direction'))     #!
        self.direc = np.array(self.direc)
        self.direc = self.direc/np.sqrt(sum(self.direc**2))
        self.beam_width = json.loads(
            config.get('Beam geometry','beam_width'))                        #!
        self.Nz = int(config.get('Beam geometry','Nz'))                      #!
        
        self.create_mesh()
        self.compute_beam_on_mesh()

    def get_width(self,pos):
        """ Return the width of the beam at the position "pos" (projected 
            against the direction of the beam)
            Used for simplification in the case that someone want to add the
            beam divergence
        """
        return self.beam_width

    def create_mesh(self):
        """ create the 1D mesh between the source of the beam and the end 
            of the mesh
        """
        # intersection between end of mesh and beam
        self.inters = self.find_wall()                                       #!
        length = np.sqrt(sum((self.pos-self.inters)**2))
        dl = np.linspace(0,length,self.Nz)
        self.mesh = np.zeros((self.Nz,3))                                    #!
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
        
        return self.pos + self.direc*t

    def get_electron_density(self,pos):
        """ get the electron density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        return Fint.trilinear_interp_1pt(self.data.grid.X3D,self.data.grid.Y3D,
                                         self.data.grid.Z3D,
                                         self.data.ne_on_grid,pos)
                                 
    def get_ion_density(self,pos):
        """ get the ion density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        # FOR ADDING MUTLIPLE IONS SPECIES CHANGE HERE [add an argument and see
        # where it does not work, and after add the loop over element]
        return Fint.trilinear_interp_1pt(self.data.grid.X3D,self.data.grid.Y3D,
                                         self.data.grid.Z3D,
                                         self.data.ni_on_grid,pos)

                    
    def compute_beam_on_mesh(self):
        """ compute the beam intensity at each position of the mesh 
            with the Gauss-Legendre quadrature (2 points)
        """
        self.density_beam = np.zeros((len(self.beam_comp),self.Nz))          #!
        self.speed = np.zeros(len(self.beam_comp))                           #!
        self.speed = np.sqrt(2*self.beam_comp/self.mass_b*1.660538921e-27)

        for j in range(len(self.mesh[:,0])):
            if j is 0:
                self.density_beam[:,j] = 0
            else:
                temp_beam = np.zeros(len(self.beam_comp))
                for k in self.list_col:
                    file_nber = k[0]
                    beam_nber = k[1]
                    # limit of the integral
                    a = self.mesh[j-1,:]
                    b = self.mesh[j,:]
                    # avoid to compute it twice
                    temp = np.sqrt(1.0/3.0)
                    # average
                    av = (a+b)/2.0
                    # difference
                    diff = (b-a)/2.0
                    # first point
                    pt1 = -temp*diff + av
                    # second point
                    pt2 = temp*diff + av

                    # compute all the values needed for the integral
                    # at the 2 positions
                    ne1 = self.get_electron_density(pt1)
                    ne2 = self.get_electron_density(pt2)
                    
                    T1 = self.get_ion_temp(pt1)
                    T2 = self.get_ion_temp(pt2)
                    ni1 = self.get_ion_density(pt1)
                    ni2 = self.get_ion_density(pt2)
                    # attenuation coefficient from adas
                    S1 = self.collisions.get_attenutation(
                        self.beam_comp[beam_nber],self.mass_b[beam_nber],
                        ni1,T1,file_nber)
                    S2 = self.collisions.get_attenutation(
                        self.beam_comp[beam_nber],self.mass_b[beam_nber],
                        ni2,T2,file_nber)

                    norm_ = 0.5*np.sqrt(sum((b-a)**2))
                    temp_beam[beam_nber] -= (ne1*S1 + ne2*S2)
                    temp_beam[beam_nber] *= norm_/self.speed[beam_nber]
                self.density_beam[:,j] = self.density_beam[:,j-1] - \
                                         temp_beam
        # initial density of the beam
        self.std_dev = self.beam_width/(2*np.sqrt(2*np.log(2)))              #!
        n0 = np.zeros(len(self.beam_comp))
        n0 = np.sqrt(2*self.mass_b*1.660538921e-27)*math.pi*self.power
        n0 *= self.frac*self.std_dev**2/self.beam_comp**(1.5)
        for i in range(len(self.beam_comp)):
            self.density_beam[i,:] = n0[i]*np.exp(self.density_beam[i,:])


    def get_electron_temp(self,pos):
        """ Return the value of the electron temperature from the
            simulation
        """
        return Fint.trilinear_interp_1pt(self.data.grid.X3D,self.data.grid.Y3D,
                                      self.data.grid.Z3D,
                                      self.data.te_on_grid,pos)

    def get_ion_temp(self,pos):
        """ Return the value of the ion temperature from the
            simulation 
        """
        return Fint.trilinear_interp_1pt(self.data.grid.X3D,self.data.grid.Y3D,
                                      self.data.grid.Z3D,
                                      self.data.ti_on_grid,pos)

    def get_mesh(self):
        """ Return the mesh (3D array)"""
        return self.mesh

    def get_origin(self):
        """ Return the origin of the beam"""
        return self.pos
