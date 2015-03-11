import math
import json
import numpy as np
import collisions as col
import ConfigParser as psr
import FPSDP.Maths.Interpolation as Fint
from scipy import interpolate
"""  BE CAREFULL WITH THE LOAD XGC, THE CARTESIAN GRID IS NOT IN
     A CLASSICAL POSITION (Y for the heigth)
     In the code of the BES, the Z axis is for the height (so switch between Y and Z)
     data_on_grid[Z,Y,X]
"""


def to_other_index(pos):
    " Change the order of the index "
    if len(pos.shape) == 2:
        a = np.array([pos[1,:],pos[2,:],pos[0,:]])
    elif len(pos.shape) == 1:
        a = np.array([pos[1],pos[2],pos[0]])
    else:
        raise NameError('Wrong shape of pos')
    return a 

class Beam1D:
    """ Simulate a 1D beam with the help of datas from simulation

        Methods:
        __init__(config_file,timesteps,data)
                            -- config_file is the name of the config file,
                               timesteps is the list of the timesteps wanted
                               and data are the data from XGC1 (for example)
       
        get_width(dist)     -- return the standard deviation at the distance given

        create_mesh()       -- create the points of the mesh

        find_wall(eps=1e-6) -- find the end (for the beam) of the mesh from the data
                               eps is for avoiding the end of the mesh (relative length)

        get_electron_density(pos,t_) 
                            -- function simplificating the acces to the data
                               (avoid the problem of the two representations)
                               pos is a 1D or 2D array (first index is the dimension)
                               the coordinate are the same than in the beam config file

        get_ion_density(pos,t_)
                            -- same as before
                    
        get_electron_temp(pos)
                            -- same as before (the perturbation of the temperature)
                               are not taken in account (small effect)

        get_ion_temp(self,pos)
                            -- same as before

        get_electron_density_fluc(pos,t_)
                            -- same as before


        compute_beam_on_mesh()
                            -- compute the density of the beam on the mesh with a
                               Gauss-Legendre formula (two points)
 
        get_mesh()          -- return the mesh, same coordinate than the beam config file
                               first index is for the dimension
 
        get_origin()        -- return the origin of the beam
        
        get_beam_density(pos,t_)
                            -- use a gaussian profile and an interpolation in order to
                               compute the beam density at the position. t_ is the index
                               for the time


        get_emis(pos,t_)    -- compute the emissivity \epsilon 

        get_emis_fluc(pos,t_)
                            -- compute the fluctuation of the emissivity with only
                               the fluctuation part of the density
        get_emis_ave(pos)   -- compute the fluctuation using the average over time

        Attribut:
    
        cfg_file     --  name of the config file
        data         --  data from the simulation
        timesteps    --  list of the timesteps considered
        adas_atte    --  list of file for the attenuation
        adas_emis    --  list of file for the emission
        collisions   --  class containing all the informations for the collisions
        coll_atte    --  tuple relating the adas file with the beams (choice of collision)
                         for the attenuation
        coll_emis    --  same as before but with the emission
        mass_b       --  mass of the particles in the beam (in amu)
        beam_comp    --  energy of each beam components
        power        --  total power of the beam
        frac         --  fraction of power for each components
        pos          --  origin of the beam
        direc        --  direction of the beam
        beam_width   --  FWHM of the beam at the origin
        Nz           --  number of point on the beam mesh

        inters       --  ending point of the mesh
        mesh         --  mesh (first index for the dimension)

        density_beam --  beam density at the grid points
        speed        --  speed of each particles in a beam component
        std_dev      --  standard deviation of the beam at the origin
        std_dev2     --  square of std_dev
        dens         --  electron density at the grid points

    """
    
    def __init__(self,config_file,timesteps,data):
        """ Load everything from the config file"""
        self.cfg_file = config_file                                          #!
        self.data = data                                                     #!
        self.timesteps = timesteps                                           #!
        print 'Reading config file (beam)'
        config = psr.ConfigParser()
        config.read(self.cfg_file)
        
        # load data for collisions
        self.adas_atte = json.loads(config.get('Collisions','adas_atte'))    #!
        self.adas_emis = json.loads(config.get('Collisions','adas_emis'))    #!
        lifetimes = json.loads(config.get('Collisions','lifetimes'))
        self.collisions = col.Collisions(self.adas_atte,
                                         self.adas_emis,lifetimes)           #!
        self.coll_atte = json.loads(config.get('Collisions','coll_atte'))    #!
        self.coll_emis = json.loads(config.get('Collisions','coll_emis'))    #!

        # load data about the beam energy
        self.mass_b = json.loads(config.get('Beam energy','mass_b'))         #!
        self.mass_b = np.array(self.mass_b)
        self.beam_comp = json.loads(config.get('Beam energy','E'))           #!
        self.beam_comp = 1000*np.array(self.beam_comp)
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

        print 'Creating mesh'
        self.create_mesh()
        print 'Computing density of the beam'
        self.compute_beam_on_mesh()

    def get_width(self,dist):
        """ Return the width of the beam at the distance "dist" (projected 
            against the direction of the beam)
            Used for simplification in the case that someone want to add the
            beam divergence
        """
        return self.beam_width*np.ones(dist.shape)

    def create_mesh(self):
        """ create the 1D mesh between the source of the beam and the end 
            of the mesh
        """
        # intersection between end of mesh and beam
        self.inters = self.find_wall()                                       #!
        length = np.sqrt(sum((self.pos-self.inters)**2))
        # distance to the origin along the central line
        self.dl = np.linspace(0,length,self.Nz)
        self.mesh = np.zeros((3,self.Nz))                                    #!
        # the first index corresponds to the dimension (X,Y,Z)
        self.mesh[0,:] = self.pos[0] + self.dl*self.direc[0]
        self.mesh[1,:] = self.pos[1] + self.dl*self.direc[1]
        self.mesh[2,:] = self.pos[2] + self.dl*self.direc[2]
                    
    def find_wall(self, eps=1e-6):
        """ find the wall (of the mesh) that will stop the beam and return
            the coordinate of the intersection with the beam
            eps is used to avoid the end of the mesh
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
            ty = abs(self.data.grid.Zmax-self.pos[1])/self.direc[1]
        elif self.direc[1] < 0:
            ty = -abs(self.data.grid.Zmin-self.pos[1])/self.direc[1]
        else:
            ty = float('Inf')

        # Z-direction
        # choose the right wall
        if self.direc[2] > 0:
            tz = abs(self.data.grid.Ymax-self.pos[2])/self.direc[2]
        elif self.direc[2] < 0:
            tz = -abs(self.data.grid.Ymin-self.pos[2])/self.direc[2]
        else:
            tz = float('Inf')

        t = min([tx,ty,tz])
        # check if there is an intersection
        if t is float('Inf'):
            raise NameError('Cannot find the wall for the beam')

        return self.pos + self.direc*t*(1-eps)

    def get_electron_density(self,pos,t_):
        """ get the electron density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        a = to_other_index(pos)
        # the order of the grid is due to the XGC loading coordinate
        return Fint.trilinear_interp_1pt(self.data.grid.Z1D,self.data.grid.Y1D,
                                         self.data.grid.X1D,
                                         self.data.ne_on_grid[0,t_,:,:,:],a)

    def get_electron_density_fluc(self,pos,t_):
        """ get the fluctuation in the electron density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        a = to_other_index(pos)
        print 'Warning inefficient'
        dne = self.data.ne_on_grid[0,t_,:,:,:] - self.data.ne0_on_grid
        # the order of the grid is due to the XGC loading coordinate
        return Fint.trilinear_interp_1pt(self.data.grid.Z1D,self.data.grid.Y1D,
                                         self.data.grid.X1D,
                                         dne,a)
    
    def get_ion_density(self,pos,t_):
        """ get the ion density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        # FOR ADDING MUTLIPLE IONS SPECIES CHANGE HERE [add an argument and see
        # where it does not work, and after add the loop over element]
        a = to_other_index(pos)
        return Fint.trilinear_interp_1pt(self.data.grid.Z1D,self.data.grid.Y1D,
                                         self.data.grid.X1D,
                                         self.data.ni_on_grid[0,t_,:,:,:],a)
                    
    def compute_beam_on_mesh(self):
        """ compute the beam intensity at each position of the mesh 
            with the Gauss-Legendre quadrature (2 points)
        """
        self.density_beam = np.zeros((len(self.timesteps),
                                     len(self.beam_comp),self.Nz))           #!
        # speed of each beam
        self.speed = np.zeros(len(self.beam_comp))                           #!
        self.speed = np.sqrt(2*self.beam_comp*9.6485383e10/self.mass_b)
        # get the standard deviation at the origin
        self.std_dev = self.beam_width/(2*np.sqrt(2*np.log(2)))              #!
        self.std_dev2 = self.std_dev**2
        # density of the beam at the origin
        n0 = np.zeros(len(self.beam_comp))
        n0 = np.sqrt(2*self.mass_b*1.660538921e-27)*math.pi*self.power
        n0 *= self.frac*self.std_dev2/(self.beam_comp*1.60217733e-16)**(1.5)
        self.dens = np.zeros((len(self.timesteps),len(self.mesh[0,:])))      #!

        for t_ in range(len(self.timesteps)):
            print 'Timestep number: ', t_ + 1, ' over ', len(self.timesteps)
            for j in range(len(self.mesh[0,:])):
                # density over the central line (usefull for some check)
                self.dens[t_,j] = self.get_electron_density(self.mesh[:,j],t_)
                if j is not 0:
                    temp_beam = np.zeros(len(self.beam_comp))
                    for k in self.coll_atte:
                        file_nber = k[0]
                        beam_nber = k[1]
                        # limit of the integral
                        a = self.mesh[:,j-1]
                        b = self.mesh[:,j]
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
                        ne1 = self.get_electron_density(pt1,t_)
                        ne2 = self.get_electron_density(pt2,t_)
                    
                        T1 = self.get_ion_temp(pt1)
                        T2 = self.get_ion_temp(pt2)
                        
                        ni1 = self.get_ion_density(pt1,t_)
                        ni2 = self.get_ion_density(pt2,t_)

                        # attenuation coefficient from adas
                        S1 = self.collisions.get_attenutation(
                            self.beam_comp[beam_nber],self.mass_b[beam_nber],
                            ni1,T1,file_nber)
                        S2 = self.collisions.get_attenutation(
                            self.beam_comp[beam_nber],self.mass_b[beam_nber],
                            ni2,T2,file_nber)
                        # half distance between a & b
                        norm_ = 0.5*np.sqrt(sum((b-a)**2))
                        temp1 = (ne1*S1 + ne2*S2)
                        temp1 *= norm_/self.speed[beam_nber]
                        temp_beam[beam_nber] += temp1

                    self.density_beam[t_,:,j] = self.density_beam[t_,:,j-1] - \
                                                temp_beam

            # initial density of the beam
            for i in range(len(self.beam_comp)):
                self.density_beam[t_,i,:] = n0[i]*np.exp(self.density_beam[t_,i,:])

    def get_electron_temp(self,pos):
        """ Return the value of the electron temperature from the
            simulation
        """
        a = to_other_index(pos)
        return Fint.trilinear_interp_1pt(self.data.grid.Z1D,self.data.grid.Y1D,
                                         self.data.grid.X1D,
                                         self.data.te_on_grid,a)
    

    def get_ion_temp(self,pos):
        """ Return the value of the ion temperature from the
            simulation 
        """
        a = to_other_index(pos)
        return Fint.trilinear_interp_1pt(self.data.grid.Z1D,self.data.grid.Y1D,
                                         self.data.grid.X1D,
                                         self.data.ti_on_grid,a)

    def get_mesh(self):
        """ Return the mesh (3D array)"""
        return self.mesh

    def get_origin(self):
        """ Return the origin of the beam"""
        return self.pos

    def get_beam_density(self,pos,t_):
        """ Return the beam density at the position and time step wanted
            assuming a gaussian profile
        """
        if len(pos.shape) == 1:
            # array to return
            nb = np.zeros(len(self.beam_comp))
            # vector from beam origin to the wanted position
            dist = pos - self.get_origin()
            proj = np.dot(dist,self.direc)
            stddev2 = self.get_width(proj)**2
            if proj < 0:
                raise NameError('Point before the origin of the beam')
            # cubic spline for finding the value along the axis
            for i in range(len(self.beam_comp)):
                tck = interpolate.splrep(self.dl,self.density_beam[t_,i,:])
                nb[i] = interpolate.splev(proj,tck)
            # radius^2 on the plane perpendicular to the beam
            R2 = dist - proj*self.direc
            R2 = sum(R2**2)
            nb = nb*np.exp(-R2/(2*stddev2))
            return nb
        elif len(pos.shape) == 2:
            # array to return
            nb = np.zeros((len(self.beam_comp),len(pos[0,:])))
            # vector from beam origin to the wanted position
            dist = np.zeros(pos.shape)
            dist[0,:] = pos[0,:] - self.get_origin()[0]
            dist[1,:] = pos[1,:] - self.get_origin()[1]
            dist[2,:] = pos[2,:] - self.get_origin()[2]
            # result is a 1D array containing the projection of the distance
            # from the origin over the direction of the beam
            proj = np.einsum('ij,i->j',dist,self.direc)
            stddev2 = self.get_width(proj)**2
            # before the beam, the density will be set to 0
            indout = np.where((proj < 0) | (proj > self.dl[-1]))
            indin =  np.where((proj > 0) & (proj < self.dl[-1]))
            # cubic spline for finding the value along the axis
            for i in range(len(self.beam_comp)):
                tck = interpolate.splrep(self.dl,self.density_beam[t_,i,:])
                for j in range(len(indin)):
                    nb[i,indin[j]] = interpolate.splev(proj[indin[j]],tck)
            # radius^2 on the plane perpendicular to the beam
            R2 = np.zeros(dist.shape)
            R2[0,:] = dist[0,:] - proj*self.direc[0]
            R2[1,:] = dist[1,:] - proj*self.direc[1]
            R2[2,:] = dist[2,:] - proj*self.direc[2]
            # compute the norm of each position
            R2 = np.einsum('ij,ij->j',R2,R2)
            for i in range(len(self.beam_comp)):
                nb[i,:] = nb[i,:]*np.exp(-R2/(2*stddev2))
            nb[:,indout] = 0.0
            return nb
        else:
            raise NameError('Error: wrong shape for pos')

    def get_emis(self,pos,t_):
        """ Return the emissivity at pos and time t_ 
            epsilon = <sigma*v> n_b n_e
            Argument:
            pos   -- 2D array, first index is for X,Y,Z
        """
        print 'DO NOT TAKE LIFETIME EFFECT'
        # first take all the value needed for the computation
        n_b = self.get_beam_density(pos,t_)
        n_e = self.get_electron_density(pos,t_)
        emis = np.zeros((len(self.beam_comp),len(pos[0,:])))
        Te = self.get_electron_temp(pos)
        # loop over all the type of collisions
        for k in self.coll_atte:
            file_nber = k[0]
            beam_nber = k[1]
            # compute the emission coefficient
            emis[beam_nber,:] += self.collisions.get_emission(
                self.beam_comp[beam_nber],n_e,self.mass_b[beam_nber],Te,file_nber)
        # compute the emissivity
        for i in range(len(self.beam_comp)):
            emis[i] *= n_e*n_b[i,:]
        return emis


    def get_emis_fluc(self,pos,t_):
        """ Return the fluctuation of the emissivity at pos and time t_ 
            epsilon = <sigma*v> n_b \delta n_e
            Argument:
            pos   -- 2D array, first index is for X,Y,Z
        """
        print 'DO NOT TAKE LIFETIME EFFECT'
        # first take all the value needed for the computation
        n_b = self.get_beam_density(pos,t_)
        dn_e = self.get_electron_density_fluc(pos,t_)
        emis = np.zeros((len(self.beam_comp),len(pos[0,:])))
        Te = self.get_electron_temp(pos)
        # loop over all the type of collisions
        for k in self.coll_atte:
            file_nber = k[0]
            beam_nber = k[1]
            # compute the emission coefficient
            emis[beam_nber,:] += self.collisions.get_emission(
                self.beam_comp[beam_nber],dn_e,self.mass_b[beam_nber],Te,file_nber)
        # compute the emissivity
        for i in range(len(self.beam_comp)):
            emis[i] *= dn_e*n_b[i,:]
        return emis


    
    def get_emis_ave(self,pos):
        """ Return the fluctuation of the emissivity at pos
            epsilon = <sigma*v> n_b n_e - <epsilon>_t
            Argument:
            pos   -- 2D array, first index is for X,Y,Z
        """
        print 'DO NOT TAKE LIFETIME EFFECT'
        # first take all the value needed for the computation
        emis = np.zeros((len(self.timesteps),len(self.beam_comp),len(pos[0,:])))
        Te = self.get_electron_temp(pos)
        for t_ in range(len(self.timesteps)):
            n_b = self.get_beam_density(pos,t_)
            n_e = self.get_electron_density(pos,t_)
            # loop over all the type of collisions
            for k in self.coll_atte:
                file_nber = k[0]
                beam_nber = k[1]
                # compute the emission coefficient
                emis[t_,beam_nber,:] += self.collisions.get_emission(
                    self.beam_comp[beam_nber],n_e,self.mass_b[beam_nber],Te,file_nber)
            # compute the emissivity
            for i in range(len(self.beam_comp)):
                emis[t_,i] *= n_e*n_b[i,:]
        ave = np.sum(emis, axis=0)/len(self.timesteps)
        for t_ in range(len(self.timesteps)):
            emis[t_,:,:] -= ave
        return emis
    
