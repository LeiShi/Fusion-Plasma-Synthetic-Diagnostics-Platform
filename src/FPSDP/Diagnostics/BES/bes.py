import numpy as np
import scipy as sp
import json # for loading input
import ConfigParser as psr # for loading input
import FPSDP.Diagnostics.Beam.beam as be
import FPSDP.Geometry.Grid as Grid
import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Maths.Integration as integ
from os.path import exists


def heuman(phi,m):
    """ Compute the Heuman's lambda function """
    F_ = sp.special.ellipkinc(phi,1.0-m) # incomplete elliptic integral of 1st kind
    K = sp.special.ellipk(m) # complete elliptic integral of 1st kind
    K_ = sp.special.ellipk(1.0-m) # complete elliptic integral of 1st kind
    E_ = sp.special.ellipe(1.0-m) # complete elliptic integral of 2nd kind
    Z_ = E_ - E_*F_/K_ # Jacobi Zeta function
    ret = F_/K_ + 2.0*K_*Z_/np.pi
    return ret

def angle(v1,v2):
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def solid_angle_circle_off_axis(pos,r):
    """ Compute the solid angle of a circle on/off-axis from the pos
        the center of the circle should be in (0,0,0)

        look the paper of Paxton: "Solid Angle Calculation for a 
        Circular Disk" in 1959
    """
    print 'change name'
    # define a few value
    R = np.sqrt(np.sum(pos[:,0:1]**2, axis=1))
    ind1 = np.where(R != 0)[0]
    ind2 = np.where(R == 0)[0]
    Rmax = np.sqrt(pos[ind1,2]**2 + (R[ind1]+r)**2)
    R1 = np.sqrt(pos[ind1,2]**2 + (R[ind1]-r)**2)
    k = np.sqrt(1-(R1/Rmax)**2)
    LK_R = 2.0*pos[ind1,2]*sp.special.ellipk(k)
    # not use for R=r but it should not append
    # often
    xsi = np.arctan(pos[ind1,2]/abs(r-R[ind1]))
    pilam = np.pi*heuman(xsi,k)
    # the three different case
    inda = np.where(R[ind1] == r)
    indb = np.where(R[ind1] < r)
    indc = np.where(R[ind1] > r)
    # compute the solid angle
    solid = np.zeros(pos.shape[0])
    solid[ind1[inda]] = np.pi - LK_R[inda]
    solid[ind1[indb]] = 2.0*np.pi - LK_R[indb] - pilam[indb]
    solid[ind1[indc]] = - LK_R[indc] - pilam[indc]

    solid[ind2] = 2*np.pi*pos[ind2,2]*(1.0/pos[ind2,2] - 1.0/np.sqrt(r**2 - pos[ind2,2]**2))
    return solid

class BES:
    """ Class computing the image of all the fiber
    """

    def __init__(self,input_file):
        """ load all the data from the input file"""
        self.cfg_file = input_file                                           #!
        if not exists(self.cfg_file):
            raise NameError('Config file not found')
        config = psr.ConfigParser()
        config.read(self.cfg_file)


        # Optics part
        self.focal = json.loads(config.get('Optics','foc'))                  #!
        self.op_direc = json.loads(config.get('Optics','dir'))               #!
        # normalize the vector
        self.op_direc = np.array(self.op_direc)
        self.op_direc = self.op_direc/np.sqrt(np.sum(self.op_direc**2))
        self.rad_foc = json.loads(config.get('Optics','rad_foc'))            #!
        self.rad_pup = json.loads(config.get('Optics','rad_pup'))            #!
        self.inter = json.loads(config.get('Optics','int'))                  #!
        self.Nint = json.loads(config.get('Optics','Nint'))                  #!
        
        # Fiber part
        X = json.loads(config.get('Fiber','X'))
        Y = json.loads(config.get('Fiber','Y'))
        Z = json.loads(config.get('Fiber','Z'))
        self.fib_pos = np.zeros((len(X),3))                                  #!
        self.fib_pos[:,0] = X
        self.fib_pos[:,1] = Y
        self.fib_pos[:,2] = Z

        # Data part
        self.data_path = config.get('Data','data_path')                      #!
        self.N = json.loads(config.get('Data','N'))                          #!
        start = json.loads(config.get('Data','timestart'))
        end = json.loads(config.get('Data','timeend'))
        timestep = json.loads(config.get('Data','timestep'))
        self.time = np.arange(start,end+1,timestep)                          #!
        # position swap due to a difference in the axis
        grid3D = Grid.Cartesian3D(Xmin=1.4, Xmax=2.1, Ymin=-0.2, Ymax=0.2,
                                  Zmin=-0.5, Zmax=0.5, NX=self.N[0], NY=self.N[2], NZ=self.N[1])
        xgc_ = xgc.XGC_Loader(self.data_path, grid3D, start, end, timestep,
                              Fluc_Only = False, load_ions=True, equilibrium_mesh = '3D')

        self.beam = be.Beam1D(input_file,range(len(self.time)),xgc_)         #!

        print 'no check of division by zero'
        self.perp1 = np.array([1,-self.op_direc[0]/self.op_direc[1],0])
        self.perp1 = self.perp1/np.sqrt(np.sum(self.perp1**2))
        self.perp2 = np.cross(self.op_direc,self.perp1)
        

    def to_cart_coord(self,pos,fiber_nber):
        """ return the cartesian coordinate from the coordinate of the lens
            Attribut:
            pos   --  (X,Y,Z) where Z is along the sightline, X coorespond
                      to perp1, and Y to perp2
        """
        if len(pos.shape) == 1:
            ret = self.fib_pos[fiber_nber,:] + self.op_direc*pos[2]
            ret += self.perp1*pos[0] + self.perp2*pos[1]
        else:
            ret = np.zeros(pos.shape)
            ret[:,0] = self.fib_pos[fiber_nber,0] + self.op_direc[0]*pos[:,2]
            ret[:,0] += self.perp1[0]*pos[:,0] + self.perp2[0]*pos[:,1]
            ret[:,1] = self.fib_pos[fiber_nber,1] + self.op_direc[1]*pos[:,2]
            ret[:,1] += self.perp1[1]*pos[:,0] + self.perp2[1]*pos[:,1]
            ret[:,2] = self.fib_pos[fiber_nber,2] + self.op_direc[2]*pos[:,2]
            ret[:,2] += self.perp1[2]*pos[:,0] + self.perp2[2]*pos[:,1]
        return ret

    def get_width(self,pos):
        """ Return the radius of the light cone at pos (optical coordinate)
            Assume two cones that meet 
        """
        if len(pos.shape) == 1:
            # distance from the ring
            a = abs(pos[2]-self.focal)
            a *= (self.rad_pup-self.rad_foc)/self.focal
            return a + self.rad_foc
        else:
            # distance from the ring
            a = abs(pos[:,2]-self.focal)
            a *= (self.rad_pup-self.rad_foc)/self.focal
            return a + self.rad_foc
    
    def check_in(self,pos):
        """ Check if the position (optical coordinate) is inside the first cone
            (if the focus ring matter or not)
            ASSUME THE SAME Z COORDINATE for all the pos!
        """
        ret = np.zeros(pos.shape[0], dtype=bool)
        # before the focus point
        ind1 = np.where(pos[:,2] < self.focal)[0]
        # after the focus point
        ind2 = np.where(pos[:,2] >= self.focal)[0]
        ret[ind1] = True
        # distance from the focus point along the z-axis
        a = self.focal-pos[ind2,2]
        # size of the 'ring' scaled to this position
        a *= (self.rad_pup-self.rad_foc)/self.focal + self.rad_foc
        # distance from the axis
        R = np.sqrt(np.sum(pos[ind2,0:1]**2, axis=1))
        ind1 = np.where(a > R)
        ret[ind2[ind1]] = True

        return ret
        
    def light_from_plane(self,z, t_, fiber_nber):
        """ Compute the light from one plane using a order 10 method (see report or
            Abramowitz and Stegun)
            Arguments:
            z          --  distance from the fiber along the sightline
            t_         --  timestep wanted (index inside the code,
                           not the one in the simulation)
            fiber_nber --  number of the fiber
        """
        center = np.zeros((len(z),3))
        center[:,2] = z # define the center of the circle
        # first step: define all the weight and points
        r = self.get_width(center)
        nber_comp = self.beam.beam_comp.shape[0]
        I = np.zeros((z.shape[0],len(t_),nber_comp))
        for i,r_ in enumerate(r):
            quad = integ.integration_points(2, 'order10', 'circle', r_)
            pos = np.zeros((len(quad.pts[:,0]),3))
            pos[:,0] = quad.pts[:,0]
            pos[:,1] = quad.pts[:,1]
            pos[:,2] = z[i]*np.ones(len(quad.pts[:,0]))
            eps = self.get_emis_from(pos,t_,fiber_nber)
            I[i,:,:] = np.einsum('k,ijk->ij',quad.w,eps)
        return I

    def intensity(self,t_,fiber_nber):
        """ Compute the light received by the fiber #fiber_nber
        """
        # first define the quadrature formula
        quad = integ.integration_points(1,'GL3') # Gauss-Legendre order 3
        I = 0.0
        foc_point = self.to_cart_coord(np.array([0,0,self.focal]),fiber_nber)
        # compute the distance from the origin of the beam
        dist = np.dot(foc_point - self.beam.pos,self.beam.direc)
        width = self.beam.get_width(dist)
        width = 0.5*(width[0] + width[1])
        border = np.linspace(-width,width,self.Nint)
        Z = 0.5*(border[0:-2] + border[1:-1])
        ba2 = 0.5*(border[2]-border[1])
        for z in Z:
            pt = z + ba2*quad.pts
            I += np.einsum('k,kij->ij',quad.w,self.light_from_plane(pt,t_,fiber_nber))
        I *= ba2
        return I
        
    def get_emis_from(self,pos,t_,fiber_nber):
        """ Compute the total emission received from pos (takes in account the
            solid angle). [Paxton 1959]
            Argument:
            pos  --  position of the emission in the optical coordinate (X,Y,Z)

            Improvement possible: keep in memory the solid angle for different time
        """
        x = self.to_cart_coord(pos,fiber_nber)
        eps = self.beam.get_emis_lifetime(x,t_)/(4.0*np.pi)
        # now compute the solid angle
        solid = self.get_solid_angle(pos)
        for i in range(eps.shape[0]):
            for j in range(eps.shape[1]):
                eps[i,j,:] *= solid
        return eps

    def get_solid_angle(self,pos):
        """ compute the solid angle from the position """
        test = self.check_in(pos)
        ind1 = np.where(test)[0]
        ind2 = np.where(~test)[0]
        solid = np.zeros(pos.shape[0])

        # first case
        solid[ind1] = solid_angle_circle_off_axis(pos[ind1,:],self.rad_pup)

        # second case
        # first find the position of the 'intersection' between the lens and the ring
        # define a few constant (look my report for the detail, too much computation)
        # to write them in the comments
        f = 1.0/(1.0-self.focal/pos[ind2,2])
        R2 = np.sum(pos[ind2,0:1]**2) # norm in the perpendicular plane
        ratio2 = (pos[ind2,0]/pos[ind2,1])**2
        print('need to make a check for division by 0')
        A = (self.rad_pup**2 - self.rad_foc*f**2 - R2*(1.0+f)**2)/(2.0*f*(1.0+f))
        print('check need here too')
        delta = 4.0*(ratio2 - (1.0-ratio2)*((A/pos[ind2,1])**2 - self.rad_foc))

        # first case
        ind = np.where(delta > 0)[0]
        if len(ind) != 0:
            print ind
            x1 = np.zeros((len(ind),2))
            x1[:,0] = 2*pos[ind2[ind],0]/pos[ind2[ind],1] + np.sqrt(delta)
            x1[:,0] /= 2.0*(1-ratio2)
            x1[:,1] = (A - x1[:,0]*pos[ind2[ind],0])/pos[ind2[ind],1]
            
            x2 = np.zeros((len(ind),2))
            x2[:,0] = 2*pos[ind2[ind],0]/pos[ind2[ind],1] - np.sqrt(delta)
            x2[:,0] /= 2.0*(1-ratio2)
            x2[:,1] = (A - x2[:,0]*pos[ind2[ind],0])/pos[ind2[ind],1]
            
            y1 = x1*f[ind] + pos[ind2[ind],0:1]*(1.0 + f[ind])
            y2 = x2*f[ind] + pos[ind2[ind],0:1]*(1.0 + f[ind])
            # now we can compute the solid angle
            solid[ind2[ind]] = np.sqrt(np.sum(pos**2))
            # compute the 'surface' of the lens/ring
            theta = angle(x1,x2)
            S = 0.5*self.rad_foc**2*(theta - np.sin(theta))
            theta = angle(y1,y2)
            S += 0.5*self.rad_pup**2*(theta - np.sin(theta))
            # compute the scalar product between the lens and the radius is simply pos_z
            # the norm is at power 3 due to the normalization of the scalar product
            solid[ind2[ind]] = pos[ind2[ind],2]*S/solid**3

        # second case
        ind = np.where(delta < 0)[0]
        if len(ind) != 0:
            q = pos[ind2[ind],:]
            q[:,2] -= self.focal
            solid[ind2[ind]] = solid_angle_circle_off_axis(q,self.rad_foc)
            
        return solid
