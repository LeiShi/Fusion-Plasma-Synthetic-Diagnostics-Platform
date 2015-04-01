# Math
import numpy as np
import scipy as sp
# Input file
import json # for loading input
import ConfigParser as psr # for loading input
# parallel
import multiprocessing as mp
import copy_reg
import types
# beam
import FPSDP.Diagnostics.Beam.beam as be
# grid for data
import FPSDP.Geometry.Grid as Grid
# data loader
import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
# quadrature formula
import FPSDP.Maths.Integration as integ
from os.path import exists # used for checking if the input file exists
# it is not clear with configparser error

# interpolation (for the ideal bes)
from scipy import interpolate


def _pickle_method(m):
    """ stuff for parallelisation"""
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

# for parallelisation
copy_reg.pickle(types.MethodType, _pickle_method)

def heuman(phi,m):
    """ Compute the Heuman's lambda function (defined in Paxton,
        see soldi_angle_disk)
    """
    m2 = 1-m
    F2 = sp.special.ellipkinc(phi,m2) # incomplete elliptic integral of 1st kind
    K = sp.special.ellipk(m) # complete elliptic integral of 1st kind
    E = sp.special.ellipe(m) # complete elliptic integral of 2nd kind
    E2 = sp.special.ellipeinc(phi,m2) # incomplete elliptic integral of 2nd kind
    ret = 2.0*(E*F2+K*E2-K*F2)/np.pi
    return ret

def angle(v1,v2):
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def solid_angle_disk(pos,r):
    """ Compute the solid angle of a circle on/off-axis from the pos
        the center of the circle should be in (0,0,0)

        look the paper of Paxton: "Solid Angle Calculation for a 
        Circular Disk" in 1959
    """
    # define a few value
    r0 = np.sqrt(np.sum(pos[:,0:2]**2, axis=1))
    ind1 = r0 != 0
    ind2 = ~ind1
    Rmax = np.sqrt(pos[ind1,2]**2 + (r0[ind1]+r)**2)
    R1 = np.sqrt(pos[ind1,2]**2 + (r0[ind1]-r)**2)
    k = np.sqrt(1-(R1/Rmax)**2)
    LK_R = 2.0*abs(pos[ind1,2])*sp.special.ellipk(k**2)/Rmax
    # not use for R=r but it should not append
    # often
    xsi = np.arctan(abs(pos[ind1,2]/(r-r0[ind1])))
    pilam = np.pi*heuman(xsi,k**2)
    # the three different case
    inda = r0[ind1] == r
    indb = r0[ind1] < r
    indc = (~inda) & (~indb)
    # compute the solid angle
    solid = np.zeros(pos.shape[0])
    temp = np.zeros(np.sum(ind1))
    temp[inda] = np.pi - LK_R[inda]
    temp[indb] = 2.0*np.pi - LK_R[indb] - pilam[indb]
    temp[indc] = - LK_R[indc] + pilam[indc]

    solid[ind1] = temp
    solid[ind2] = 2*np.pi*abs(pos[ind2,2])*(np.abs(1.0/pos[ind2,2]) - 1.0/np.sqrt(r**2 + pos[ind2,2]**2))
    if (solid <= 0).any():
        print('Solid angle:',solid)
        print('Position:', pos)
        raise NameError('Solid angle smaller than 0')
    return solid

class BES:
    """ Class computing the image of all the fiber
    """

    def __init__(self,input_file, parallel=False):
        """ load all the data from the input file"""
        self.cfg_file = input_file                                           #!
        if not exists(self.cfg_file):
            raise NameError('Config file not found')
        config = psr.ConfigParser()
        config.read(self.cfg_file)
        # variable for knowing if works in parallel or serie
        self.para = parallel                                                 #!

        # the example input file is well commented, look there for more
        # information
        
        # Optics part
        self.pos_lens = json.loads(config.get('Optics','pos_lens'))          #!

        self.rad_ring = json.loads(config.get('Optics','rad_ring'))          #!
        self.rad_lens = json.loads(config.get('Optics','rad_lens'))          #!
        self.inter = json.loads(config.get('Optics','int'))                  #!
        self.Nint = json.loads(config.get('Optics','Nint'))                  #!
        self.Nsol = json.loads(config.get('Optics','Nsol'))                  #!
    
        X = json.loads(config.get('Optics','X'))
        Y = json.loads(config.get('Optics','Y'))
        Z = json.loads(config.get('Optics','Z'))

        self.pos_foc = np.zeros((len(Z),3))                                  #!
        self.pos_foc[:,0] = X
        self.pos_foc[:,1] = Y
        self.pos_foc[:,2] = Z
        self.op_direc = self.pos_foc-self.pos_lens                           #!
        norm_ = np.sqrt(np.sum(self.op_direc**2,axis=1))
        if isinstance(self.rad_ring,float):
            self.rad_ring = self.rad_ring*np.ones(self.pos_foc.shape[0])
        
        # normalize the vector
        self.op_direc[:,0] /= norm_
        self.op_direc[:,1] /= norm_
        self.op_direc[:,2] /= norm_

        self.dist = np.sqrt(np.sum((self.pos_foc - self.pos_lens)**2,axis=1))#!

        self.type_int = config.get('Optics','type_int')                      #!


        # Data part
        self.lifetime = json.loads(config.get('Collisions','t_max'))         #!
        self.lifetime = self.lifetime != 0
        
        self.tau_max = json.loads(config.get('Data','tau_max'))              #!
        self.data_path = config.get('Data','data_path')                      #!
        self.N = json.loads(config.get('Data','N'))                          #!
        start = json.loads(config.get('Data','timestart'))
        end = json.loads(config.get('Data','timeend'))
        timestep = json.loads(config.get('Data','timestep'))
        self.time = np.arange(start,end+1,timestep)                          #!
        self.beam = be.Beam1D(input_file,range(len(self.time)))              #!
        self.compute_limits()      # compute the limits of the mesh
        # position swap due to a difference in the axis        
        grid3D = Grid.Cartesian3D(Xmin=self.Xmin, Xmax=self.Xmax, Ymin=self.Zmin, Ymax=self.Zmax,
                                  Zmin=self.Ymin, Zmax=self.Ymax, NX=self.N[0], NY=self.N[2], NZ=self.N[1])
        xgc_ = xgc.XGC_Loader(self.data_path, grid3D, start, end, timestep,
                              Fluc_Only = False, load_ions=True, equilibrium_mesh = '3D')


        if (self.time != xgc_.time_steps).any():
            raise NameError('Time steps wrong')

        self.beam.set_data(xgc_)
        print 'no check of division by zero'
        self.perp1 = np.zeros(self.pos_foc.shape)
        self.perp1[:,0] = np.ones(self.pos_foc.shape[0])
        self.perp1[:,1] = -self.op_direc[:,2]/self.op_direc[:,1]
        self.perp1[:,2] = np.zeros(self.pos_foc.shape[0])
        norm_ = np.sqrt(np.sum(self.perp1**2,axis=1))
        self.perp1[:,0] /= norm_
        self.perp1[:,1] /= norm_
        self.perp1[:,2] /= norm_

        self.perp2 = np.zeros(self.pos_foc.shape)
        for j in range(self.pos_foc.shape[0]):
            self.perp2[j,:] = np.cross(self.op_direc[j,:],self.perp1[j,:])

    def compute_limits(self, eps=1e-3, dxmin = 0.1, dymin = 0.1, dzmin = 0.5):
        """ Compute the limits of the mesh that should be loaded
            The only limitation comes from the sampling volume
            eps is for adding a small amount to the limits (avoids problems)
        """
        print('need to improve this ')
        w = 0.5*(self.beam.stddev_h + self.beam.stddev_v)
        d = self.inter*w
        center_max = np.zeros((self.pos_foc.shape[0],3))
        center_max[:,0] = self.pos_foc[:,0] + \
                          d*self.op_direc[:,0]
        center_max[:,1] = self.pos_foc[:,1] + \
                          d*self.op_direc[:,1]
        center_max[:,2] = self.pos_foc[:,2] + \
                          d*self.op_direc[:,2]

        center_min = np.zeros((self.pos_foc.shape[0],3))
        center_min[:,0] = self.pos_foc[:,0] - \
                          d*self.op_direc[:,0]
        center_min[:,1] = self.pos_foc[:,1] - \
                          d*self.op_direc[:,1]
        center_min[:,2] = self.pos_foc[:,2] - \
                          d*self.op_direc[:,2]


        w_min = np.zeros(self.pos_foc.shape[0])
        w_max = np.zeros(self.pos_foc.shape[0])

        pos_optical_min = center_min - self.pos_lens
        pos_optical_max = center_max - self.pos_lens

        for k in range(self.pos_foc.shape[0]):
            w_min[k] = self.get_width(pos_optical_min[k,:],k)
            w_max[k] = self.get_width(pos_optical_max[k,:],k)
        # first in X
        self.Xmax = np.max([center_max[:,0] + w_max,
                            center_min[:,0] + w_min])

        self.Xmin = np.min([center_max[:,0] - w_max,
                            center_min[:,0] - w_min])
        # second in Y
        self.Ymax = np.max([center_max[:,1] + w_max,
                            center_min[:,1] + w_min])

        self.Ymin = np.min([center_max[:,1] - w_max,
                            center_min[:,1] - w_min])
        # third in Z
        self.Zmax = np.max([center_max[:,2] + w_max,
                            center_min[:,2] + w_min])

        self.Zmin = np.min([center_max[:,2] - w_max,
                            center_min[:,2] - w_min])


        be_dir = self.beam.direc
       
        l = max(self.beam.speed)*self.tau_max*self.beam.t_max
        
        self.Xmax = max([self.Xmax, self.Xmax - be_dir[0]*l])
        self.Xmin = min([self.Xmin, self.Xmin - be_dir[0]*l])
        self.Ymax = max([self.Ymax, self.Ymax - be_dir[1]*l])
        self.Ymin = min([self.Ymin, self.Ymin - be_dir[1]*l])
        self.Zmax = max([self.Zmax, self.Zmax - be_dir[2]*l])
        self.Zmin = min([self.Zmin, self.Zmin - be_dir[2]*l])


        # try to keep an interval big enough
        dX = self.Xmax-self.Xmin
        dY = self.Ymax-self.Ymin
        dZ = self.Zmax-self.Zmin
        if dX < dxmin:
            dX = dxmin
            av = 0.5*(self.Xmin + self.Xmax)
            self.Xmin = av - 0.5*dX
            self.Xmax = av + 0.5*dX
        if dY < dymin:
            dY = dymin
            av = 0.5*(self.Ymin + self.Ymax)
            self.Ymin = av - 0.5*dY
            self.Ymax = av + 0.5*dY
        if dZ < dzmin:
            dZ = dzmin
            av = 0.5*(self.Zmin + self.Zmax)
            self.Zmin = av - 0.5*dZ
            self.Zmax = av + 0.5*dZ
        self.Xmax += dX*eps
        self.Xmin -= dX*eps
        self.Ymax += dY*eps
        self.Ymin -= dY*eps
        self.Zmax += dZ*eps
        self.Zmin -= dZ*eps

    def get_bes(self):
        """ Compute the image of the turbulence in density
            This function should be the only one used outside the class
            Argument:
            t_  -- list of timesteps
        """
        t_ = range(len(self.time)) # compute all the timestep
        print self.time
        nber_fiber = self.pos_foc.shape[0]
        # loop over all the fiber
        print 'do not take in account the wavelenght'
        print 'only the main component is used'
        print 'Use multiprocessing module'
        if self.para:
            p = mp.Pool()
            I = np.array(p.map(self.intensity_para, range(nber_fiber)))
            I = I[:,:,0]
        else:
            I = np.zeros((nber_fiber,len(t_)))
            for i in range(nber_fiber):
                print('Fiber number: ' + str(i+1) + '/' + str(nber_fiber))
                # compute the light received by each fiber
                I[i,:] = self.intensity(t_,i)[:,0]
        I_av = np.sum(I,axis=1)/len(t_)
        for j in range(len(t_)):
            I[:,j] = (I[:,j]/I_av) - 1.0
        return I
        
    def intensity_para(self,i):
        """ Same as intensity, but have only one argument (all the timesteps
            are computed)
        """
        t_ = range(len(self.time))
        return self.intensity(t_,i)

    def to_cart_coord(self,pos,fiber_nber):
        """ return the cartesian coordinate from the coordinate of the lens
            Attribut:
            pos   --  (X,Y,Z) where Z is along the sightline, X coorespond
                      to perp1, and Y to perp2
        """
        if len(pos.shape) == 1:
            ret = self.pos_lens + self.op_direc[fiber_nber,:]*pos[2]
            ret += self.perp1[fiber_nber,:]*pos[0] + self.perp2[fiber_nber,:]*pos[1]
        else:
            ret = np.zeros(pos.shape)
            ret[:,0] = self.pos_lens[0] + self.op_direc[fiber_nber,0]*pos[:,2]
            ret[:,0] += self.perp1[fiber_nber,0]*pos[:,0] + self.perp2[fiber_nber,0]*pos[:,1]
            ret[:,1] = self.pos_lens[1] + self.op_direc[fiber_nber,1]*pos[:,2]
            ret[:,1] += self.perp1[fiber_nber,1]*pos[:,0] + self.perp2[fiber_nber,1]*pos[:,1]
            ret[:,2] = self.pos_lens[2] + self.op_direc[fiber_nber,2]*pos[:,2]
            ret[:,2] += self.perp1[fiber_nber,2]*pos[:,0] + self.perp2[fiber_nber,2]*pos[:,1]
        return ret

    def get_width(self,pos,fiber_nber):
        """ Return the radius of the light cone at pos (optical coordinate)
            Assume two cones that meet at the focus disk
        """
        
        if len(pos.shape) == 1:
            # distance from the ring
            a = abs(pos[2]-self.dist[fiber_nber])
            a *= (self.rad_lens-self.rad_ring[fiber_nber])/self.dist[fiber_nber]
        else:
            # distance from the ring
            a = abs(pos[:,2]-self.dist[fiber_nber])
            a *= (self.rad_lens-self.rad_ring[fiber_nber])/self.dist[fiber_nber]
        return a + self.rad_ring[fiber_nber]
    
    def check_in(self,pos,fib):
        """ Check if the position (optical coordinate) is inside the first cone
            (if the focus ring matter or not)
            fib is the fiber number
        """
        ret = np.zeros(pos.shape[0], dtype=bool)
        # before the focus point
        ind = pos[:,2] < self.dist[fib]
        ret[ind] = True
        # distance from the focus point along the z-axis
        a = pos[~ind,2]-self.dist[fib]
        # size of the 'ring' scaled to this position
        a = a*(self.rad_ring[fib]-self.rad_lens)/self.dist[fib] + self.rad_ring[fib]
        # distance from the axis
        R = np.sqrt(np.sum(pos[~ind,0:2]**2, axis=1))
        ind1 = a > R
        temp = np.zeros(np.sum(~ind), dtype=bool)
        temp[ind1] = True
        ret[~ind] = temp

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
        nber_comp = self.beam.beam_comp.shape[0]
        I = np.zeros((z.shape[0],len(t_),nber_comp))
        if self.type_int == '2D':
            center = np.zeros((len(z),3))
            center[:,2] = z # define the center of the circle
            r = self.get_width(center,fiber_nber)
            for i,r_ in enumerate(r):
                quad = integ.integration_points(2, 'order10', 'circle', r_)
                pos = np.zeros((quad.pts.shape[0],3))
                pos[:,0] = quad.pts[:,0]
                pos[:,1] = quad.pts[:,1]
                pos[:,2] = z[i]*np.ones(quad.pts.shape[0])
                eps = self.get_emis_from(pos,t_,fiber_nber)
                I[i,:,:] = np.einsum('k,ijk->ij',quad.w,eps)
        elif self.type_int == '1D':
            for i,z_ in enumerate(z):
                pos = np.array([0,0,z_])
                I[i,:,:] = self.get_emis_from(pos[np.newaxis,:],t_,fiber_nber)[:,:,0]
        else:
            raise NameError('This type of integration does not exist')
        return I

    def intensity(self,t_,fiber_nber):
        """ Compute the light received by the fiber #fiber_nber
        """
        # first define the quadrature formula
        quad = integ.integration_points(1,'GL3') # Gauss-Legendre order 3
        nber_comp = self.beam.beam_comp.shape[0]
        I = np.zeros((len(t_),nber_comp))
        # compute the distance from the origin of the beam
        dist = np.dot(self.pos_foc[fiber_nber,:] - self.beam.pos,self.beam.direc)
        width = self.beam.get_width(dist)
        width = 0.5*(width[0] + width[1])*self.inter
        border = np.linspace(-width,width,self.Nint)
        Z = 0.5*(border[:-1] + border[1:])
        ba2 = 0.5*(border[2]-border[1])
        print 'check NaN'
        for z in Z:
            pt = z + ba2*quad.pts + self.dist[fiber_nber]
            light = self.light_from_plane(pt,t_,fiber_nber)
            ind = np.isnan(light)
            # NaN comes from value outside the tokamak => 0 photon
            light[ind] = 0.0
            I += np.einsum('k,kij->ij',quad.w,light)
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
        if self.lifetime:
            eps = self.beam.get_emis_lifetime(x,t_)/(4.0*np.pi)
        else:
            eps = self.beam.get_emis(x,t_)/(4.0*np.pi)
        # now compute the solid angle
        solid = self.get_solid_angle(pos,fiber_nber)
        return eps*solid

    def get_solid_angle(self,pos,fib):
        """ compute the solid angle from the position 
            and for the fiber number (fib)
        """
        test = self.check_in(pos,fib)
        #ind2 = np.where(~test)[0]
        solid = np.zeros(pos.shape[0])


        abcdef = sum(test)
        print abcdef
        # first case
        solid[test] = solid_angle_disk(pos[test,:],self.rad_lens)
        # second case
        # first find the position of the 'intersection' between the lens and the ring
        # define a few constant (look my report for the detail, too much computation)
        # to write them in the comments

        if ((pos[~test,0] == 0) & (pos[~test,1] != 0)).any():
            print ~test
            print pos[~test,:]
            raise NameError('pos[:,0] == 0 gives a division by 0')
        ratio = pos[~test,2]/self.dist[fib]
        f = 1.0/(1.0-ratio)
        A = 0.5*(((pos[~test,0]**2 + pos[~test,1]**2)-(self.rad_lens/f)**2)/ratio + ratio*self.rad_ring[fib]**2)/pos[~test,0]
        B = -pos[~test,1]/pos[~test,0]
        delta = 4*B**2*A**2 - 4*(A**2-self.rad_ring[fib]**2)*(B**2+1)
        ind = (delta > 0) & (~np.isnan(delta)) & (~np.isinf(delta))
        temp = np.zeros(np.sum(~test))
        if ind.any():
            print '2',sum(ind)
            # x1 = plus sign
            delta = np.sqrt(delta[ind])
            x1 = np.zeros((sum(ind),2))
            x1[:,0] = (-2*B[ind]*A[ind] + delta)/(2*(B[ind]**2+1))
            x1[:,1] = A[ind] + B[ind]*x1[:,0]

            x2 = np.zeros((sum(ind),2))
            x2[:,0] = (-2*B[ind]*A[ind] - delta)/(2*(B[ind]**2+1))
            x2[:,1] = A[ind] + B[ind]*x2[:,0]

            print 'pos',pos[~test,:][ind,0:2].T
            print 'comp',x1.T*ratio[ind]
            print 'x1',x1
            print 'comp2',(pos[~test,:][ind,0:2].T-x1.T*ratio[ind])
            print 'ratio',f[ind], ratio[ind]
            y1 = ((pos[~test,:][ind,0:2].T-x1.T*ratio[ind])*f[ind]).T
            y2 = ((pos[~test,:][ind,0:2].T-x2.T*ratio[ind])*f[ind]).T

            print 'YY', np.sum(y1**2,axis=1),np.sum(y2**2,axis=1)
            print x1/self.rad_ring[fib], x2/self.rad_ring[fib]

            print 'y',y1/self.rad_lens,y2/self.rad_lens
            temp[ind] = self.solid_angle_mix_case(pos[~test,:][ind,:],[x1, x2],[y1, y2],fib)
        # second case
        ind = ~ind
        if ind.any():
            print '3',sum(ind)
            q = pos[~test,:][ind,:]
            q[:,2] -= self.dist[fib]
            print 'q',q[:,2]
            temp[ind] = solid_angle_disk(q,self.rad_ring[fib])
        solid[~test] = temp
        print solid
        if (solid < 0).any() or (solid > 4*np.pi).any():
            raise NameError('solid angle smaller than 0 or bigger than 4pi')
        return solid

    def solid_angle_mix_case(self,pos,x,y,fib):
        """ Compute numerically the solid angle for the mixted case
            (where the lens AND the ring limit the size of the solid angle)
            Arguments:
            pos -- position of the emission
            x   -- position of the intersection on the ring
            y   -- position of the intersection on the lens
            fib -- fiber number
        """
        omega = self.solid_angle_seg(pos-np.array([0,0,self.dist[fib]]),x,
                                     self.rad_ring[fib])
        omega +=self.solid_angle_seg(pos,y,self.rad_lens)
        return omega


    def solid_angle_seg(self,pos,x,r):
        """
            Compute the solid angle of a disk without a segment
            Argument:
            pos -- position of the emission
            x   -- position of the intersection
            r   -- radius of the disk
        """
        print 'x is not well computed, there is some value at 1e15'
        x1 = x[0]
        x2 = x[1]
        const = np.abs(pos[:,2])
        theta = np.linspace(0,2*np.pi,self.Nsol)
        quadr = integ.integration_points(1,'GL19') # Gauss-Legendre order 3
        quadt = integ.integration_points(1,'GL19') # Gauss-Legendre order 3

        av = 0.5*(theta[:-1] + theta[1:])
        diff = 0.5*np.diff(theta)
        # first compute the integral of the biggest part (use the diff between the full computation
        # the biggest part if not the good one)
        temp1 = x1-x2

        norm_ = np.sqrt(np.sum(temp1**2,axis=1))
        temp = (temp1.T/norm_).T
        # unit vector perpendicular to the straight line x1->x2 (sign does not matter)
        dxperp = np.zeros(temp.shape)
        dxperp[:,0] = -temp[:,1]
        dxperp[:,1] =  temp[:,0]
        thpts = ((diff[:,np.newaxis]*quadt.pts).T + av).T
        # first check if the radius is full or not
        # project along x1, x2
        co = np.cos(thpts) # x-coord
        si = np.sin(thpts) # y-coord
        
        # can be used for knowing angle where rmax < r
        #proj1 = x1[:,np.newaxis,np.newaxis,0]*co + x1[:,np.newaxis,np.newaxis,1]*si
        #proj2 = x2[:,np.newaxis,np.newaxis,0]*co + x2[:,np.newaxis,np.newaxis,1]*si
        # rmax, will be change for the case where rmax is smaller
        print 'useless computations & should improve memory'
        cospsi = np.einsum('ijk,mi->mjk',np.array([co, si]),dxperp)
        # distance between the center and the x1-x2 line
        d = np.abs(x2[:,1]*x1[:,0]-x1[:,1]*x2[:,0])/norm_
        r_ = abs(d/cospsi.T).T
        r_ = np.minimum(r_,r)
        R = np.zeros((pos.shape[0],av.shape[0],
                      quadt.pts.shape[0],quadr.pts.shape[0],3))
        R[...,2] = (np.ones(R[...,2].T.shape)*pos[:,2]).T
        R[...,1] = (np.ones(R[...,1].T.shape)*pos[:,1]).T + 0.5*(r_*np.sin(thpts))[...,np.newaxis]*(quadr.pts + 1.0)
        R[...,0] = (np.ones(R[...,0].T.shape)*pos[:,1]).T + 0.5*(r_*np.cos(thpts))[...,np.newaxis]*(quadr.pts + 1.0)
        R = 1.0/np.sum(R**2,axis=4)**(3.0/2.0)
        R = np.sum(np.sum(R*quadr.w,axis=3)*quadt.w,axis=2)
        
        omega = np.sum(R,axis=1)
        # assume a constant angle
        omega *= const*diff[0]/4.0

        # check if the position is in the same direction than x1
        # is used for checking if we need to do: omega = \omega_{tot} - \omega_{big}
        ind = np.einsum('ij,ij->i',x1,pos[:,0:2]) < 0
        omega[ind] = solid_angle_disk(pos[ind,:],r)-omega[ind]
        
        print('Still need to do the computation of the full minus the small')
        return omega














class BES_ideal:
    """ Take the output of the simulation and just 
        compute the fluctuation

    """
    def __init__(self,input_file,mesh=False):
        """ load all the data from the input file"""
        self.cfg_file = input_file                                           #!
        if not exists(self.cfg_file):
            raise NameError('Config file not found')
        config = psr.ConfigParser()
        config.read(self.cfg_file)

        self.mesh = mesh

        X = json.loads(config.get('Optics','X'))
        Y = json.loads(config.get('Optics','Y'))
        Z = json.loads(config.get('Optics','Z'))

        self.pos_foc = np.zeros((len(Z),3))                                  #!
        self.pos_foc[:,0] = X
        self.pos_foc[:,1] = Y
        self.pos_foc[:,2] = Z

        # Data part
        self.data_path = config.get('Data','data_path')                      #!
        self.N = json.loads(config.get('Data','N'))                          #!
        start = json.loads(config.get('Data','timestart'))
        end = json.loads(config.get('Data','timeend'))
        timestep = json.loads(config.get('Data','timestep'))
        self.time = np.arange(start,end+1,timestep)                          #!
        self.compute_limits()      # compute the limits of the mesh
        # position swap due to a difference in the axis        
        grid3D = Grid.Cartesian3D(Xmin=self.Xmin, Xmax=self.Xmax, Ymin=self.Zmin, Ymax=self.Zmax,
                                  Zmin=self.Ymin, Zmax=self.Ymax, NX=self.N[0], NY=self.N[2], NZ=self.N[1])
        xgc_ = xgc.XGC_Loader(self.data_path, grid3D, start, end, timestep,
                              Fluc_Only = False, load_ions=True, equilibrium_mesh = '3D')


        self.data = xgc_
        if (self.time != xgc_.time_steps).any():
            raise NameError('Time steps wrong')
        

    def compute_limits(self, eps=1e-3, dxmin = 0.1, dymin = 0.1, dzmin = 0.5):
        """ find max of the focus points """
        # first in X
        self.Xmax = np.max(self.pos_foc[:,0])

        self.Xmin = np.min(self.pos_foc[:,0])
        # second in Y
        self.Ymax = np.max(self.pos_foc[:,1])

        self.Ymin = np.min(self.pos_foc[:,1])
        # third in Z
        self.Zmax = np.max(self.pos_foc[:,2])

        self.Zmin = np.min(self.pos_foc[:,2])

        # try to keep an interval big enough
        dX = self.Xmax-self.Xmin
        dY = self.Ymax-self.Ymin
        dZ = self.Zmax-self.Zmin
        if dX < dxmin:
            dX = dxmin
            av = 0.5*(self.Xmin + self.Xmax)
            self.Xmin = av - 0.5*dX
            self.Xmax = av + 0.5*dX
        if dY < dymin:
            dY = dymin
            av = 0.5*(self.Ymin + self.Ymax)
            self.Ymin = av - 0.5*dY
            self.Ymax = av + 0.5*dY
        if dZ < dzmin:
            dZ = dzmin
            av = 0.5*(self.Zmin + self.Zmax)
            self.Zmin = av - 0.5*dZ
            self.Zmax = av + 0.5*dZ
        self.Xmax += dX*eps
        self.Xmin -= dX*eps
        self.Ymax += dY*eps
        self.Ymin -= dY*eps
        self.Zmax += dZ*eps
        self.Zmin -= dZ*eps


    def get_bes(self):
        """ Compute the image of the turbulence in density
            This function should be the only one used outside the class
        """
        if self.mesh:
            print('Need to edit for special case')
            a = np.linspace(self.Xmin,self.Xmax,100)
            b = np.linspace(self.Zmin,self.Zmax,100)
            a,b = np.meshgrid(a,b)
            a = np.reshape(a,-1)
            b = np.reshape(b,-1)
            self.pos_foc = np.zeros((a.shape[0],3))
            self.pos_foc[:,0] = a
            self.pos_foc[:,2] = b
            

        t_ = range(len(self.time)) # compute all the timestep
        nber_fiber = self.pos_foc.shape[0]
        I = np.zeros((nber_fiber,len(t_)))
        # loop over all the fiber
        ne_int = []
        grid_ = (self.data.grid.Z1D,self.data.grid.Y1D,
                 self.data.grid.X1D)
        for i in range(len(t_)):
            ne_int.append(interpolate.RegularGridInterpolator(
                grid_,self.data.ne_on_grid[0,i,:,:,:]))
            
        a = be.to_other_index(self.pos_foc).T
        for i in range(nber_fiber):
            for j in range(len(t_)):
                # compute the light received by each fiber
                I[i,j] = ne_int[j](a[i,:])
        I_av = np.sum(I,axis=1)/len(t_)
        for j in range(len(t_)):
            I[:,j] = (I[:,j]/I_av) - 1.0
        return I
