"""Class definitions of grids 
"""
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np

from .Geometry import Geometry


class GridError(Exception):
    def __init__(self,*p):
        self.args = p

class Grid(object):
    """Base class for all kinds of grids
    """
    __metaclass__ = ABCMeta
    def __init__(self):
        self._name = 'General Grids'
        
    @abstractmethod    
    def __str__(self):
        return self._name
        
    @abstractmethod
    def get_mesh(self):
        raise NotImplemented('Derived classes from Grid must substantiate \
get_mesh method!')

class ExpGrid(Grid):
    """Base class for grids using in loading experimental data, mainly for 
    Cartesian coordinates in laboratory frame.
    """
    __metaclass__ = ABCMeta
    

class Cartesian1D(ExpGrid):
    """Cartesian1D(self, Xmin, Xmax, NX=None, ResX=None)
    1D Cartesian grids
    
    :param float ResX: resolution in X
    :param float Xmin: minimum in X
    :param float Xmax: maximum in X
    :param int NX: Grid point number in X
    
    :raises GridError: if none or both of ``NX`` and ``ResX`` are given
    
    """
    def __init__(self, Xmin, Xmax, NX=None, ResX=None):
        self._name = '1D Cartesian Grids'
        if( NX is None and ResX is None):
            raise GridError('Either NX and ResX must be given!')
        if( NX is not None and ResX is not None):
            raise GridError('Only one of NX and ResX should be given!')
        self.dimension = 1
        if (Xmax < Xmin):
            self.reversed = True
            self.Xmin = Xmax
            self.Xmax = Xmin
        else:
            self.reversed = False
            self.Xmin = Xmin
            self.Xmax = Xmax
        
        if (NX is not None):
            self.NX = NX
            self.ResX = float(self.Xmax-self.Xmin)/NX
            self._X1D_ordered = np.linspace(self.Xmin, self.Xmax, NX)
            self.shape = self._X1D_ordered.shape
        else:
            self.NX = int(np.floor((self.Xmax-self.Xmin)/float(ResX))+2)
            self.ResX = float(self.Xmax-self.Xmin)/(self.NX-1)
            assert np.abs(self.ResX) <= np.abs(ResX)
            self._X1D_ordered = np.linspace(self.Xmin, self.Xmax, self.NX)
            self.shape = self._X1D_ordered.shape
            
    @property
    def X1D(self):
        if self.reversed:
            return self._X1D_ordered[::-1]
        else:
            return self._X1D_ordered
        
    def get_mesh(self):
        return (self.X1D,)
        
    def get_ndmesh(self):
        return (self.X1D,)
        
    def __str__(self):
        info = self._name + '\n'
        info += 'Xmin :' + str(self.Xmin) +'\n'
        info += 'Xmax :' + str(self.Xmax) +'\n'
        info += 'NX,ResX :' + str( (self.NX,self.ResX) ) +'\n'
        return info
            

class FinePatch1D(Cartesian1D):
    """1D grid providing local patches of finer mesh
    
    initialize with the coarse grid range, and fine patch specifics. A finer 
    patch is assumed uniform. Overlap patches must be given in the right order,
    such that a later patch is expected to overwrite former patches.
    
    :param float Xmin, Xmax: range of the coarse mesh
    :param int NX: grid number in coarse mesh
    :param patches: list of patches add on to coarse mesh
    :type patches: list of :py:class:`FPSDP.Geometry.Grid.Cartesian1D` object
    
    Methods:
    
        add_patch(patch, layer=None): 
            insert a new patch into patches list before
            layer location, default to be the last
        remove_patch(layer=None): 
            remove patch from location layer, default last
        get_mesh(): return 1D grid with all patches applied
        get_ndmesh(): same as get_mesh for 1D
    """
    
    def __init__(self, Xmin, Xmax, NX=None, ResX=None, patches=None):
        super(FinePatch1D, self).__init__(Xmin=Xmin, Xmax=Xmax, NX=NX, 
                                          ResX=ResX)
        if patches is None:
            self.patch_list = []
        else:
            self.patch_list = patches
            
    def add_patch(self, patch, layer=None):
        """insert a new patch into patches list after layer location, default 
        to be the last
           
        :param patch: coordinate patch to be added
        :type patch: :py:class:`FPSDP.Geometry.Grid.Cartesian1D` object
        :param int layer: location to add the patch. The patch will be added 
                          **before** the `layer` item. If not given, patch 
                          will be added at the end of patch_list.
            
        """
        assert isinstance(patch, Cartesian1D)
        if layer is None:
            self.patch_list.append(patch)
        else:
            self.patch_list.insert(layer, patch)
            
    def remove_patch(self, patch=None, layer=None):
        """remove a patch from patch_list based on the given value or its 
        location.
        
        :param patch: the patch to be removed. This must be a reference to the
                      exact patch that is stored in patch_list, a new patch 
                      with same parameters won't work, and will raise a 
                      ValueError exception.
        :type patch: :py:class:`Cartesian1D` object
        :param int layer: the location of the patch to be removed. Default to 
                          be the last if neither patch nor layer is given.
                          
        :raise ValueError: if patch is given, but none in patch_list references
                           patch. 
        
        """
        if patch is not None:
            self.patch_list.remove(patch)
        elif layer is None:
            self.patch_list.pop()
        else:
            self.patch_list.pop(layer)
            
    @property
    def X1D(self):
        """1D mesh with all patches applied.
        """
        temp_X1D = self._X1D_ordered
        for p in self.patch_list:
            xmin = p.Xmin
            xmax = p.Xmax
            minidx = np.searchsorted(temp_X1D, xmin)
            maxidx = np.searchsorted(temp_X1D, xmax, side='right')
            temp_X1D= np.delete(temp_X1D, slice(minidx, maxidx))
            temp_X1D = np.insert(temp_X1D, minidx, p._X1D_ordered)
        if self.reversed:
            return temp_X1D[::-1]
        else:
            return temp_X1D  
            
    def patch_info(self):
        info = 'Patches:\n'
        for p in self.patch_list:
            info += str(p)
            info += '\n'
        return info
        
    def __str__(self):
        info = 'FinePatch1D:\n\nMain Mesh:\n   '
        info = super(FinePatch1D, self).__str__()
        info += '\n'
        info += self.patch_info()
        return info
        

class Cartesian2D(ExpGrid):
    """Cartesian grids in 2D space. Generally corresponds a toroidal slice, 
    i.e. R-Z plane. Rectangular shape assumed.

    :param float ResR: the resolution in R direction. 
    :param float ResZ: the resolution in Z direction.
    :param DownLeft: the coordinates of the down left cornor point, given 
                     in (Zmin,Rmin) value pair form.
    :type DownLeft: tuple, or list of float        
    :param UpRight: the coordinates of the up right cornor point, given in 
                    (Zmax,Rmax) form.
    :type UpRight: tuple, or list of float
    :param int NR,NZ: The grid number in R,Z directions. Can be specified 
                      initially or derived from other parameters.
    :raises GridError: if ``DownLeft`` and/or ``UpRight`` are not given
    :raises GridError: if none or both of ``NR``,``ResR`` are given
    :raises GridError: if none or both of ``NZ``, ``ResZ`` are given
    
    Creates the following attributes:
    
    :var 2darray R2D: R values on 2D grids. R(0,:) gives the 1D R values.
    :var 2darray Z2D: Z values on 2D grids. R is the fast changing 
                        variable so Z2D(:,0) gives the 1D Z values.

        
    """
    def __init__(self, **P):
        """initialize the cartesian grid object.

        If either DownLeft or UpRight is not specified, a GridError exception 
        will be raised.
        
        Either NR or ResN can be specified. If none or both, a GridError 
        exception will be raised. Same as NZ and ResZ
        """
        self._name = '2D Cartesian Grids'
        try:
            if ( 'DownLeft' in P.keys() and 'UpRight' in P.keys() ):
                self.DownLeft ,self.UpRight = P['DownLeft'], P['UpRight']
                self.Zmax,self.Rmax = self.UpRight
                self.Zmin,self.Rmin = self.DownLeft
                rangeR = float(self.Rmax - self.Rmin)
                rangeZ = float(self.Zmax - self.Zmin)
                if ( 'NR' in P.keys() and not 'ResR' in P.keys() ):
                    self.NR = P['NR']
                    self.ResR = rangeR / self.NR                
                elif ('ResR' in P.keys() and not 'NR' in P.keys() ):
                    # make sure the actual resolution is finer than the 
                    # required one
                    self.NR = int ( rangeR/P['ResR'] + 2 ) 
                    self.ResR = rangeR / self.NR
                else:
                    raise GridError('NR and ResR missing or conflicting, make \
                    sure you specify exactly one of them.')
                if ( 'NZ' in P.keys() and not 'ResZ' in P.keys() ):
                    self.NZ = P['NZ']
                    self.ResZ = rangeZ / self.NZ                
                elif ('ResZ' in P.keys() and not 'NZ' in P.keys() ):
                    # make sure the actual resolution is finer than the 
                    # required one
                    self.NZ = int ( rangeZ/P['ResZ'] + 2 ) 
                    self.ResZ = rangeZ / self.NZ
                else:
                    raise GridError('NZ and ResZ missing or conflicting, make \
                    sure you specify exactly one of them.')
            else:
                raise GridError("Initializing Grid fails: DownLeft or UpRight \
                not set.")
        except GridError:
            #save for further upgrades, may handle GridError here
            raise
        except:
            print 'Unexpected error in grid initialization! During reading and \
            comprehensing the arguments.'
            raise
        
        #create 1D array for R and Z
        self.R1D = np.linspace(self.Rmin,self.Rmax,self.NR)
        self.Z1D = np.linspace(self.Zmin,self.Zmax,self.NZ)
        #now create the 2darrays for R and Z
        self.Z2D = np.zeros((self.NZ,self.NR)) + self.Z1D[:,np.newaxis]
        self.R2D = np.zeros(self.Z2D.shape) + self.R1D[np.newaxis,:]
        self.shape = self.R2D.shape
        self.dimension = 2
    

    def get_mesh(self):
        return (self.Z1D, self.R1D)  
    
    def get_ndmesh(self):
        return (self.Z2D, self.R2D)

    def __str__(self):
        """returns the key informations of the grids
        """
        info = self._name + '\n'
        info += 'DownLeft :' + str(self.DownLeft) +'\n'
        info += 'UpRight :' + str(self.UpRight) +'\n'
        info += 'NR,ResR :' + str( (self.NR,self.ResR) ) +'\n'
        info += 'NZ,ResZ :' + str( (self.NZ,self.ResZ) ) +'\n'
        return info

class Cartesian3D(ExpGrid):
    """Cartesian grids in 3D space. Rectangular shape assumed.

    :param float ResX: the resolution in X direction. 
    :param float ResY: the resolution in Y direction.
    :param float ResZ: the resolution in Z direction.
    :param Xmin,Xmax: minimun and maximum value in X
    :type Xmin,Xmax: float
    :param Ymin,Ymax: minimun and maximun value in Y
    :type Ymin,Ymax: float        
    :param Zmin,Zmax: minimun and maximun value in Z
    :type Zmin,Zmax: float        
    :param int NX,NY,NZ: The gird number in X,Y,Z directions. Can be 
                         specified initially or derived from other 
                         parameters.
    :raises GridError: if any min/max value in X/Y/Z is missing.
    :raises GridError: Either NX or ResX can be specified. If none or both,
                       a GridError exception will be raised. Same in Y/Z 
                       direction.                     
    
    Creates the following attributes:
    
    :param 1darray X1D: 1D X values 
    :param 1darray Y1D: 1D Y values
    :param 1darray Z1D: 1D Z values
    :param 3darray X3D: X values on 3D grids.
                        X3D[0,0,:] gives the 1D X values
    :param 3darray Y3D: Y values on 3D grids. 
                        Y3D[0,:,0] gives the 1D Y values.
    :param 3darray Z3D: Z values on 3D grids. 
                        Z3D[:,0,0] gives the 1D Z values.
                            
        
        
    """
    def __init__(self, **P):
        """initialize the cartesian grid object.
        
        :param float ResX: the resolution in X direction. 
        :param float ResY: the resolution in Y direction.
        :param float ResZ: the resolution in Z direction.
        :param Xmin,Xmax: minimun and maximum value in X
        :type Xmin,Xmax: float
        :param Ymin,Ymax: minimun and maximun value in Y
        :type Ymin,Ymax: float        
        :param Zmin,Zmax: minimun and maximun value in Z
        :type Zmin,Zmax: float        
        :param int NX,NY,NZ: The gird number in X,Y,Z directions. Can be 
                             specified initially or derived from other
        
        :raises GridError: if any min/max value in X/Y/Z is missing.
        :raises GridError: Either NX or ResX can be specified. If none or both,
                           a GridError exception will be raised. Same in Y/Z 
                           direction. 
        """
        self._name = '3D Cartesian Grids'
        try:
            if ( ('Xmin' in P.keys()) and ('Xmax' in P.keys()) and ('Ymin' in \
                 P.keys()) and ('Ymax' in P.keys()) and ('Zmin' in P.keys()) \
                 and ('Zmax' in P.keys()) ):
                self.Xmin ,self.Xmax = P['Xmin'], P['Xmax']
                self.Ymin ,self.Ymax = P['Ymin'], P['Ymax']
                self.Zmin ,self.Zmax = P['Zmin'], P['Zmax']
                rangeX = float(self.Xmax - self.Xmin)
                rangeY = float(self.Ymax - self.Ymin)
                rangeZ = float(self.Zmax - self.Zmin)
                if ( 'NX' in P.keys() and not 'ResX' in P.keys() ):
                    self.NX = P['NX']
                    if(self.NX>1):
                        self.ResX = rangeX / (self.NX-1)
                    else:
                        self.ResX = 0
                elif ('ResX' in P.keys() and not 'NX' in P.keys() ):
                    # make sure the actual resolution is finer than the 
                    # required one
                    self.NX = int ( rangeX/P['ResX'] + 2 ) 
                    self.ResX = rangeX / self.NX
                else:
                    raise GridError('NX and ResX missing or conflicting, make \
                    sure you specify exactly one of them.')
                if ( 'NY' in P.keys() and not 'ResY' in P.keys() ):
                    self.NY = P['NY']
                    if(self.NY>1):
                        self.ResY = rangeY / (self.NY-1)               
                    else:
                        self.ResY = 0
                elif ('ResY' in P.keys() and not 'NY' in P.keys() ):
                    # make sure the actual resolution is finer than the 
                    # required one
                    self.NY = int ( rangeY/P['ResY'] + 2 ) 
                    self.ResY = rangeY / self.NY
                else:
                    raise GridError('NY and ResY missing or conflicting, make \
                    sure you specify exactly one of them.')
                if ( 'NZ' in P.keys() and not 'ResZ' in P.keys() ):
                    self.NZ = P['NZ']
                    if(self.NZ>1):
                        self.ResZ = rangeZ / (self.NZ-1)               
                    else:
                        self.ResZ = 0
                elif ('ResZ' in P.keys() and not 'NZ' in P.keys() ):
                    # make sure the actual resolution is finer than the 
                    # required one
                    self.NZ = int ( rangeZ/P['ResZ'] + 2 ) 
                    self.ResZ = rangeZ / self.NZ
                else:
                    raise GridError('NZ and ResZ missing or conflicting, make \
                    sure you specify exactly one of them.')
            else:
                raise GridError("Initializing Grid fails: X/Y/Z limits not \
                set.")
        except GridError:
            #save for further upgrades, may handle GridError here
            raise
        except:
            print 'Unexpected error in grid initialization! During reading \
            and comprehensing the arguments.'
            raise
        
        #create 1D array for R and Z
        self.X1D = np.linspace(self.Xmin,self.Xmax,self.NX)
        self.Y1D = np.linspace(self.Ymin,self.Ymax,self.NY)
        self.Z1D = np.linspace(self.Zmin,self.Zmax,self.NZ)
        #now create the 3darrays for X, Y, and Z
        self.shape = (self.NZ, self.NY, self.NX)
        zero3D = np.zeros(self.shape)
        self.Z3D = zero3D + self.Z1D[:, np.newaxis, np.newaxis]
        self.Y3D = zero3D + self.Y1D[np.newaxis,:,np.newaxis]
        self.X3D = zero3D + self.X1D[np.newaxis,np.newaxis, :]
        
        self.dimension = 3
        
    
    def get_mesh(self):
        return (self.Z1D, self.Y1D, self.X1D)
        
    def get_ndmesh(self):
        return (self.Z3D, self.Y3D, self.X3D)


    def ToCylindrical(self):
        """Create the corresponding R-Phi-Z cylindrical coordinates mesh.
        
        Note that since X corresponds to R, Y to Z(vertical direction), then 
        the positive Phi direction is opposite to positive Z direction. Such 
        that X-Y-Z and R-Phi-Z(vertical) are both right-handed.

        Creates attributes:
        :attr r3D,z3D,phi3D: phi3D is in radian,[0,2*pi) .
        :type r3D,z3D,phi3D: 3darray
        """
        try:
            print self.phi3D[0,0,0]
            print 'Cynlindrical mesh already created.'
        except AttributeError:
            self.r3D = np.sqrt(self.X3D**2 + self.Z3D**2)
            self.z3D = self.Y3D
            PHI3D = np.where(self.X3D == 0, -np.pi/2 * np.sign(self.Z3D), 
                             np.zeros(self.X3D.shape))
            PHI3D = np.where(self.X3D != 0, np.arctan(-self.Z3D/self.X3D), 
                             PHI3D)
            PHI3D = np.where(self.X3D < 0, PHI3D+np.pi , PHI3D )
            self.phi3D = np.where(PHI3D < 0, PHI3D+2*np.pi, PHI3D)

        
    def __str__(self):
        """returns the key informations of the grids
        """
        info = self._name + '\n'
        info += 'Xmin,Xmax :' + str((self.Xmin,self.Xmax)) +'\n'
        info += 'Ymin,Ymax :' + str((self.Ymin,self.Ymax)) +'\n'
        info += 'Zmin,Zmax :' + str((self.Zmin,self.Zmax)) +'\n'
        info += 'NX,ResX :' + str( (self.NX,self.ResX) ) +'\n'
        info += 'NY,ResY :' + str( (self.NY,self.ResY) ) +'\n'
        info += 'NZ,ResZ :' + str( (self.NZ,self.ResZ) ) +'\n'
        return info
    

class AnalyticGrid(Grid):
    """Abstract base class for analytic grids. 
    
    Analytic grids are in flux coordinates, for convienently creating analytic 
    equilibrium profile and/or fluctuations.
    
    In addition to the grid coordinates, geometry is stored in a 
    :py:class:`Geometry` object. Analytic conversion functions are provided to 
    get corresponding Cartesian coordinates for each grid point. 
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, g):
        assert isinstance(g, Geometry)
        self.geometry = g
        self._name = 'General Analytic Grid'

        
    @property
    def geometry(self):
        return self._g
        
    @geometry.setter
    def geometry(self,g):
        self._g = g
        
    @geometry.deleter
    def geometry(self):
        del self._g


class AnalyticGrid3D(AnalyticGrid):
# TODO Finish 3D analytic grid class
    """
    """

class path(object):
    """class of the light path, basically just a series of points

    Attributes:
    :param int n: number of points on the path
    :param R: R coordinates of the points
    :type R: 1darray of float with length *n*
    :param Z: Z coordinates of the points
    :type Z: 1darray of floats with length *n*
    """    
    def __init__(self, n=0, R=np.zeros(1), Z=np.zeros(1)):
        self.n = n
        self.R = R
        self.Z = Z
    def __setitem__(self,p2):
        self.n = p2.n
        self.R = np.copy(p2.R)
        self.Z = np.copy(p2.Z)


class Path2D(Grid):
    """ R-Z Grid created based on an light path

        :param pth: specified light path
        :type pth: :py:class:`path` object
        :param float ResS: the required resolution on light path length

        Creates Attributes:
        :param double ResS : resolution in light path length variable s
        :param R2D : R coordinates still stored in 2D array, but one 
                     dimension is shrunk.
        :type R2D: 2darray of floats
        :param Z2D : Z coordinates corresponding to R2D
        :type Z2D: 2darray of floats
        :param s: path length coordinates, start with s=0
        :type s: 1darray of float
        :param N : number of grid points, accumulated in each section
        :param s2D: s values corresponding to R2D and Z2D
        :type s2D: 2darray of float
                
    """
    def __init__(self, pth, ResS):
        """initialize with a path object pth, and a given resolution ResS
        """
        self._name = "2D Light Path Grid"
        self.pth = pth
        n = pth.n
        self.ResS = ResS
        # s is the array stores the length of path variable
        self.s = np.empty((n)) 
        self.s[0]=0 # start with s=0
        
        # N is the array stores the number of grid points
        self.N = np.empty((n),dtype='int')  
        self.N[0]=1 # The starting point is considered as 1 grid
        for i in range(1,n):
            # increase with the length of each section
            self.s[i]=( np.sqrt((pth.R[i]-pth.R[i-1])**2 + \
                                (pth.Z[i]-pth.Z[i-1])**2) + self.s[i-1] )
            # increase with the number that meet the resolution requirement
            self.N[i]=( np.ceil((self.s[i]-self.s[i-1])/ResS)+ self.N[i-1] ) 
        self.R2D = np.empty((1,self.N[n-1]))
        self.Z2D = np.empty((1,self.N[n-1]))
        self.s2D = np.empty((1,self.N[n-1]))
        for i in range(1,n):
            # fill in the middle points with equal space
            self.R2D[ 0, (self.N[i-1]-1): self.N[i]] = \
                 np.linspace(pth.R[i-1],pth.R[i],self.N[i]-self.N[i-1]+1) 
            self.Z2D[ 0, (self.N[i-1]-1): self.N[i]] = \
                 np.linspace(pth.Z[i-1],pth.Z[i],self.N[i]-self.N[i-1]+1)
            self.s2D[ 0, (self.N[i-1]-1): self.N[i]] = \
                 self.s[i-1]+ np.sqrt( (self.R2D[0,(self.N[i-1]-1): self.N[i]]\
                                       - self.pth.R[i-1])**2 + \
                                       (self.Z2D[0,(self.N[i-1]-1): self.N[i]]\
                                       - self.pth.Z[i-1])**2 )
        self.shape = self.R2D.shape
        self.dimension = 2
        
    def get_mesh(self):
        warnings.warn('Path2D doesn\'t have regular mesh. get_mesh will return\
 a tuple containing (Z,R,s) coordinates of points on the path.')
        return (self.Z2D[0,:], self.R2D[0,:], self.s2D[0,:]) 

    def __str__(self):
        """display information
        """
        n=self.pth.n
        info = self._name + "\n"
        info += "created by path:\n"
        info += "\tnumber of points: "+ str(self.pth.n)+"\n"
        info += "\tR coordinates:\n\t"+ str(self.pth.R)+"\n"
        info += "\tZ coordinates:\n\t"+ str(self.pth.Z)+"\n"
        info += "with resolution in S: "+str(self.ResS)+"\n"
        info += "total length of path: "+str(self.s[n-1])+"\n"
        info += "total number of grids:"+str(self.N[n-1])+"\n"
        return info
        
        
# Here are some useful little tools to generate special shaped grids
        
def squarespace(start, end, N, center=0):
    """ Generate a grid from start to end with N points that is uniform after
    taken square root.
    
    :param float start: the start of the mesh, must be non-negative
    :param float end: the end of the mesh, must be non-negative
    :param int N: the total number of mesh points
    :param float center: the relative center where the mesh has largest 
                         density.
    
    :return: the mesh 
    :rtype: 1d array of float
    """
    
    assert (start-center >= 0) and (end-center >= 0)
    sqstart, sqend = np.sqrt([start-center, end-center])
    sqmesh = np.linspace(sqstart, sqend, N)
    return sqmesh*sqmesh + center
    
def cubicspace(start, end, N, center=0):
    """ Generate a grid from start to end with N points that is uniform after
    taken cubic root. 
    
    :param float start: the start of the mesh,
    :param float end: the end of the mesh, 
    :param int N: the total number of mesh points
    :param float center: the relative center where the mesh has largest 
                         density.    
    
    :return: the mesh 
    :rtype: 1d array of float
    """
    start = start - center
    end = end - center
    startsign = np.sign(start)
    endsign = np.sign(end)
    cqstart, cqend = startsign*abs(start)**(1/3.0), endsign*abs(end)**(1/3.0)
    cqmesh = np.linspace(cqstart, cqend, N)
    return cqmesh*cqmesh*cqmesh + center
    
def quadspace(start, end, N, center=0):
    """ Generate a grid from start to end with N points that is uniform after
    taken quadrature root.
    
    :param float start: the start of the mesh, must be non-negative
    :param float end: the end of the mesh, must be non-negative
    :param int N: the total number of mesh points
    :param float center: the relative center where the mesh has largest 
                         density.    
    
    :return: the mesh 
    :rtype: 1d array of float
    """
    start = start - center
    end = end - center
    assert (start >= 0) and (end >= 0)
    quadstart, quadend = start**0.25, end**0.25
    quadmesh = np.linspace(quadstart, quadend, N)
    sqmesh = quadmesh*quadmesh
    return sqmesh*sqmesh + center
    
    




