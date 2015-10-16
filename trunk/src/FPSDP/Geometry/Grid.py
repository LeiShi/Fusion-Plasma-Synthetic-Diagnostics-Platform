"""Class definitions of grids 
"""
#moudle depends on numpy package
import numpy as np
from abc import ABCMeta, abstractmethod




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

class ExpGrid(Grid):
    """Base class for grids using in loading experimental data, mainly for Cartesian coordinates in laboratory frame.
    """
    __metaclass__ = ABCMeta
    

class Cartesian2D(ExpGrid):
    """Cartesian grids in 2D space. Generally corresponds a toroidal slice, i.e. R-Z plane. Rectangular shape assumed.

    Attributes:
        float ResR: the resolution in R direction. 
        float ResZ: the resolution in Z direction.
        point DownLeft: the coordinates of the down left cornor point, given in (Zmin,Rmin) value pair form.
        point UpRight: the coordinates of the up right cornor point, given in (Zmax,Rmax) form.
        int NR,NZ: The gird number in R,Z directions. Can be specified initially or derived from other parameters.
        2darray R2D: R values on 2D grids. R(0,:) gives the 1D R values.
        2darray Z2D: Z values on 2D grids. R is the fast changing variable so Z2D(:,0) gives the 1D Z values.

        
    """
    def __init__(self, **P):
        """initialize the cartesian grid object.

        If either DownLeft or UpRight is not specified, a GridError exception will be raised.
        Either NR or ResN can be specified. If none or both, a GridError exception will be raised. Same as NZ and ResZ
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
                    self.NR = int ( rangeR/P['ResR'] + 2 ) # make sure the actual resolution is finer than the required one
                    self.ResR = rangeR / self.NR
                else:
                    raise GridError('NR and ResR missing or conflicting, make sure you specify exactly one of them.')
                if ( 'NZ' in P.keys() and not 'ResZ' in P.keys() ):
                    self.NZ = P['NZ']
                    self.ResZ = rangeZ / self.NZ                
                elif ('ResZ' in P.keys() and not 'NZ' in P.keys() ):
                    self.NZ = int ( rangeZ/P['ResZ'] + 2 ) # make sure the actual resolution is finer than the required one
                    self.ResZ = rangeZ / self.NZ
                else:
                    raise GridError('NZ and ResZ missing or conflicting, make sure you specify exactly one of them.')
            else:
                raise GridError("Initializing Grid fails: DownLeft or UpRight not set.")
        except GridError:
            #save for further upgrades, may handle GridError here
            raise
        except:
            print 'Unexpected error in grid initialization! During reading and comprehensing the arguments.'
            raise
        
        #create 1D array for R and Z
        self.R1D = np.linspace(self.Rmin,self.Rmax,self.NR)
        self.Z1D = np.linspace(self.Zmin,self.Zmax,self.NZ)
        #now create the 2darrays for R and Z
        self.Z2D = np.zeros((self.NZ,self.NR)) + self.Z1D[:,np.newaxis]
        self.R2D = np.zeros(self.Z2D.shape) + self.R1D[np.newaxis,:]
             

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

    Attributes:
        float ResX: the resolution in X direction. 
        float ResY: the resolution in Y direction.
        float ResZ: the resolution in Z direction.
        Xmin, Xmax: minimun and maximum value in X
        Ymin, Ymax: minimun and maximun value in Y
        Zmin, Zmax: minimun and maximun value in Z
        int NX,NY,NZ: The gird number in X,Y,Z directions. Can be specified initially or derived from other parameters.
        1darray X1D: 1D X values 
        1darray Y1D: 1D Y values
        1darray Z1D: 1D Z values
        3darray X3D: X values on 3D grids. X3D[0,0,:] gives the 1D X values.
        3darray Y3D: Y values on 3D grids. Y3D[0,:,0] gives the 1D Y values
        3darray Z3D: Z values on 3D grids. Z3D[:,0,0] gives the 1D Z values.

        
    """
    def __init__(self, **P):
        """initialize the cartesian grid object.

        If any min/max value in X/Y/Z is missing, a GridError exception will be raised.
        Either NX or ResX can be specified. If none or both, a GridError exception will be raised. Same in Y/Z direction.
        """
        self._name = '3D Cartesian Grids'
        try:
            if ( 'Xmin' in P.keys() and 'Xmax' in P.keys() and  'Ymin' in P.keys() and 'Ymax' in P.keys() and 'Zmin' in P.keys() and 'Zmax' in P.keys() ):
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
                    self.NX = int ( rangeX/P['ResX'] + 2 ) # make sure the actual resolution is finer than the required one
                    self.ResX = rangeX / self.NX
                else:
                    raise GridError('NX and ResX missing or conflicting, make sure you specify exactly one of them.')
                if ( 'NY' in P.keys() and not 'ResY' in P.keys() ):
                    self.NY = P['NY']
                    if(self.NY>1):
                        self.ResY = rangeY / (self.NY-1)               
                    else:
                        self.ResY = 0
                elif ('ResY' in P.keys() and not 'NY' in P.keys() ):
                    self.NY = int ( rangeY/P['ResY'] + 2 ) # make sure the actual resolution is finer than the required one
                    self.ResY = rangeY / self.NY
                else:
                    raise GridError('NY and ResY missing or conflicting, make sure you specify exactly one of them.')
                if ( 'NZ' in P.keys() and not 'ResZ' in P.keys() ):
                    self.NZ = P['NZ']
                    if(self.NZ>1):
                        self.ResZ = rangeZ / (self.NZ-1)               
                    else:
                        self.ResZ = 0
                elif ('ResZ' in P.keys() and not 'NZ' in P.keys() ):
                    self.NZ = int ( rangeZ/P['ResZ'] + 2 ) # make sure the actual resolution is finer than the required one
                    self.ResZ = rangeZ / self.NZ
                else:
                    raise GridError('NZ and ResZ missing or conflicting, make sure you specify exactly one of them.')
            else:
                raise GridError("Initializing Grid fails: X/Y/Z limits not set.")
        except GridError:
            #save for further upgrades, may handle GridError here
            raise
        except:
            print 'Unexpected error in grid initialization! During reading and comprehensing the arguments.'
            raise
        
        #create 1D array for R and Z
        self.X1D = np.linspace(self.Xmin,self.Xmax,self.NX)
        self.Y1D = np.linspace(self.Ymin,self.Ymax,self.NY)
        self.Z1D = np.linspace(self.Zmin,self.Zmax,self.NZ)
        #now create the 2darrays for R and Z
        zero3D = np.zeros((self.NZ,self.NY,self.NX))
        self.Z3D = zero3D + self.Z1D[:, np.newaxis, np.newaxis]
        self.Y3D = zero3D + self.Y1D[np.newaxis,:,np.newaxis]
        self.X3D = zero3D + self.X1D[np.newaxis,np.newaxis, :]
             


    def ToCylindrical(self):
        """Create the corresponding R-Phi-Z cylindrical coordinates mesh.
        Note that since X corresponds to R, Y to Z(vertical direction), then the positive Phi direction is opposite to positive Z direction. Such that X-Y-Z and R-Phi-Z(vertical) are both right-handed.

        added attributes:
        r3D,z3D,phi3D: 3D arrays. phi3D is in radian,[0,2*pi) .
        """
        try:
            print self.phi3D[0,0,0]
            print 'Cynlindrical mesh already created.'
        except AttributeError:
            self.r3D = np.sqrt(self.X3D**2 + self.Z3D**2)
            self.z3D = self.Y3D
            PHI3D = np.where(self.X3D == 0, -np.pi/2 * np.sign(self.Z3D), np.zeros(self.X3D.shape))
            PHI3D = np.where(self.X3D != 0, np.arctan(-self.Z3D/self.X3D), PHI3D)
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
    Analytic grids are in flux coordinates, for convienently creating analytic equilibrium profile and/or fluctuations.
    In addition to the grid coordinates, geometry is stored in a :py:class:`Geometry` object. Analytic conversion functions are provided to get corresponding Cartesian coordinates for each grid point. 
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, g):
        self.geometry = g
        
    @property
    def geometry(self):
        return self._g
        
    @geometry.setter
    def geometry(self,g):
        self._g = g
        
    @geometry.deler
    def geometry(self):
        del self._g


class 

class path(object):
    """class of the light path, basically just a series of points

    Attributes:
    n: int, number of points on the path
    R: double[n], R coordinates of the points
    Z: double[n], Z coordinates of the points 
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

        Attributes:
        ResS : double, resolution in light path length variable s
        R2D : double[], R coordinates still stored in 2D array, but one dimension is actually not used
        Z2D : double[], Corresponding Z coordinates,
        s : double[], path length coordinates, start with s=0
        N : int[], number of grid points, accumulated in each section
                
    """
    def __init__(self, pth, ResS):
        """initialize with a path object pth, and a given resolution ResS
        """
        self._name = "2D Light Path Grid"
        self.pth = pth
        n = pth.n
        self.ResS = ResS
        self.s = np.empty((n)) #s is the array stores the length of path variable
        self.s[0]=0 # start with s=0
        self.N = np.empty((n)) #N is the array stores the number of grid points 
        self.N[0]=1 # The starting point is considered as 1 grid
        for i in range(1,n):
            self.s[i]=( np.sqrt((pth.R[i]-pth.R[i-1])**2 + (pth.Z[i]-pth.Z[i-1])**2) + self.s[i-1] ) # increase with the length of each section
            self.N[i]=( np.ceil((self.s[i]-self.s[i-1])/ResS)+ self.N[i-1] ) # increase with the number that meet the resolution requirement
        self.R2D = np.empty((1,self.N[n-1]))
        self.Z2D = np.empty((1,self.N[n-1]))
        self.s2D = np.empty((1,self.N[n-1]))
        for i in range(1,n):
            self.R2D[ 0, (self.N[i-1]-1): self.N[i]] = np.linspace(pth.R[i-1],pth.R[i],self.N[i]-self.N[i-1]+1) #fill in the middle points with equal space
            self.Z2D[ 0, (self.N[i-1]-1): self.N[i]] = np.linspace(pth.Z[i-1],pth.Z[i],self.N[i]-self.N[i-1]+1)
            self.s2D[ 0, (self.N[i-1]-1): self.N[i]] = self.s[i-1]+ np.sqrt( (self.R2D[0,(self.N[i-1]-1): self.N[i]] - self.pth.R[i-1])**2 + (self.Z2D[0,(self.N[i-1]-1): self.N[i]] - self.pth.Z[i-1])**2 )

    def __str__(self):
        """display information
        """
        info = self._name + "\n"
        info += "created by path:\n"
        info += "\tnumber of points: "+ str(self.pth.n)+"\n"
        info += "\tR coordinates:\n\t"+ str(self.pth.R)+"\n"
        info += "\tZ coordinates:\n\t"+ str(self.pth.Z)+"\n"
        info += "with resolution in S: "+str(self.ResS)+"\n"
        info += "total length of path: "+str(self.s[self.pth.n-1])+"\n"
        return info




