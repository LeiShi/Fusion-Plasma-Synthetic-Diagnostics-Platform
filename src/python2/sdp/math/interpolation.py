"""This module contains some useful interpolation methods
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator

class InterpolationError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class OutofBoundError(InterpolationError, ValueError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def linear_3d_3point(X,Y,Z,x,y,tol = 1e-8):
    """3D interpolation method
    Linearly interpolate the value of z for given x,y.
    By using 3 points data, the unknown value of z is assumed on the same plane.
    The method used here is the cross product method. From P(x1,y1,z1),Q(x2,y2,z2),and R(x3,y3,z3), construct 2 vectors on the plane, PQ(x2-x1,y2-y1,z2-z1) and PR(x3-x1,y3-y1,z3-z1). Then do the cross product, PQ*PR = N. This gives the normal vector of the plane. The plane's equation is then 'N dot X = d', where X is an arbitary point and d to be determined. d can be easily gotten from any one of the given points, say P. d = N dot P. Then the equation of the plane is found. The equation can be written as 'ax+by+cz = d', then z can be solved for given x and y.
    
    Arguments:
    x1,y1,z1: coordinates of the first point
    x2,y2,z2: the second point
    x3,y3,z3: the third point
    x,y: the x,y coordinates for the wanted

    return value:
    interpolated z value on given (x,y)
    """
    x1,x2,x3 = X[0],X[1],X[2]
    y1,y2,y3 = Y[0],Y[1],Y[2]
    z0 = np.max(Z)
    z1,z2,z3 = Z[0]/z0,Z[1]/z0,Z[2]/z0


    Nx = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1)
    Ny = (x3-x1)*(z2-z1)-(x2-x1)*(z3-z1)
    Nz = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)

    z_base = (x2-x1)*(y3-y1)

    print Nx,Ny,Nz,z_base

    if(np.absolute(Nz/z_base) <= tol ):
        raise InterpolationError('3 points interpolation failed: given points are on a plane vertical to XY plane, no z value being able to interpolated.')

    d = Nx*x1 + Ny*y1 + Nz*z1
    print d, d-Nx*x-Ny*y

    return (d - Nx*x - Ny*y)/float(Nz)*z0


def trilinear_interp(X,Y,Z,F,x, fill_value=0.0):
    """ Trilinear interpolation (3D) for 1 point on a cubic mesh
    See Wikipedia for a better description than the following:
    First choose a direction and interpolate all the corners along this 
    direction (so 8pts -> 4pts) at the value of the wanted point.
    Choose a second direction and interpolate the 4pts at the wanted point
    (4pts -> 2pts).
    Finish with the interpolation along the last line
    
    Arguments:
    X  -- 1D array containing the X coordinate of F
    Y  -- 1D array containing the Y coordinate of F
    Z  -- 1D array containing the Z coordinate of F
    F  -- 3D array containing the data
    x  -- position (3D) where the interpolation is wanted

    return value:
    interpolated z value on given (x,y)
    """
    raise NameError('Does not work, should use RegularGridInterpolator')
    if len(x.shape) == 1:
        # if outside the box, put the value to fill_value
        if x[0] < X[0] or x[1] < Y[0] or x[2] < Z[0]\
               or x[0] > X[-1] or x[1] > Y[-1] or x[2] > Z[-1]:
            return fill_value
        else:
            # First find the x,y,z coordinate of the corner of the cube
            indx = np.where(X < x[0])[0].max()
            indy = np.where(Y < x[1])[0].max()
            indz = np.where(Z < x[2])[0].max()

            # relative coordinates
            rx = (x[0]-X[indx])/(X[indx+1]-X[indx])
            ry = (x[1]-Y[indy])/(Y[indy+1]-Y[indy])
            rz = (x[2]-Z[indz])/(Z[indz+1]-Z[indz])
            
            # compute the first linear interpolation
            temp = 1-rx
            c00 = F[indx,indy,indz]*temp + F[indx+1,indy,indz]*rx
            c10 = F[indx,indy+1,indz]*temp + F[indx+1,indy+1,indz]*rx
            c01 = F[indx,indy,indz+1]*temp + F[indx+1,indy,indz+1]*rx
            c11 = F[indx,indy+1,indz+1]*temp + F[indx+1,indy+1,indz+1]*rx
            
            # compute the second linear interpolation
            temp = 1-ry
            c0 = c00*temp + c10*ry
            c1 = c01*temp + c11*ry
        
            # compute the last linear interpolation
            return c0*(1-rz) + c1*rz
    elif len(x.shape) == 2:
        """this part is the same that before but with a mesh (not only one point).
           the comments will be only for trick due to the shape of the positions
           abd not on the method (look the first part for them)
        """
        G = np.zeros(len(x[:,0]))
        # First find the x,y,z coordinate of the corner of the cube
        ind = ~((x[:,0] < X[0]) | (x[:,1] < Y[0]) | (x[:,2] < Z[0]) |
                (x[:,0] > X[-1]) | (x[:,1] > Y[-1]) | (x[:,2] > Z[-1]))

        G[~ind] = fill_value
        indx = np.where(X <= x[ind,0])[0].max()
        indy = np.where(Y <= x[ind,1])[0].max()
        indz = np.where(Z <= x[ind,2])[0].max()
                
        # relative coordinates
        rx = (x[ind,0]-X[indx])/(X[indx+1]-X[indx])
        ry = (x[ind,1]-Y[indy])/(Y[indy+1]-Y[indy])
        rz = (x[ind,2]-Z[indz])/(Z[indz+1]-Z[indz])
        
        # compute the first linear interpolation
        temp = 1-rx
        c00 = F[indx,indy,indz]*temp + F[indx+1,indy,indz]*rx
        c10 = F[indx,indy+1,indz]*temp + F[indx+1,indy+1,indz]*rx
        c01 = F[indx,indy,indz+1]*temp + F[indx+1,indy,indz+1]*rx
        c11 = F[indx,indy+1,indz+1]*temp + F[indx+1,indy+1,indz+1]*rx
        
        # compute the second linear interpolation
        temp = 1-ry
        c0 = c00*temp + c10*ry
        c1 = c01*temp + c11*ry
        
        # compute the last linear interpolation
        G[ind] = c0*(1-rz) + c1*rz
        return G
    else:
        raise NameError('Error: wrong shape of the position to interpolate')
        

# BarycentricInterpolator with boundary check
class BoundaryWarnBarycentricInterpolator(BarycentricInterpolator):
    """Barycentric Interpolator with Boundary Check. Based on 
    :py:class:`scipy.interpolate.BarycentricInterpolator`.
    
    The boundary is set as minimun x and maximum x. If called with x outside 
    the available range, a OutofBoundError will be raised.
    
    __init__(xi, yi=None, axis=0, bound_error=True, fill_value=0)
    
    :param xi: x coordinates for interpolation
    :type xi: array of float
    :param yi: Optional, y values on each xi location. If not given, need to be
               provided later using :py:method`set_yi` method.
    :type yi: array of float
    :param int axis: the axis of yi along which the interpolator will be 
                     created.
    :param bool bound_error: If True, out of bound interpolation will result a
                             OutofBoundError. Otherwise fill_value will be used
                             . Default to be True
    :param float fill_value: If bound_error is False, out of bound values will
                             be automatically filled with fill_value.
    
    see :py:class:`scipy.interpolate.BarycentricInterpolator` for further 
    information.                 
    """
    
    def __init__(self, xi, yi=None, axis=0, bound_error=True, fill_value=0):
        
        self._xmin = np.min(xi)
        self._xmax = np.max(xi)
        self._bound_error = bound_error
        self._fill_value = fill_value
        
        super(BoundaryWarnBarycentricInterpolator, self).__init__(xi, yi, axis)
        
            
        
    def __call__(self, x):
        if (self._bound_error):
            if np.any(x < self._xmin) or np.any(x > self._xmax):
                raise OutofBoundError('x out of bound! xmin: {}, xmax: {}'.\
                                       format(self._xmin, self._xmax))
            return super(BoundaryWarnBarycentricInterpolator, self).__call__(x)
        else:
            outbound_idx = np.logical_or(x < self._xmin, x > self._xmax)
            result = np.empty_like(x)
            result[~outbound_idx] = super(BoundaryWarnBarycentricInterpolator, 
                                          self).__call__(x[~outbound_idx]) 
            result[outbound_idx] = self._fill_value
            return result
            

    def add_xi(self, xi, yi=None):
        super(BoundaryWarnBarycentricInterpolator, self).add_xi(xi, yi)
        self._xmin = np.min( [np.min(xi), self._xmin] )
        self._xmax = np.max( [np.max(xi), self._xmax] )
        
    
    def set_yi(self, yi, axis=None):
        yi = np.array(yi)
        if not self._bound_error:
            assert yi.ndim == 1
        super(BoundaryWarnBarycentricInterpolator, self).set_yi(yi, axis)
        

            