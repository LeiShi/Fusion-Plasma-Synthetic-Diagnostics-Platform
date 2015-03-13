"""This module contains some useful interpolation methods
"""

import numpy as np
import scipy.interpolate as interpolate

class InterpolationError(Exception):
    def __init__(self,value):
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
    if len(x.shape) == 1:
        # First find the x,y,z coordinate of the corner of the cube
        indx = max(np.where(X < x[0])[0])
        indy = max(np.where(Y < x[1])[0])
        indz = max(np.where(Z < x[2])[0])

        # if outside the box, put the value to fill_value
        if indx == [] or indy == [] or indz == 0\
           or x[0] > X[-1] or x[1] > Y[-1] or x[2] > Z[-1]:
            return fill_value
        else:
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
        for i in range(len(x[:,0])):
            indx = np.where(X <= x[i,0])[0].max()
            indy = np.where(Y <= x[i,1])[0].max()
            indz = np.where(Z <= x[i,2])[0].max()

            if indx == [] or indy == [] or indz == 0\
               or x[i,0] > X[-1] or x[i,1] > Y[-1] or x[i,2] > Z[-1]:
                G[i] = fill_value
            else:
                # relative coordinates
                rx = (x[i,0]-X[indx])/(X[indx+1]-X[indx])
                ry = (x[i,1]-Y[indy])/(Y[indy+1]-Y[indy])
                rz = (x[i,2]-Z[indz])/(Z[indz+1]-Z[indz])
                
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
                G[i] = c0*(1-rz) + c1*rz
        return G
    else:
        raise NameError('Error: wrong shape of the position to interpolate')
