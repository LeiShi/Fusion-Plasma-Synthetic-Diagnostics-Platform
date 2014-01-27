"""This module contains some useful interpolation methods
"""

import numpy as np

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
