# create analytical plasma profiles for test use

# module depends on numpy package
import numpy as np

# Parameter is a keyword dictionary contains all the pre-defined profile parameters.
# can be modified via method set_parameter(kwargs).
Parameter = { 'RZ_field': [(2,0.6), (2.6,-0.6)],'RZ_dimensions': [100,200],'ne_0': 1e20, 'Te_0': 10, 'B_0': 0.5, 'ne_shape': 'exp', 'Te_shape': 'exp', 'a': 0.6, 'R_0': 2.0}

# shape table is a dictionary contains the shape parameters
# Do not suggest to change it by outside programs
ShapeTable = {'exp': {'NeDecayScale': 3, 'TeDecayScale':5} }

def show_parameter():
    """
        Print out the parameters at the moment
    """
    print Parameter.keys()
    print Parameter

def set_parameter( RZ_field=Parameter['RZ_field'], RZ_dimensions=Parameter['RZ_dimensions'], ne_0=Parameter['ne_0'], Te_0=Parameter['Te_0'], B_0=Parameter['B_0'], ne_shape=Parameter['ne_shape'], Te_shape=Parameter['Te_shape'], a=Parameter['a'], R_0=Parameter['R_0']):
    """
        A explicit method to change the parameters.

        Although it's possible to directly assign new values to parameter keywords, it may be more comfortable for C programmers to do it this way.
    """
    Parameter['ne_0'] = ne_0
    Parameter['Te_0'] = Te_0
    Parameter['B_0'] = B_0
    Parameter['a'] = a
    Parameter['R_0'] = R_0
    Parameter['ne_shape'] = ne_shape
    Parameter['Te_shape'] = Te_shape
    Parameter['RZ_field'] = RZ_field
    Parameter['RZ_dimensions'] = RZ_dimensions

def create_profile():
    """
    Create the profiles and return it in a dictionary structure

    ne, Te, B, and RZ coordinates are returned
    """

#the return value is a dictionary
    profile = {}

#extract the coordinates information
    a= Parameter['a']
    R_0= Parameter['R_0']
    NR= Parameter['RZ_dimensions'][0]
    NZ= Parameter['RZ_dimensions'][1]
    Rmin= Parameter['RZ_field'][0][0]
    Rmax= Parameter['RZ_field'][1][0]
    Zmax= Parameter['RZ_field'][0][1]
    Zmin= Parameter['RZ_field'][1][1]

#construct the output coordinates field. Note that the fast changing index is in Z direction!
    field = np.zeros((2,NR,NZ)) 
    R= np.linspace(Rmin,Rmax,NR)
    field[0] += R[:,np.newaxis]
    Z= np.linspace(Zmin,Zmax,NZ)
    field[1] += Z[np.newaxis,:]
    profile['field'] = field

#extract the density information
    nshp = Parameter['ne_shape']
    ne_0 = Parameter['ne_0']


#evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
#exponential decay shape
        DecScale= ShapeTable['exp']['NeDecayScale']
        DecLength= a/DecScale
        R= field[0,:,:]
        Z= field[1,:,:]
        ne_array= ne_0 * np.exp( -np.sqrt(((R-R_0)**2+Z**2))/DecLength )
    profile['ne']= ne_array

#Te profile
    tshp = Parameter['Te_shape']
    Te_0 = Parameter['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/ShapeTable['exp']['TeDecayScale']
        R= field[0,:,:]
        Z= field[1,:,:]
        Te_array= Te_0 * np.exp( -np.sqrt(((R-R_0)**2+Z**2))/DecLength )
    profile['Te']= Te_array

# B profile
    B_0 = Parameter['B_0']
    R= field[0,:,:]
    Z= field[1,:,:]
    B_array= B_0 * R_0/R
    profile['B']= B_array
    
    return profile

    
    
    
