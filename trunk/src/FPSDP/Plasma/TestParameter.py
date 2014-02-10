
# create analytical plasma profiles for test use

# module depends on numpy package
import numpy as np

from ..GeneralSettings.UnitSystem import cgs

from ..Geometry import Grid 

# Parameter is a keyword dictionary contains all the pre-defined profile parameters.
# can be modified via method set_parameter(kwargs).
# Note that all the coordinates are in (Z,R) order.
Parameter2D = { 'DownLeft':(-60,200), 'UpRight':(60,260),'NR':200, 'NZ':200, 'ne_0': 2e13, 'Te_0': 1*cgs['keV'], 'B_0': 20000, 'ne_shape': 'Hmode', 'Te_shape': 'Hmode', 'a': 50, 'R_0': 200}


xgc_test ={'DownLeft':(-0.5,1.7),'UpRight':(0.5,2.2),'NR':51,'NZ':51}

#not finished 3D parameter dictionary.
Parameter3D = {'Xmin':1.85,'Xmax':2.0,'Ymin':-0.15,'Ymax':0.15}

# shape table is a dictionary contains the shape parameters
# Do not suggest to change it by outside programs
ShapeTable = {'exp': {'NeDecayScale': 3, 'TeDecayScale':5} , 'Hmode':{'PedWidthT': 0.02,'PedWidthN': 0.02 ,'PedHightT': 0.8, 'PedHightN': 0.7, 'ne_out': 1e-10, 'Te_out': 1e-10}}

def show_parameter2D():
    """Print out the parameters at the moment
    """
    print Parameter2D.keys()
    print Parameter2D

def set_parameter2D( DownLeft=Parameter2D['DownLeft'], UpRight=Parameter2D['UpRight'], NR=Parameter2D['NR'], NZ=Parameter2D['NZ'], ne_0=Parameter2D['ne_0'], Te_0=Parameter2D['Te_0'], B_0=Parameter2D['B_0'], ne_shape=Parameter2D['ne_shape'], Te_shape=Parameter2D['Te_shape'], a=Parameter2D['a'], R_0=Parameter2D['R_0']):
    """A explicit method to change the parameters.

    Although it's possible to directly assign new values to parameter keywords, it may be more comfortable for C programmers to do it this way.
    """
    Parameter2D['ne_0'] = ne_0
    Parameter2D['Te_0'] = Te_0
    Parameter2D['B_0'] = B_0
    Parameter2D['a'] = a
    Parameter2D['R_0'] = R_0
    Parameter2D['ne_shape'] = ne_shape
    Parameter2D['Te_shape'] = Te_shape
    Parameter2D['DownLeft'] = DownLeft
    Parameter2D['UpRight'] = UpRight
    Parameter2D['NR'] = NR
    Parameter2D['NZ'] = NZ

def create_profile2D():
    """Create the profiles and return it in a dictionary structure

    ne, Te, B values on RZ mesh are returned
    """

#the return value is a dictionary
    profile = {}

#extract the plasma shape information
    a= Parameter2D['a']
    R_0= Parameter2D['R_0']
    RZGrid = Grid.Cartesian2D(**Parameter2D)

#
    profile['Grid'] = RZGrid

#extract the density information
    nshp = Parameter2D['ne_shape']
    ne_0 = Parameter2D['ne_0']


#evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
#exponential decay shape
        DecScale= ShapeTable['exp']['NeDecayScale']
        DecLength= a/DecScale
        ne_array= ne_0 * np.exp( -np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))/DecLength )
    elif (nshp == 'Hmode') :
#linear H mode profile
        nped = ne_0 * ShapeTable['Hmode']['PedHightN']
        nout = ne_0 * ShapeTable['Hmode']['ne_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthN'])
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
# linear function connecting axis, top of pedestal and the vacuum
        ne_array = np.select([ a_array <= a_core, a_array >= a, a_array > a_core ],[a_array* (nped - ne_0)/a_core + ne_0 , nout,  (a_array - a) * (nped-nout)/(a_core - a)+ nout])
    profile['ne']= ne_array

#Te profile
    tshp = Parameter2D['Te_shape']
    Te_0 = Parameter2D['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/ShapeTable['exp']['TeDecayScale']
        Te_array= Te_0 * np.exp( -np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))/DecLength )
    elif (tshp == 'Hmode') :
#linear H mode profile
        tped = Te_0 * ShapeTable['Hmode']['PedHightT']
        tout = Te_0 * ShapeTable['Hmode']['Te_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthT'])
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
# linear function connecting axis, top of pedestal and the vacuum
        Te_array = np.select([ a_array <= a_core, a_array >= a, a_array > a_core  ],[a_array* (tped - Te_0)/a_core + Te_0 , tout , (a_array - a) * (tped-tout)/(a_core - a)+ tout])
    profile['Te']= Te_array

# B profile
    B_0 = Parameter2D['B_0']
    B_array= B_0 * R_0/RZGrid.R2D
    profile['B']= B_array
    
    return profile

    
    
    
