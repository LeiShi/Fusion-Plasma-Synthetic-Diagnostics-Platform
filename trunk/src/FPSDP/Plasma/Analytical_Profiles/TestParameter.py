
""" create analytical plasma profiles for test use


"""
from collections import OrderedDict

import numpy as np
from numpy.random import random

from ...GeneralSettings.UnitSystem import cgs
from ...Geometry import Grid 
from ..PlasmaProfile import ECEI_Profile

# Parameter is a keyword dictionary contains all the pre-defined profile 
# parameters.
# can be modified via method set_parameter(kwargs).
# Note that all the coordinates are in (Z,R) order.
Parameter2D = OrderedDict()
Parameter2D['R_0']=200
Parameter2D['a']=50
Parameter2D['DownLeft']=(-60,150)
Parameter2D['UpRight']=(60,300)
Parameter2D['NR']=201
Parameter2D['NZ']=201
Parameter2D['ne_0']=2e13
Parameter2D['Te_0']=1*cgs['keV']
Parameter2D['B_0']=20000
Parameter2D['ne_shape']='Hmode'
Parameter2D['Te_shape']='Hmode'
Parameter2D['dne_ne']=0.01
Parameter2D['dte_te']=0.01
Parameter2D['dB_B']=0
Parameter2D['timesteps']=[0]

Parameter1D = OrderedDict()
Parameter1D['R_0']=200
Parameter1D['a']=50
Parameter1D['Xmin']=150
Parameter1D['Xmax']=300
Parameter1D['NX']=201
Parameter1D['ne_0']=2e13
Parameter1D['Te_0']=1*cgs['keV']
Parameter1D['B_0']=20000
Parameter1D['ne_shape']='Hmode'
Parameter1D['Te_shape']='Hmode'
Parameter1D['dne_ne']=0.01
Parameter1D['dte_te']=0.01
Parameter1D['dB_B']=0
Parameter1D['timesteps']=[0]
               


xgc_test2D ={'DownLeft':(-0.5,0.9),'UpRight':(0.5,1.6),'NR':101,'NZ':101}
xgc_test3D = {'Xmin':0.9,'Xmax':1.6,'Ymin':-0.5, 'Ymax':0.5, 'Zmin':-0.1, 
              'Zmax':0.1, 'NX':32,'NY':32,'NZ':16}


# TODO finish 3D parameter dictionary.
# Parameter3D = {'Xmin':1.85,'Xmax':2.0,'Ymin':-0.15,'Ymax':0.15}

# shape table is a dictionary contains the shape parameters
# Do not suggest to change it by outside programs
# DecayScale means within a minor radius, it will decay to exponential of which
# power.
ShapeTable = {'exp': {'NeDecayScale': 3, 'TeDecayScale':5} , 
              'Hmode':{'PedWidthT': 0.02,'PedWidthN': 0.02 ,'PedHightT': 0.8, 
                       'PedHightN': 0.7, 'ne_out': 1e-10, 'Te_out': 1e-10}}

def show_parameter2D():
    """Print out the parameters at the moment
    """
    for key,value in Parameter2D.items():
        print '{} : {}'.format(key, value)
    
def show_parameter1D():
    """Print out the parameters at the moment
    """
    for key,value in Parameter1D.items():
        print '{} : {}'.format(key, value) 

def set_parameter2D(DownLeft=Parameter2D['DownLeft'], 
                    UpRight=Parameter2D['UpRight'], 
                    NR=Parameter2D['NR'], 
                    NZ=Parameter2D['NZ'], 
                    ne_0=Parameter2D['ne_0'], 
                    Te_0=Parameter2D['Te_0'], 
                    B_0=Parameter2D['B_0'], 
                    ne_shape=Parameter2D['ne_shape'], 
                    Te_shape=Parameter2D['Te_shape'], 
                    a=Parameter2D['a'], 
                    R_0=Parameter2D['R_0'],
                    dne_ne=Parameter2D['dne_ne'],
                    dte_te=Parameter2D['dte_te'],
                    dB_B=Parameter2D['dB_B'],
                    timesteps=Parameter2D['timesteps']):
    """A explicit method to change the parameters.

    Although it's possible to directly assign new values to parameter keywords,
    it may be more comfortable for C programmers to do it this way.
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
    Parameter2D['dne_ne'] = dne_ne
    Parameter2D['dte_te'] = dte_te
    Parameter2D['dB_B'] = dB_B
    Parameter2D['timesteps'] = timesteps
    
def set_parameter1D(Xmin=Parameter1D['Xmin'], 
                    Xmax=Parameter1D['Xmax'], 
                    NX=Parameter1D['NX'],
                    ne_0=Parameter1D['ne_0'], 
                    Te_0=Parameter1D['Te_0'], 
                    B_0=Parameter1D['B_0'], 
                    ne_shape=Parameter1D['ne_shape'], 
                    Te_shape=Parameter1D['Te_shape'], 
                    a=Parameter1D['a'], 
                    R_0=Parameter1D['R_0'],
                    dne_ne=Parameter1D['dne_ne'],
                    dte_te=Parameter1D['dte_te'],
                    dB_B=Parameter1D['dB_B'],
                    timesteps=Parameter1D['timesteps']):
    """A explicit method to change the parameters.

    Although it's possible to directly assign new values to parameter keywords,
    it may be more comfortable for C programmers to do it this way.
    """
    Parameter1D['ne_0'] = ne_0
    Parameter1D['Te_0'] = Te_0
    Parameter1D['B_0'] = B_0
    Parameter1D['a'] = a
    Parameter1D['R_0'] = R_0
    Parameter1D['ne_shape'] = ne_shape
    Parameter1D['Te_shape'] = Te_shape
    Parameter1D['Xmin'] = Xmin
    Parameter1D['Xmax'] = Xmax
    Parameter1D['NX'] = NX
    Parameter1D['dne_ne'] = dne_ne
    Parameter1D['dte_te'] = dte_te
    Parameter1D['dB_B'] = dB_B
    Parameter1D['timesteps'] = timesteps

def create_profile1D(random_fluctuation=False, fluc_level=None):
    """Create the profiles and return it in a dictionary structure

    ne, Te, B values on RZ mesh are returned
    """

    # the return value will be generated by this dictionary
    profile = {}

    # extract the plasma shape information
    a= Parameter1D['a']
    R_0= Parameter1D['R_0']
    XGrid = Grid.Cartesian1D(Parameter1D['Xmin'], Parameter1D['Xmax'], 
                             Parameter1D['NX'])

    profile['Grid'] = XGrid

    # extract the density information
    nshp = Parameter1D['ne_shape']
    ne_0 = Parameter1D['ne_0']


    # evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
        # exponential decay shape
        DecScale= ShapeTable['exp']['NeDecayScale']
        DecLength= a/DecScale
        ne_array= ne_0 * np.exp(-np.abs(XGrid.X1D-R_0)/DecLength )
    elif (nshp == 'Hmode') :
        # linear H mode profile
        nped = ne_0 * ShapeTable['Hmode']['PedHightN']
        nout = ne_0 * ShapeTable['Hmode']['ne_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthN'])
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, top of pedestal and the vacuum
        ne_array = np.select([a_array <= a_core, a_array >= a, 
                              a_array > a_core ], 
                             [a_array* (nped - ne_0)/a_core + ne_0 , nout,  
                              (a_array - a) * (nped-nout)/(a_core - a)+ nout])
    profile['ne']= ne_array

    # Te profile
    tshp = Parameter1D['Te_shape']
    Te_0 = Parameter1D['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/ShapeTable['exp']['TeDecayScale']
        Te_array= Te_0 * np.exp(-np.abs(XGrid.X1D-R_0)/DecLength )
    elif (tshp == 'Hmode') :
        # linear H mode profile
        tped = Te_0 * ShapeTable['Hmode']['PedHightT']
        tout = Te_0 * ShapeTable['Hmode']['Te_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthT'])
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, top of pedestal and the vacuum
        Te_array = np.select([a_array<=a_core, a_array>=a, a_array>a_core],
                             [a_array*(tped-Te_0)/a_core + Te_0, tout,
                              (a_array-a)*(tped-tout)/(a_core-a)+ tout])
    profile['Te']= Te_array

    # B profile
    B_0 = Parameter1D['B_0']
    B_array= B_0 * R_0/XGrid.X1D
    profile['B']= B_array
    
    
    if random_fluctuation:
        
        time = Parameter1D['timesteps']
        d_shape = [len(time), XGrid.shape[0]]
        if fluc_level is None:
            dne = 2*Parameter1D['dne_ne']*profile['ne']*(random(d_shape)-0.5)
            dTe = 2*Parameter1D['dte_te']*profile['Te']*(random(d_shape)-0.5)
            dB = 2*Parameter1D['dB_B']*profile['B']*(random(d_shape)-0.5)
        else:
            dne = 2*fluc_level*profile['ne']*(random(d_shape)-0.5)
            dTe = 2*fluc_level*profile['Te']*(random(d_shape)-0.5)
            dB = 2*fluc_level*profile['B']*(random(d_shape)-0.5)
        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                        profile['B'], time, dne, dTe, dTe, dB)
    else:
        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                        profile['B'])

def create_profile2D(random_fluctuation=False, fluc_level=None):
    """Create the profiles and return it in a dictionary structure

    ne, Te, B values on RZ mesh are returned
    """

    # the return value will be generated by this dictionary
    profile = {}

    # extract the plasma shape information
    a= Parameter2D['a']
    R_0= Parameter2D['R_0']
    RZGrid = Grid.Cartesian2D(**Parameter2D)

    profile['Grid'] = RZGrid

    # extract the density information
    nshp = Parameter2D['ne_shape']
    ne_0 = Parameter2D['ne_0']


    # evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
        # exponential decay shape
        DecScale= ShapeTable['exp']['NeDecayScale']
        DecLength= a/DecScale
        ne_array= ne_0 * np.exp(-np.sqrt(((RZGrid.R2D-R_0)**2 +RZGrid.Z2D**2))\
                  /DecLength )
    elif (nshp == 'Hmode') :
        # linear H mode profile
        nped = ne_0 * ShapeTable['Hmode']['PedHightN']
        nout = ne_0 * ShapeTable['Hmode']['ne_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthN'])
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, top of pedestal and the vacuum
        ne_array = np.select([a_array <= a_core, a_array >= a, 
                              a_array > a_core ], 
                             [a_array* (nped - ne_0)/a_core + ne_0 , nout,  
                              (a_array - a) * (nped-nout)/(a_core - a)+ nout])
    profile['ne']= ne_array

    # Te profile
    tshp = Parameter2D['Te_shape']
    Te_0 = Parameter2D['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/ShapeTable['exp']['TeDecayScale']
        Te_array= Te_0 * np.exp(-np.sqrt(((RZGrid.R2D-R_0)**2 +RZGrid.Z2D**2))\
                  /DecLength )
    elif (tshp == 'Hmode') :
        # linear H mode profile
        tped = Te_0 * ShapeTable['Hmode']['PedHightT']
        tout = Te_0 * ShapeTable['Hmode']['Te_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthT'])
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, top of pedestal and the vacuum
        Te_array = np.select([a_array<=a_core, a_array>=a, a_array>a_core],
                             [a_array*(tped-Te_0)/a_core + Te_0, tout,
                              (a_array-a)*(tped-tout)/(a_core-a)+ tout])
    profile['Te']= Te_array

    # B profile
    B_0 = Parameter2D['B_0']
    B_array= B_0 * R_0/RZGrid.R2D
    profile['B']= B_array
    
    
    if random_fluctuation:
        time = Parameter2D['timesteps']
        d_shape = [len(time)]
        d_shape.extend([i for i in RZGrid.R2D.shape])
        if fluc_level is None:
            dne = 2*Parameter2D['dne_ne']*profile['ne']*(random(d_shape)-0.5)
            dTe = 2*Parameter2D['dte_te']*profile['Te']*(random(d_shape)-0.5)
            dB = 2*Parameter2D['dB_B']*profile['B']*(random(d_shape)-0.5)
        else:
            dne = 2*fluc_level*profile['ne']*(random(d_shape)-0.5)
            dTe = 2*fluc_level*profile['Te']*(random(d_shape)-0.5)
            dB = 2*fluc_level*profile['B']*(random(d_shape)-0.5)
        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                        profile['B'], time, dne, dTe, dTe, dB)
    else:
        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                        profile['B'])


class PlasmaModelError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Model:
    """base class for all plasma models

    Attributes:
        _name:The name of the model
        _type:The type of the model, 'Geometry','Equilibrium', or 'Fluctuation'
        _description: A short description explains the basic idea of the model

    Method:
        info(): prints info about the model
    
    """
    def __init__(self):
        
        #Attributes:
        self._name = 'Model_Base'
        self._type = 'Base'
        self._description = 'This is a base model object.'

    def info(self):
        print('name:{0}\ntype:{1}\n{2}').format(self._name,self._type,
                                                self._description)
        
        
        
    



class TestPlasmaCreator:
    """ A Class that contains all the plasma models, and creates plasma profile
    objects. 

    Existing Models
    ===============
    
    Shape of Magnetic Geometry:
        Concentric : all flux surfaces are concentric circles
        Shafranov_Shifted: flux surfaces are circles that have been shifted 
        outwards according to Shafranov shift
    Equilibrium:
        Exponential_Decay: profiles are exponentially decaying radially
        Hmode_linear: profiles are slowly decreasing in the core region, and 
                      rapidly decreasing in the pedestal region. Both 
                      are linear in radial flux coordinate.
        Hmode_tanh: profiles are in a shape of tanh function, where the half 
                    height location corresponds the separatrix.

    Fluctuations:
        Single_Mode: add a single mode to a certain region. specify the mode 
                     frequency, amplitude, and mode numbers
        Turbulence: add turbulent structure to a certain region. specify 
                    auto-correlation lengths and time, and fluctuation level.
        
    """

    

    
    
    
