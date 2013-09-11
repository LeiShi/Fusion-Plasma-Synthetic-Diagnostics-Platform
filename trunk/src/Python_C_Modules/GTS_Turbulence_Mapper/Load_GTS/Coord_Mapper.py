"""Mapping functions that get values on a prescribed Cartesian coordinates grids from GTS output data files which are in flux coordinates.
"""
from Map_Mod_C import *
import numpy as np

Xmin0, Xmax0, NX0 = 2.0, 2.6, 100
Ymin0, Ymax0, NY0 = -0.6, 0.6, 200
Zmin0, Zmax0, NZ0 = 0.0, 0.0, 1

NBoundary = 1001;

TStart0, TStep0, NT0 = 100, 10, 50

FlucFilePath0 = './Fluctuations/'
EqFileName0 = './esiP.1.5'
NTFileName0 = './NTProfile.cdf'
PHIFileNameStart0 = 'PHI'
PHIDataDir0 = './PHI_FILES'
def set_para(Xmin=Xmin0,Xmax=Xmax0,NX=NX0,
             Ymin=Ymin0,Ymax=Ymax0,NY=NY0,
             Zmin=Zmin0,Zmax=Zmax0,NZ=NZ0,
             NBoundary=NBoundary0,
             TStart=TStart0,TStep=TStep0,NT=NT0,
             FlucFilePath=FlucFilePath0,
             EqFileName = EqFileName0,
             PHIFileNameStart = PHIFileNameStart0,
             PHIDataDir = PHIDataDir0):
    Xmin0 = Xmin
    Xmax0 = Xmax
    NX0 = NX
    Ymin0 = Ymin
    Ymax0 = Ymax
    NY0 = NY
    Zmin0 = Zmin
    Zmax0 = Zmax
    NZ0 = NZ
    NBoundary0 = NBoundary
    TStart0 = TStart
    TStep0 = TStep
    NT0 = NT
    FlucFilePath0 = FlucFilePath
    EqFileName0 = EqFileName
    PHIFileNameStart0 = PHIFileNameStart0
    PHIDataDir0 = PHIDataDir
    set_para_(Xmin=Xmin,Xmax=Xmax,NX=NX,
              Ymin=Ymin,Ymax=Ymax,NY=NY,
              Zmin=Zmin,Zmax=Zmax,NZ=NZ,
              NBoundary=NBoundary,
              TStart=TStart,TStep=TStep,NT=NT,
              FlucFilePath=FlucFilePath,
              EqFileName=EqFileName,
              PHIFileNameStart=PHIFileNameStart,
              PHIDataDir=PHIDataDir)
    show_para_()

def show_para():
    show_para_()

def get_fluctuations_from_GTS(x3d,y3d,z3d,ne,Te,Bm):
    """wrapper for C_function
    
    x3d,y3d,z3d: ndarray with shape (NZ,NY,NX), store each desired cartesian coordinate on every grid points.
    ne: ndarray (NT,NZ,NY,NX), the loaded total ne will be stored in this array,with NT time steps.
    Te,Bm: ndarray (NZ,NY,NX), the loaded equilibrium Te and B_mod will be in these arrays.
    """
    get_GTS_profiles_(x3d,y3d,z3d,ne,Te,Bm)
