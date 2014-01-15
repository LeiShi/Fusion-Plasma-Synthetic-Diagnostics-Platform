"""Mapping functions that get values on a prescribed Cartesian coordinates grids from GTS output data files which are in flux coordinates.
"""
import Map_Mod_C as mmc
import numpy as np
from FPSDP.Geometry import Grid

Xmin0, Xmax0, NX0 = 2.0, 2.6, 101
Ymin0, Ymax0, NY0 = -0.6, 0.6, 201
Zmin0, Zmax0, NZ0 = 0.0, 0.0, 1

NBoundary0 = 1001

TStart0, TStep0, NT0 = 100, 10, 50

Fluc_Amplification0 = 50

FlucFilePath0 = './Fluctuations/'
EqFileName0 = './ESI_EQFILE'
NTFileName0 = './NTProfile.cdf'
PHIFileNameStart0 = 'PHI.'
PHIDataDir0 = './PHI_FILES/'
def set_para(Xmin=Xmin0,Xmax=Xmax0,NX=NX0,
             Ymin=Ymin0,Ymax=Ymax0,NY=NY0,
             Zmin=Zmin0,Zmax=Zmax0,NZ=NZ0,
             NBoundary=NBoundary0,
             TStart=TStart0,TStep=TStep0,NT=NT0,
             Fluc_Amplification = Fluc_Amplification0,
             FlucFilePath=FlucFilePath0,
             EqFileName = EqFileName0,
             PHIFileNameStart = PHIFileNameStart0,
             PHIDataDir = PHIDataDir0):
    """Set Loading Parameters:
    Xmin,Xmax,Ymin,Ymax,Zmin,Zmax: double; the range of desired cartesian coordinates, in meter. X in major R direction, Y the vertical direction, and Z the direction perpendicular to both X and Y.
    NX,NY,NZ: int; The grid point number in 3 dimensions.
    TStart: int; Starting time of the sampling series, in simulation record step counts.
    TStep: int; The interval between two sample points, in unit of simulation record step counts.
    NT: C-int; The total number of sampling.
    NBoundary: int; The total number of grid points resolving the plasma last closed flux surface. Normally not important.
    FlucFilePath: string; directory where to store the output fluctuation files
    EqFileName: string; filename of the equalibrium file, either absolute or relative path.
    PHIFileNameStart: string; The first 3 letters of the record file, usually PHI
    PHIDataDir: string; the directory where the PHI data files are stored.
    """
    global Xmin0,Xmax0,NX0,Ymin0,Ymax0,NY0,Zmin0,Zmax0,NZ0,NBoundary0,TStart0,TStep0,NT0,Fluc_Amplification0,FlucFilePath0,EqFileName0,PHIFileNameStart0,PHIDataDir0
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
    Fluc_Amplification0 = Fluc_Amplification
    FlucFilePath0 = FlucFilePath
    EqFileName0 = EqFileName
    PHIFileNameStart0 = PHIFileNameStart
    PHIDataDir0 = PHIDataDir
    mmc.set_para_(Xmin=Xmin,Xmax=Xmax,NX=NX,
              Ymin=Ymin,Ymax=Ymax,NY=NY,
              Zmin=Zmin,Zmax=Zmax,NZ=NZ,
              NBOUNDARY=NBoundary,
              TStart=TStart,TStep=TStep,NT=NT,
              Fluc_Amplification=Fluc_Amplification,
              FlucFilePath=FlucFilePath,
              EqFileName=EqFileName,
              PHIFileNameStart=PHIFileNameStart,
              PHIDataDir=PHIDataDir)
    mmc.show_para_()

def show_para():
    mmc.show_para_()

def make_grid():
    """create 3D Cartesian grid using X/Y/Z parameters set. 
    """
    return Grid.Cartesian3D(Xmin = Xmin0,Xmax=Xmax0, Ymin=Ymin0, Ymax=Ymax0, Zmin=Zmin0, NX=NX0, NY=NY0,NZ=NZ0)

def get_fluctuations_from_GTS(x3d,y3d,z3d,ne,Te,Bm):
    """wrapper for C_function
    
    x3d,y3d,z3d: ndarray with shape (NZ,NY,NX), store each desired cartesian coordinate on every grid points.
    ne: ndarray (NT,NZ,NY,NX), the loaded total ne will be stored in this array,with NT time steps.
    Te,Bm: ndarray (NZ,NY,NX), the loaded equilibrium Te and B_mod will be in these arrays.
    """
    mmc.get_GTS_profiles_(x3d,y3d,z3d,ne,Te,Bm)
