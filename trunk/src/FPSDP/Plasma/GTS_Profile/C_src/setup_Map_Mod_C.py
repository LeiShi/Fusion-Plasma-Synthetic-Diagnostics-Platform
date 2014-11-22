from distutils.core import setup, Extension
import os

My_Sys = 'NERSC'


if(My_Sys == 'NERSC'):
    netcdf_dir = os.environ['NETCDF_DIR']
    gsl_dir = os.environ['GSL_DIR']
    numpy_dir = os.environ['NUMPY_HOME']+'/lib/python/numpy/core'
elif(My_Sys == 'PPPL'):
    netcdf_dir = os.environ['NETCDF_DIR']
    gsl_dir = os.environ['GSL_DIR']
    numpy_dir = os.environ['NUMPY_HOME']    
    
#mpi_dir=os.environ['MPICH_DIR']
#mpi_dir='/opt/cray/mpt/6.0.2/gni/mpich2-cray/81'
#mpi_dir = '/opt/cray/mpt/5.6.4/gni/mpich2-pgi/119'  #for Hopper

module_Map_C = Extension('Map_Mod_C',
                         sources = ['Mapper_Mod_C.c','esiZ120813.c','supplementary.c'],
                         library_dirs = [gsl_dir+'/lib',netcdf_dir+'/lib'],
                         libraries = ['gsl','gslcblas','netcdf'],
                         include_dirs = [netcdf_dir+'/include',gsl_dir+'/include',numpy_dir+'/include'],
                         extra_compile_args=['-O0'])


setup(name = 'GTS_Profile',
      version = '0.1',
      description = 'Provide functions that read GTS output and create desired quantities on grids.',
      ext_modules = [module_Map_C],
      py_modules=['GTS_Loader'],)
