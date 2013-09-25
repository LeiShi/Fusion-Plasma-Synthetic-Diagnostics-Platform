from distutils.core import setup, Extension

netcdf_dir = '/opt/cray/netcdf/4.2.1.1/pgi/119'
gsl_dir = '/usr/common/usg/gsl/1.15/gnu'
numpy_dir = '/usr/common/usg/python/numpy/1.6.2/lib/python/numpy/core'
mpi_dir = '/opt/cray/mpt/5.6.4/gni/mpich2-pgi/119'

module_Map_C = Extension('Map_Mod_C',
                         sources = ['Mapper_Mod_C.c','esiZ120813.c','supplementary.c'],
                         library_dirs = [gsl_dir+'/lib',netcdf_dir+'/lib'],
                         libraries = ['gsl','gslcblas','netcdf'],
                         include_dirs = [netcdf_dir+'/include',gsl_dir+'/include',numpy_dir+'/include',mpi_dir+'/include'])


setup(name = 'GTS_Interface',
      version = '0.1',
      description = 'Provide functions that read GTS output and create desired quantities on grids.',
      ext_modules = [module_Map_C],
      py_modules=['Coord_Mapper'])
