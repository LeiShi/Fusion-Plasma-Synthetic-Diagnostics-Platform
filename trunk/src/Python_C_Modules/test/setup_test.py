from distutils.core import setup, Extension

module1 = Extension('test_C',
                    sources = ['testmodules.c'],
                    include_dirs = ['/usr/common/usg/python/numpy/1.6.2/lib/python/numpy/core/include'])

setup(name = 'TestPackage',
      version = '0.1',
      description = 'Some Test Extension Modules',
      ext_modules = [module1],
      py_modules=['test'])
