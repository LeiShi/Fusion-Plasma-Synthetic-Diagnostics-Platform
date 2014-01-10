"""stand alone driver script for FWR3D.
Basically creating all the needed input files, and put them into a given directory along with links to executables
Runable by it self, also usable as a module in a user written script.
"""

import numpy as np

############################
# user system environment settings. 
# Make sure these variables are set to YOUR running environment values.
###########################

# reflect3d executable location
reflect3d = '/p/lpi/valeo/reflect3d/bin/pgf90/g/bc_kernel'

# existing input file location
default_input_path = './inps/'

# directory to put the new input files and link to executable
run_path = './'


###########################
# Parameters for a single run  (all lenghth in cm)
###########################

#****************
# Geometry Setup 
#****************

# Y&Z mesh (assumed the same in all 3 regions) 
Ymin = -60
Ymax = 60
Zmin = -60
Zmax = 60
NY = 64 # Y grid points, need to be 2**n number for FFT
NZ = 64 # Z grid number, same reason 

# 3 regions in X are:
# vacuum region: from Antenna to the plasma boundary
# paraxial region: from plasma boudary to somewhere near reflecting layer
# full wave region: containing reflecting layer 

# 4 X coordinates needed:

# antenna location 
x_antenna = 1560
# plasma boundary (boundary between vacuum and paraxial region)
x_paraxial_vacuum_bnd = 1500
# paraxial and fullwave boundary
x_full_wave_paraxial_bnd = 1400
# left boundary of full wave region
x_min_full_wave = 1350

# grid numbers for the inner 2 regions
nx_paraxial = 40
nx_full_wave = 50 

#*******************
# Antenna Parameters
#*******************

# Antenna location (cm)
# center location in Y&Z 
ant_center = 0

# beam width (cm)
ant_height = 20 #total width of the injecting light beam

# wave frequency (Hz)
ant_freq = 6.0E10

# focal length of the beam (cm)
ant_focal = -400 

# launch angle (radian)
ant_angle = -0.5*np.pi # np.pi is the constant PI stored in numpy module

#*********************
# Full Wave Solver Parameters
#*********************

# total time step
nt = 10000
# total simulation time, in unit of vacuum crossing time
nr_crossings = 10

#********************
# Epsilon Calculation Parameters
#********************

# plasma data file
plasma_file = './ufile.cdf'

# wave polarization (O or X) 
polarization = 'O'


##############################
# End of the Parameter Setting Up
##############################

##############################
# Functions that make all the input files
##############################

# dictionary of file names:

FILE_NAMES = {'ant':'antenna.inp',
              'eps':'epsilon.inp',
              'geo':'geometry.inp',
              'schr':'schradi.inp'
             }

# dictionary of all file heads:

NML_HEADS = {'ant':'&ANTENNA_NML\n',
             'eps':'&EPSILON_NML\n',
             'geo':"&geometry_spec_nml\nspecification = 'nested' \n/\n&geometry_nested_nml\n",
             'schr':'&SCHR_NML\nABSORBING_LAYER_X = T\n/\n&SCHRADI_NML\n',
             }

# dictionaries containing all the parameters

ANT_PARA = {'ANT_CENTER':ant_center,
            'ANT_HEIGHT':ant_height,
            'FOCAL_LENGTH':ant_focal,
            'ANT_LAUNCH_ANGLE':ant_angle,
            'ANT_FREQUENCY':ant_freq
           }
EPS_PARA = {'DATA_FILE':plasma_file,
            'POLARIZATION':polarization            
           }

GEO_PARA = {'x_min_full_wave':x_min_full_wave,
            'x_full_wave_paraxial_bnd':x_full_wave_paraxial_bnd,
            'x_paraxial_vacuum_bnd':x_paraxial_vacuum_bnd,
            'x_antenna':x_antenna,
            'nx_full_wave':nx_full_wave,
            'nx_paraxial':nx_paraxial,
            'nx_plasma':nx_full_wave + nx_paraxial,
            'nz_overall':NZ,
            'ny_overall':NY,
            'z_limits_overall':str(Zmin)+' , '+str(Zmax),
            'z_limits_full_wave':str(Zmin)+' , '+str(Zmax),
            'z_limits_paraxial':str(Zmin)+' , '+str(Zmax),
            'y_limits_overall':str(Ymin)+' , '+str(Ymax),
            'y_limits_full_wave':str(Ymin)+' , '+str(Ymax),
            'y_limits_paraxial':str(Ymin)+' , '+str(Ymax)            
            }
SCHR_PARA = {'NT' : nt,
             'NR_CROSSINGS':nr_crossings
            }
            
def make_input(ftype, **para):
    """ create .inp files
    ftype: string, see the keys of NML_HEADS dict for different ftype strings
    Receive arbitary keyword arguments, all the keyword-value pairs will be written into antenna.inp file
    use global variable 'run_path' 
    """
    global run_path
    fname = run_path + FILE_NAMES[ftype]
    with open(fname,'w') as f:
        f.write(NML_HEADS[ftype])
        for i in range(len(para.keys())):
            key = para.keys()[i]
            if (i==0):
                f.write(key + '=' + str(para[key]) )
            else:
                f.write(',\n'+ key + '=' + str(para[key]))
        f.write('\n/')
    

def create_all_input_files():
    """create all input files using parameters defined at the beginning.
    """
    print 'creating all the input files in:'+ run_path
    make_input('ant',**ANT_PARA)
    make_input('eps',**EPS_PARA)
    make_input('geo',**GEO_PARA)
    make_input('schr',**SCHR_PARA)
    print 'input files created.'
    
# run the script if executed from command line.

if __name__ == '__main__':
    create_all_input_files()
   
        
