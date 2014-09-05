"""stand alone driver script for FWR3D.
Basically creating all the needed input files, and put them into a given directory along with links to executables
Runable by it self, also usable as a module in a user written script.
"""

import numpy as np
import os

############################
# user system environment settings. 
# Make sure these variables are set to YOUR running environment values.
###########################

# reflect3d executable location
reflect3d = '/p/lpi/valeo/reflect3d/bin/ifort/O/reflect'

# existing input file location
default_input_path = './inps/'

# directory to put the new input files and link to executable
run_path = './'


###########################
# Parameters for a single run  (all lenghth in cm)
###########################

#*****************
# Output flags
# T for true, F for false
#*****************

#antenna,epsilon,vacuum,paraxial,fullwave output flags
ant_out = '.TRUE.'
eps_out = '.TRUE.'
eps_1d_out = '.TRUE.'
vac_out = '.TRUE.'
para_out = '.TRUE.'
pp_out = '.TRUE.'
fullw_out = '.TRUE.'

#****************
# Geometry Setup 
#****************

# Y&Z mesh (assumed the same in all 3 regions) 
Ymin = -30
Ymax = 30
Zmin = -30
Zmax = 30
NY = 128 # Y grid points, need to be 2**n number for FFT
NZ = 128 # Z grid number, same reason 

# 3 regions in X are:
# vacuum region: from Antenna to the plasma boundary
# paraxial region: from plasma boudary to somewhere near reflecting layer
# full wave region: containing reflecting layer 

# 4 X coordinates needed:

# antenna location 
x_antenna = 160
# plasma boundary (boundary between vacuum and paraxial region)
x_paraxial_vacuum_bnd = 158
# paraxial and fullwave boundary
x_full_wave_paraxial_bnd = 151
# left boundary of full wave region
x_min_full_wave = 130


#Antenna wave frequency(Hz)
ant_freq = 7.5E10

# grid numbers for the inner 2 regions
nx_paraxial = 8

nx_full_wave = int((x_full_wave_paraxial_bnd-x_min_full_wave)*ant_freq/3e9) 

dx_fw = float(x_full_wave_paraxial_bnd-x_min_full_wave)/nx_full_wave

#*******************
# Antenna Parameters
#*******************

#information for loading antenna pattern from file
#code5 file flag, '.TRUE.' or '.FALSE.'
read_code5_data = '.TRUE.'
#code5 file
code5_datafile = 'antenna_pattern.txt'




#information for analytical antenna setup
# Antenna location (cm)
# center location in Y&Z 
ant_center = 0

# beam width (cm)
ant_height = 20 #total width of the injecting light beam

# focal length of the beam (cm)
ant_focal = -400 

# launch angle (radian)
ant_angle = -0.5*np.pi # np.pi is the constant PI stored in numpy module

#*********************
# Full Wave Solver Parameters
#*********************

#time step allowed by stability condition
omega_dt = dx_fw*ant_freq*2*np.pi/6e10
# total simulation time in terms of time for light go through full wave region
nr_crossings = 3
#total time steps calculated from dt and total simulation time
nt = nx_full_wave*nr_crossings*2

#flags to couple with reflect main progrom
read_paraxial = '.TRUE.'
submesh = '.FALSE.'
propagation_medium = '"plasma"'

#source location for full wave solver (x indices of the 2 points in full wave mesh)
ixs = [nx_full_wave*9/10,nx_full_wave*9/10+1]

#total time step for output(Note that NX*NY*NZ*itime*16Byte should not exceed 4GB)
itime = 10
iskip = int(nt/itime)

#********************
# Epsilon Calculation Parameters
#********************

# plasma equilibrium loading type. dataset: loading from files
generator = 'dataset'
# plasma equilibrium file
equilibrium_file = './equilibrium.cdf'
# equilibrium file format
data_file_format = 'leishi'

#fluctuation flag
with_fluctuations = '.TRUE.'
generate_fluctuations = '.TRUE.'
#fluctuation data format
fluctuation_type = 'leishi_3d_dataset'
#fluctuation file
fluctuation_file = 'fluctuation.cdf'

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
#              'schr':'schradi.inp',
              'para':'paraxial.inp',
              'pp':'pparaxial.inp',
              'vac':'vacuum.inp',
              'mod':'model.inp',
              'bc_fft':'BC_fftk.inp',
              'bc_ker':'BC_kernel.inp',
              'fw':'FW_expl.inp'
             }

# dictionary of all file heads:

NML_HEADS = {'ant':'&ANTENNA_NML\n',
             'eps':'&EPSILON_NML\n',
             'geo':"&geometry_spec_nml\nspecification = 'nested' \n/\n&geometry_nested_nml\n",
#             'schr':'&SCHR_NML\nABSORBING_LAYER_X = T\n/\n&SCHRADI_NML\n',
             'para':'&PARAXIAL_NML\n',
             'pp':'&PARAXIAL_NML\n',
             'vac':'&VACUUM_NML\n',
             'mod':'&model_nml\n',
             'bc_fft':'BC_fftk_nml\n',
             'bc_ker':'BC_kernel_nml\n',
             'fw':'&FW_EXPL_NML\n'
             }

NML_ENDS = {'ant':'\n/',
            'eps':'\n/',
            'geo':'\n/',
#            'schr':'\n/',
            'para':'\n/',
            'pp':'\n/',
            'vac':'\n/',
            'mod':'\n/\n&FW_expl_interface_nml\nixs={0},{1}\n/'.format(ixs[0],ixs[1]),
            'bc_fft':'\n/',
            'bc_ker':'\n/',
            'fw':'\n/'
           }

# dictionaries containing all the parameters


ANT_PARA = {}
EPS_PARA = {}
GEO_PARA = {}
FW_PARA = {}
MOD_PARA= {}
PARA_PARA = {}
PP_PARA = {}
VAC_PARA = {}
EMPTY_PARA = {}


def renew_para():

    global ANT_PARA,EPS_PARA,GEO_PARA,FW_PARA,MOD_PARA,PARA_PARA,PP_PARA,VAC_PARA,NML_ENDS

    #renew derived parameters:

    nx_full_wave = int((x_full_wave_paraxial_bnd-x_min_full_wave)*ant_freq/3e9)
    dx_fw = float(x_full_wave_paraxial_bnd-x_min_full_wave)/nx_full_wave
    omega_dt = dx_fw*ant_freq*2*np.pi/6e10
    nt = nx_full_wave*nr_crossings*2
    iskip = int(nt/itime)
    ixs = [nx_full_wave*9/10,nx_full_wave*9/10+1]
    
    NML_ENDS['mod']='\n/\n&FW_expl_interface_nml\nixs={0},{1}\n/'.format(ixs[0],ixs[1])

    
    ANT_PARA = {'read_code5_data' : read_code5_data,
                'code5_datafile' : '"'+code5_datafile+'"',   
                'ANT_CENTER':ant_center,
                'ANT_HEIGHT':ant_height,
                'FOCAL_LENGTH':ant_focal,
                'ANT_LAUNCH_ANGLE':ant_angle,
                'ANT_FREQUENCY':ant_freq,
                'OUTPUT':ant_out
                }
    EPS_PARA = {'DATA_FILE':'"'+equilibrium_file+'"',
                'data_file_format':'"'+data_file_format+'"',
                'generator':'"'+generator+'"',
                'with_fluctuations':with_fluctuations,
                'generate_fluctuations':generate_fluctuations,
                'fluctuation_type':'"'+fluctuation_type+'"',
                'fluctuation_file':'"'+fluctuation_file+'"',
                'POLARIZATION':'"'+polarization+'"',            
                'yz_cut':'0.,0.',
                'OUTPUT':eps_out,
                'OUTPUT_EPS_1D':eps_1d_out
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
#SCHR_PARA = {'NT' : nt,
#             'NR_CROSSINGS':nr_crossings,
#             'OUTPUT':fullw_out
#            }
    FW_PARA = {'read_paraxial' : read_paraxial,
               'submesh' : submesh,
               'propagation_medium':propagation_medium,
               'nt':nt,
               'omega_dt':omega_dt,
               'iskip':iskip,
               'do_output':fullw_out
               }
    
    MOD_PARA ={
        'ANTENNA':'.TRUE.',
        'VACUUM' :'.TRUE.',
        'PARAXIAL':'.TRUE.',
        'FULL_WAVE':'.TRUE.',
        'full_wave_solver':'"explicit"'
        }
    PARA_PARA = {
        'OUTPUT':para_out,
        'OUTPUT_3D_FIELDS':para_out,
        'numerical_method':'"fft"'
        }
    PP_PARA = {
        'THETA':0.5,
        'OUTPUT':pp_out,
        'CDF_PREFIX':'pp_',
        'output_3d_fields':pp_out,
        'numerical_method':'"fft"'
        }
    VAC_PARA = {
        'OUTPUT':vac_out
        }

    
def make_input(ftype, **para):
    """ create .inp files
    ftype: string, see the keys of NML_HEADS dict for different ftype strings
    Receive arbitary keyword arguments, all the keyword-value pairs will be written into *.inp file
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
        f.write(NML_ENDS[ftype])
    

def create_all_input_files():
    """create all input files using parameters defined at the beginning.
    """
    if __name__ == 'main':
        print 'creating all the input files in:'+ run_path
    renew_para()
    make_input('ant',**ANT_PARA)
    make_input('eps',**EPS_PARA)
    make_input('geo',**GEO_PARA)
#    make_input('schr',**SCHR_PARA)
    make_input('fw',**FW_PARA)
    make_input('para',**PARA_PARA)
    make_input('pp',**PP_PARA)
    make_input('vac',**VAC_PARA)
    make_input('mod',**MOD_PARA)
    make_input('bc_fft',**EMPTY_PARA)
    make_input('bc_ker',**EMPTY_PARA)

    if __name__ == 'main':
        print 'input files created.'


    
# run the script if executed from command line.

if __name__ == '__main__':
    create_all_input_files()
   
        
