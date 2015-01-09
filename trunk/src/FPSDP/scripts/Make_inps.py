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
reflect3d = '/p/gkp/lshi/reflect3d/bin/ifort/O/reflect'

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
Ymin = 26.5
Ymax = 39.5
Zmin = -6.5
Zmax = 6.5
NY = 128 # Y grid points, need to be 2**n number for FFT
NZ = 32 # Z grid number, same reason 

# 3 regions in X are:
# vacuum region: from Antenna to the plasma boundary
# paraxial region: from plasma boudary to somewhere near reflecting layer
# full wave region: containing reflecting layer 

# 4 X coordinates needed:

# antenna location 
x_antenna = 846.2
# plasma boundary (boundary between vacuum and paraxial region)
x_paraxial_vacuum_bnd = 840
# paraxial and fullwave boundary
x_full_wave_paraxial_bnd = 819
# left boundary of full wave region
x_min_full_wave = 815

#MPI Processes number setup
#Process number in x,y,z dimension:
nproc_x = 4
nproc_y = 4
nproc_z = 2

#Antenna wave frequency(Hz)
ant_freq = 1.5E11

# grid numbers for the inner 2 regions
nx_paraxial = 93

nx_full_wave = 4*int((x_full_wave_paraxial_bnd-x_min_full_wave)*ant_freq/3e9) 

dx_fw = float(x_full_wave_paraxial_bnd-x_min_full_wave)/nx_full_wave

#*******************
# Antenna Parameters
#*******************

#information for loading antenna pattern from file
#code5 file flag, '.TRUE.' or '.FALSE.'
read_code5_data = '.TRUE.'
#code5 file
code5_datafile = 'antenna_150.TXT'

# center location in Y&Z 
ant_center_y = 35
ant_center_z = 0

#information for analytical antenna setup
# Antenna location (cm)
# center location in Y&Z already set

# beam width (cm)
ant_height_y = 0 #total width of the injecting light beam in y
ant_height_z = 0

# focal length of the beam (cm)
ant_focal = 100 

# launch angle (radian)
ant_angle_y = -0.5*np.pi # np.pi is the constant PI stored in numpy module
ant_angle_z = -0.5*np.pi


#modification parameters
# antenna amplitude
ant_amp = 1

# number of images
ant_nimg = 1

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

#absorbing boundary condition parameters
#artificial collision frequency, in the unit of wave real frequency
Nu_ampl = 1
#absorption layer width in x,y,z direction, in the unit of vacuum wave length, symmetric on both ends assumed
Nu_width_x = 0.5
Nu_width_y = 0.5
Nu_width_z = 0.5
#switchs to turn on the absorbing layers
absorb_x_min = '.FALSE.'
absorb_x_max = '.TRUE.'
absorb_y_min = '.TRUE.'
absorb_y_max = '.TRUE.'
absorb_z_min = '.TRUE.'
absorb_z_max = '.TRUE.'

#source turn on time, in unit of wave peroid
src_ton = 1

#source location for full wave solver (x indices of the 2 points in full wave mesh)
ixs = [nx_full_wave*9/10,nx_full_wave*9/10+1]
#location to collect reflected field, default to be one grid outside the source
ix_refl = ixs[1]+1

#total time step for output(Note that NX*NY*NZ*itime*3*16Byte should not exceed 4GB)
itime = 10
iskip = int(nt/itime)

#********************
# Epsilon Calculation Parameters
#********************

# plasma equilibrium loading type. dataset: loading from files
generator = 'dataset'
# plasma equilibrium file
equilibrium_file = './iter_ufile.cdf'
# equilibrium file format: 'default' or 'leishi'
data_file_format = 'default'

#fluctuation flag
with_fluctuations = '.FALSE.'
generate_fluctuations = '.TRUE.'
#fluctuation data format
fluctuation_type = 'leishi_3d_dataset'
#fluctuation file
fluctuation_file = 'fluctuation.cdf'

# wave polarization (O or X) 
polarization = 'X'



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
              'fw':'VEC_FW_expl.inp',
              'mpi':'MPI.inp'
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
             'fw':'&VEC_FW_EXPL_NML\n',
             'mpi':'&MPI_nml\n'
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
            'fw':'\n/',
            'mpi':'\n/'
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
MPI_PARA = {}
EMPTY_PARA = {}



def renew_para():

    global ANT_PARA,EPS_PARA,GEO_PARA,FW_PARA,MOD_PARA,PARA_PARA,PP_PARA,VAC_PARA,MPI_PARA,NML_ENDS

    #renew derived parameters:

    nx_full_wave = 4*int((x_full_wave_paraxial_bnd-x_min_full_wave)*ant_freq/3e9)
    dx_fw = float(x_full_wave_paraxial_bnd-x_min_full_wave)/nx_full_wave
    omega_dt = dx_fw*ant_freq*2*np.pi/6e10
    nt = nx_full_wave*nr_crossings*2
    iskip = int(nt/itime)
    ixs = [nx_full_wave*9/10,nx_full_wave*9/10+1]
    ix_refl = ixs[1]+1
    
    NML_ENDS['mod']='\n/\n&FW_expl_interface_nml\nixs={0},{1}\n ix_refl = {2}\n/'.format(ixs[0],ixs[1],ix_refl)

    
    ANT_PARA = {'read_code5_data' : read_code5_data,
                'code5_datafile' : '"'+code5_datafile+'"',   
                'ANT_CENTER':'{0},{1}'.format(ant_center_y,ant_center_z),
                'ANT_HEIGHT':'{0},{1}'.format(ant_height_y,ant_height_z),
                'FOCAL_LENGTH':ant_focal,
                'ANT_LAUNCH_ANGLE':'{0},{1}'.format(ant_angle_y,ant_angle_z),
                'ANT_FREQUENCY':ant_freq,
                'ANT_AMPLITUDE':ant_amp,
                'ANT_N_IMAGES':ant_nimg,
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
                'yz_cut':'35.,0.',
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
#               'propagation_medium':propagation_medium,
               'nt':nt,
               'omega_dt':omega_dt,
               'iskip':iskip,
               'do_output':fullw_out,
               'Nu_ampl' : Nu_ampl,
               'Nu_width' : '{0},{1},{2}'.format(Nu_width_x,Nu_width_y,Nu_width_z),
               'absorbing_layer': '{0},{1},{2},{3},{4},{5}'.format(absorb_x_min,absorb_x_max,absorb_y_min,absorb_y_max,absorb_z_min,absorb_z_max),
               'src_ton':src_ton
               }
    
    MOD_PARA ={
        'ANTENNA':'.TRUE.',
        'VACUUM' :'.TRUE.',
        'PARAXIAL':'.TRUE.',
        'FULL_WAVE':'.TRUE.',
        'full_wave_solver':'"explicit"',
        'timer': '.TRUE.',
        'detector': '.FALSE.',
        'spectrum': '.FALSE.',
        'zdim_is_unlimited': '.TRUE.',
        'log':'.FALSE.'
        }
    PARA_PARA = {
        'THETA':0.5,
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
    MPI_PARA = {
        'nproc':'{0},{1},{2}'.format(nproc_x,nproc_y,nproc_z)
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
    make_input('mpi',**MPI_PARA)
#    make_input('bc_fft',**EMPTY_PARA)
#    make_input('bc_ker',**EMPTY_PARA)

    if __name__ == 'main':
        print 'input files created.'


    
# run the script if executed from command line.

if __name__ == '__main__':
    create_all_input_files()
   
        
