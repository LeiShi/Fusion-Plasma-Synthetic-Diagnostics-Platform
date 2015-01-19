"""stand alone driver script for FWR3D.
Basically creating all the needed input files, and put them into a given directory along with links to executables
Runable by it self, also usable as a module in a user written script.
"""

import numpy as np
import os

class FWR3D_input_maker:
    """ wrapper class for setting all the parameters and creating proper input files
    """
    def __init__(self):
        self.initialize()
        
    def initialize(self):
        
        ############################
        # user system environment settings. 
        # Make sure these variables are set to YOUR running environment values.
        ###########################

        # reflect3d executable location
        self.reflect3d = '/p/gkp/lshi/reflect3d/bin/ifort/O/reflect'
        self.vec_fw = '/p/gkp/lshi/reflect3d/bin/ifort/O/vec_fw'

        # directory to put the new input files and link to executable
        self.run_path = './'


        ###########################
        # Parameters for a single run  (all lenghth in cm)
        ###########################

        #*****************
        # Output flags
        #*****************

        #antenna,epsilon,vacuum,paraxial,fullwave output flags
        self.ant_out = '.TRUE.'
        self.eps_out = '.TRUE.'
        self.eps_1d_out = '.TRUE.'
        self.vac_out = '.TRUE.'
        self.para_out = '.TRUE.'
        self.pp_out = '.TRUE.'
        self.fullw_out = '.TRUE.'

        #****************
        # Geometry Setup 
        #****************

        # Y&Z mesh (assumed the same in all 3 regions) 
        self.Ymin = -50
        self.Ymax = 50
        self.Zmin = -20
        self.Zmax = 20
        self.NY = 1024 # Y grid points, need to be 2**n number for FFT
        self.NZ = 32 # Z grid number, 2**n for same reason 

        # 3 regions in X are:
        # vacuum region: from Antenna to the plasma boundary
        # paraxial region: from plasma boudary to somewhere near reflecting layer
        # full wave region: containing reflecting layer 

        # 4 X coordinates needed:

        # antenna location 
        self.x_antenna = 160
        # plasma boundary (boundary between vacuum and paraxial region)
        self.x_paraxial_vacuum_bnd = 158
        # paraxial and fullwave boundary
        self.x_full_wave_paraxial_bnd = 150
        # left boundary of full wave region
        self.x_min_full_wave = 135

        #MPI Processes number setup
        #Domain number in x,y,z dimension:
        #BE CAREFUL, the total number of domains (nproc_x*nproc_y*nproc_z) MUST be equal to the total number of the processes in MPI_COMM_WORLD, the latter is normally the total number of processors requested. 
        self.nproc_x = 2
        self.nproc_y = 8
        self.nproc_z = 2

        #Antenna wave frequency(Hz)
        self.ant_freq = 5.5e10

        # grid numbers for the inner 2 regions
        self.nx_paraxial = 16

        self.nx_full_wave = int((self.x_full_wave_paraxial_bnd-self.x_min_full_wave)*self.ant_freq/3e9) 

        self.dx_fw = float(self.x_full_wave_paraxial_bnd-self.x_min_full_wave)/self.nx_full_wave

        #*******************
        # Antenna Parameters
        #*******************

        #information for loading antenna pattern from file
        #code5 file flag, '.TRUE.' or '.FALSE.'
        self.read_code5_data = '.TRUE.'
        #code5 file
        self.code5_datafile = 'antenna_pattern_launch_nstx550.txt'

        # center location in Y&Z
        # Note that if you read the antenna pattern from a code5 file, these coordinates will relocate your whole pattern, keep the center aligned. 
        self.ant_center_y = 0
        self.ant_center_z = 0

        #information for analytical antenna setup
        # Antenna location (cm)
        # center location in Y&Z already set

        # beam width (cm)
        self.ant_height_y = 0 #total width of the injecting light beam in y
        self.ant_height_z = 0

        # focal length of the beam (cm)
        self.ant_focal = 100 

        # launch angle (radian)
        self.ant_angle_y = -0.5*np.pi # np.pi is the constant PI stored in numpy module
        self.ant_angle_z = -0.5*np.pi


        #modification parameters
        # antenna amplitude, rescale the maximum value of the whole incidental field to be ant_amp
        self.ant_amp = 1

        # number of images
        # This parameter is used in case the initial antenna pattern is larger than the calculation area in y-z dimension, and gives non-zero field on the boudary. This parameter will be set to 1 as long as the antenna pattern is well inside the calculation area.
        self.ant_nimg = 1

        #*********************
        # Full Wave Solver Parameters
        #*********************

        #time step allowed by stability condition
        self.omega_dt = self.dx_fw*self.ant_freq*2*np.pi/6e10
        # total simulation time in terms of time for light go through full wave region
        self.nr_crossings = 3
        #total time steps calculated from dt and total simulation time
        self.nt = self.nx_full_wave*self.nr_crossings*2

        #flags to couple with reflect main program
        #read paraxial output from main program or not. Normally True.
        self.read_paraxial = '.TRUE.'
        #use specified mesh or not. If set False, will use the mesh defined in main program. Normally False. 
        self.submesh = '.FALSE.'

        #absorbing boundary condition parameters
        #These parameters controls the numerical boundary condition of the full wave solver. If the absorption switch is turned off, a simple E=0 boundary condition will be used. If absorption is on, then a finite width layer will be placed right inside the boundary, and the field will gradually decay to zero due to artificially added numerical collision damping. This means in the wave equation, an imaginary part of frequency is added, denoted as Nu.  
        #artificial collision damping rate, in the unit of wave real frequency
        self.Nu_ampl = 1
        #absorption layer width in x,y,z direction, in the unit of vacuum wave length, symmetric on both ends assumed
        self.Nu_width_x = 0.5
        self.Nu_width_y = 0.5
        self.Nu_width_z = 0.5
        #switchs to turn on the absorbing layers
        #Normally, all the 6 boundaries should all be set to .TRUE. . However, since the reflection layer SHOULD be inside the calculation area, ideally all the wave power will be reflected back, thus no wave touches x_min boundary. It is then reasonable to set the x_min boundary to be .FALSE., so there will be no artificial damping. When everything is set correctly, this should not significantly change the solution. If something went wrong, a non zero field may appear at x_min boundary. This can be used as a warning. 
        self.absorb_x_min = '.FALSE.'
        self.absorb_x_max = '.TRUE.'
        self.absorb_y_min = '.TRUE.'
        self.absorb_y_max = '.TRUE.'
        self.absorb_z_min = '.TRUE.'
        self.absorb_z_max = '.TRUE.'

        #source turn on time, the source will be turned on gradually to avoid pulse-like perturbations. In the unit of wave peroid
        self.src_ton = 5

        #source location for full wave solver (x indices of the 2 points in full wave mesh)
        self.ixs = [self.nx_full_wave*9/10,self.nx_full_wave*9/10+1]
        #location to collect reflected field, default to be one grid outside the source
        self.ix_refl = self.ixs[1]+1

        #total time step for output(Note that NX*NY*itime*3*16Byte should not exceed 4GB)
        self.itime = 0.9
        self.iskip = int(self.nt/self.itime)

        #********************
        # Epsilon Calculation Parameters
        #********************

        #epsilon model. Default to be 'warm_plasma'. Other options: 'weakly_relativistic'(not complete yet)
        self.model = 'warm_plasma'

        # plasma equilibrium loading type. dataset: loading from files
        self.generator = 'dataset'
        # plasma equilibrium file
        self.equilibrium_file = './equilibrium.cdf'
        # equilibrium file format: 'default'(use 'bpol') or 'leishi'(use 'br' and 'bz')
        self.data_file_format = 'default'

        #fluctuation flag
        self.with_fluctuations = '.FALSE.'

        #fluctuation data format
        self.fluctuation_type = 'leishi_3d_dataset'
        #fluctuation file
        self.fluctuation_file = 'fluctuation1_0.cdf'

        # wave polarization (O or X) 
        self.polarization = 'O'



        ##############################
        # End of the Parameter Setting Up
        ##############################

        ##############################
        # Functions that make all the input files
        ##############################

        # dictionary of file names:

        self.FILE_NAMES = {'ant':'antenna.inp',
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

        self.NML_HEADS = {'ant':'&ANTENNA_NML\n',
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

        self.NML_ENDS = {'ant':'\n/',
                    'eps':'\n/',
                    'geo':'\n/',
        #            'schr':'\n/',
                    'para':'\n/',
                    'pp':'\n/',
                    'vac':'\n/',
                    'mod':'\n/\n&FW_expl_interface_nml\nixs={0},{1}\n/'.format(self.ixs[0],self.ixs[1]),
                    'bc_fft':'\n/',
                    'bc_ker':'\n/',
                    'fw':'\n/',
                    'mpi':'\n/'
                   }

        # dictionaries containing all the parameters


        self.ANT_PARA = {}
        self.EPS_PARA = {}
        self.GEO_PARA = {}
        self.FW_PARA = {}
        self.MOD_PARA= {}
        self.PARA_PARA = {}
        self.PP_PARA = {}
        self.VAC_PARA = {}
        self.MPI_PARA = {}
        self.EMPTY_PARA = {}



    def renew_para(self):

        global ANT_PARA,EPS_PARA,GEO_PARA,FW_PARA,MOD_PARA,PARA_PARA,PP_PARA,VAC_PARA,MPI_PARA,NML_ENDS

        #renew derived parameters:

        self.nx_full_wave = int((self.x_full_wave_paraxial_bnd-self.x_min_full_wave)*self.ant_freq/3e9)
        self.dx_fw = float(self.x_full_wave_paraxial_bnd-self.x_min_full_wave)/self.nx_full_wave
        self.omega_dt = self.dx_fw*self.ant_freq*2*np.pi/6e10
        self.nt = self.nx_full_wave*self.nr_crossings*2
        self.iskip = int(self.nt/self.itime)
        self.ixs = [self.nx_full_wave*9/10,self.nx_full_wave*9/10+1]
        self.ix_refl = self.ixs[1]+1

        self.NML_ENDS['mod']='\n/\n&FW_expl_interface_nml\nixs={0},{1}\n ix_refl = {2}\n/'.format(self.ixs[0],self.ixs[1],self.ix_refl)


        self.ANT_PARA = {'read_code5_data' : self.read_code5_data,
                    'code5_datafile' : '"'+self.code5_datafile+'"',   
                    'ANT_CENTER':'{0},{1}'.format(self.ant_center_y,self.ant_center_z),
                    'ANT_HEIGHT':'{0},{1}'.format(self.ant_height_y,self.ant_height_z),
                    'FOCAL_LENGTH':self.ant_focal,
                    'ANT_LAUNCH_ANGLE':'{0},{1}'.format(self.ant_angle_y,self.ant_angle_z),
                    'ANT_FREQUENCY':self.ant_freq,
                    'ANT_AMPLITUDE':self.ant_amp,
                    'ANT_N_IMAGES':self.ant_nimg,
                    'OUTPUT':self.ant_out
                    }
        self.EPS_PARA = {'DATA_FILE':'"'+self.equilibrium_file+'"',
                    'data_file_format':'"'+self.data_file_format+'"',
                    'generator':'"'+self.generator+'"',
                    'with_fluctuations':self.with_fluctuations,
                    'generate_fluctuations':'.TRUE.',
                    'fluctuation_type':'"'+self.fluctuation_type+'"',
                    'fluctuation_file':'"'+self.fluctuation_file+'"',
                    'POLARIZATION':'"'+self.polarization+'"',            
                    'yz_cut':'35.,0.',
                    'OUTPUT':self.eps_out,
                    'OUTPUT_EPS_1D':self.eps_1d_out
                    }

        self.GEO_PARA = {'x_min_full_wave':self.x_min_full_wave,
                    'x_full_wave_paraxial_bnd':self.x_full_wave_paraxial_bnd,
                    'x_paraxial_vacuum_bnd':self.x_paraxial_vacuum_bnd,
                    'x_antenna':self.x_antenna,
                    'nx_full_wave':self.nx_full_wave,
                    'nx_paraxial':self.nx_paraxial,
                    'nx_plasma':self.nx_full_wave + self.nx_paraxial,
                    'nz_overall':self.NZ,
                    'ny_overall':self.NY,
                    'z_limits_overall':str(self.Zmin)+' , '+str(self.Zmax),
                    'z_limits_full_wave':str(self.Zmin)+' , '+str(self.Zmax),
                    'z_limits_paraxial':str(self.Zmin)+' , '+str(self.Zmax),
                    'y_limits_overall':str(self.Ymin)+' , '+str(self.Ymax),
                    'y_limits_full_wave':str(self.Ymin)+' , '+str(self.Ymax),
                    'y_limits_paraxial':str(self.Ymin)+' , '+str(self.Ymax)            
                    }
    #SCHR_PARA = {'NT' : nt,
    #             'NR_CROSSINGS':nr_crossings,
    #             'OUTPUT':fullw_out
    #            }
        self.FW_PARA = {'read_paraxial' : self.read_paraxial,
                   'submesh' : self.submesh,
    #               'propagation_medium':propagation_medium,
                   'nt':self.nt,
                   'omega_dt':self.omega_dt,
                   'iskip':self.iskip,
                   'do_output':self.fullw_out,
                   'Nu_ampl' : self.Nu_ampl,
                   'Nu_width' : '{0},{1},{2}'.format(self.Nu_width_x,self.Nu_width_y,self.Nu_width_z),
                   'absorbing_layer': '{0},{1},{2},{3},{4},{5}'.format(self.absorb_x_min,self.absorb_x_max,self.absorb_y_min,self.absorb_y_max,self.absorb_z_min,self.absorb_z_max),
                   'src_ton':self.src_ton
                   }

        self.MOD_PARA ={
            'ANTENNA':'.TRUE.',
            'VACUUM' :'.TRUE.',
            'PARAXIAL':'.TRUE.',
            'FULL_WAVE':'.TRUE.',
            'full_wave_solver':'"explicit"',
            'timer': '.TRUE.',
            'detector': '.FALSE.',
            'spectrum': '.FALSE.',
            'zdim_is_unlimited': '.TRUE.',#set z-dimension to be unlimited in cdf_file record, get around with the 4GB array size limit. 
            'log':'.FALSE.'
            }
        self.PARA_PARA = {
            'THETA':0.5,
            'OUTPUT':self.para_out,
            'OUTPUT_3D_FIELDS':self.para_out,
            'numerical_method':'"fft"'
            }
        self.PP_PARA = {
            'THETA':0.5,
            'OUTPUT':self.pp_out,
            'CDF_PREFIX':'pp_',
            'output_3d_fields':self.pp_out,
            'numerical_method':'"fft"'
            }
        self.VAC_PARA = {
            'OUTPUT':self.vac_out
            }
        self.MPI_PARA = {
            'nproc':'{0},{1},{2}'.format(self.nproc_x,self.nproc_y,self.nproc_z)
            }


    def make_input(self,ftype, **para):
        """ create .inp files
        ftype: string, see the keys of NML_HEADS dict for different ftype strings
        Receive arbitary keyword arguments, all the keyword-value pairs will be written into *.inp file
        use global variable 'run_path' 
        """
        global run_path
        fname = self.run_path + self.FILE_NAMES[ftype]
        with open(fname,'w') as f:
            f.write(self.NML_HEADS[ftype])
            for i in range(len(para.keys())):
                key = para.keys()[i]
                if (i==0):
                    f.write(key + '=' + str(para[key]) )
                else:
                    f.write(',\n'+ key + '=' + str(para[key]))
            f.write(self.NML_ENDS[ftype])


    def create_all_input_files(self):
        """create all input files using parameters defined at the beginning.
        """
        if __name__ == 'main':
            print 'creating all the input files in:'+ self.run_path
        self.renew_para()
        self.make_input('ant',**self.ANT_PARA)
        self.make_input('eps',**self.EPS_PARA)
        self.make_input('geo',**self.GEO_PARA)
    #    make_input('schr',**SCHR_PARA)
        self.make_input('fw',**self.FW_PARA)
        self.make_input('para',**self.PARA_PARA)
        self.make_input('pp',**self.PP_PARA)
        self.make_input('vac',**self.VAC_PARA)
        self.make_input('mod',**self.MOD_PARA)
        self.make_input('mpi',**self.MPI_PARA)
    #    make_input('bc_fft',**EMPTY_PARA)
    #    make_input('bc_ker',**EMPTY_PARA)

        if __name__ == 'main':
            print 'input files created.'

    def create_executable_links(self):
        """create links to executables
        """
        import subprocess as subp
        subp.check_call(['ln','-s',self.reflect3d,self.run_path+'reflect_O'])
        subp.check_call(['ln','-s',self.vec_fw,self.run_path+'mvfw_O'])


    
# run the script if executed from command line.

if __name__ == '__main__':
    maker = FWR3D_input_maker()
    maker.create_executable_links()
    maker.create_all_input_files()
   
        
