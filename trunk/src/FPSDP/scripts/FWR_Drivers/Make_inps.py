"""stand alone driver script for FWR3D.
Basically creating all the needed input files, and put them into a given directory along with links to executables
Runable by it self, also usable as a module in a user written script.
"""

import numpy as np
import os

# an IO package takes care of fortran style namelists
from FPSDP.IO.f90nml.namelist import NmlDict

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
        self.vec_fw = '/p/gkp/lshi/reflect3d/bin/ifort/O/mpi_vec_fw'

        # directory to put the new input files and link to executable
        self.run_path = './'


        ###########################
        # Parameters for a single run  (all lenghth in cm)
        ###########################

        #*****************
        # Output flags
        #*****************

        #antenna,epsilon,vacuum,paraxial,fullwave output flags
        self.ant_out = True
        self.eps_out = True
        self.eps_1d_out = True
        self.vac_out = True
        self.para_out = True
        self.pp_out = True
        self.fullw_out = True

        #****************
        # Geometry Setup 
        #****************

        # Y&Z mesh (assumed the same in all 3 regions) 
        self.Ymin = -5
        self.Ymax = 5
        self.Zmin = -5
        self.Zmax = 5
        self.NY = 32 # Y grid points, need to be 2**n number for FFT
        self.NZ = 32 # Z grid number, 2**n for same reason 

        # 3 regions in X are:
        # vacuum region: from Antenna to the plasma boundary
        # paraxial region: from plasma boudary to somewhere near reflecting layer
        # full wave region: containing reflecting layer 

        # 4 X coordinates needed:

        # antenna location 
        self.x_antenna = 120
        # plasma boundary (boundary between vacuum and paraxial region)
        self.x_paraxial_vacuum_bnd = 110
        # paraxial and fullwave boundary
        self.x_full_wave_paraxial_bnd = 90
        # left boundary of full wave region
        self.x_min_full_wave = 70

        #MPI Processes number setup
        #Domain number in x,y,z dimension:
        #BE CAREFUL, the total number of domains (nproc_x*nproc_y*nproc_z) MUST be equal to the total number of the processes in MPI_COMM_WORLD, the latter is normally the total number of processors requested. 
        self.nproc_x = 8
        self.nproc_y = 1
        self.nproc_z = 1

        #Antenna wave frequency(Hz)
        self.ant_freq = 1.40e11

        # grid numbers for the inner 2 regions
        self.nx_paraxial = 32

        # vacuum wave length resolution in full wave grids: set how many full wave grid points in one vacuum wave length. Need more than 4 to resolve a wave pattern. Normally set to 10.
        self.nx_wavelength = 10
        # formula for derived quantities(will be evaluated in method update_para):
        # full wave total grid number in x direction, set to have 10 points with in one vacuum wave length
        # self.nx_full_wave = int((self.x_full_wave_paraxial_bnd-self.x_min_full_wave)*self.ant_freq/3e10 * self.nx_wavelength) 

        # full wave grid step size in x direction, equals total full-wave region length divided by total grid number. Should be very close to 1/10 of vacuum wave length 
        # self.dx_fw = float(self.x_full_wave_paraxial_bnd-self.x_min_full_wave)/self.nx_full_wave

        #*******************
        # Antenna Parameters
        #*******************

        #information for loading antenna pattern from file
        #code5 file flag, '.TRUE.' or '.FALSE.'
        self.read_code5_data = '.TRUE.'
        #code5 file
        self.code5_datafile = 'antenna_pattern_launch_1400.txt'

        # center location in Y&Z
        # Note that if you read the antenna pattern from a code5 file, these coordinates will relocate your whole pattern, keep the center aligned. 
        self.ant_center_y = 0
        self.ant_center_z = 0

        #information for analytical antenna setup
        # Antenna location (cm)
        # center location in Y&Z already set

        # beam width (cm)
        self.ant_height_y = 2.7 #total width of the injecting light beam in y
        self.ant_height_z = 1.8

        # focal length of the beam (cm)
        self.ant_focal = -100 

        # launch angle (radian)
        self.ant_angle_y = -0.5017*np.pi # np.pi is the constant PI stored in numpy module
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

        #courant condition number: time step should be longer than half of the time for wave passing through a spatial grid size. For FWR3D, stability requires more strict condition. Suggested value is 0.25. If see numerical instability, try change this to a smaller number.
        self.courant_number = 0.25

        #time step allowed by stability condition: formula only, evaluated in method "update_para"
        #self.omega_dt = self.dx_fw*self.ant_freq*2*np.pi/3e10*self.courant_number
        # total simulation time in terms of time for light go through full wave region
        self.nr_crossings = 3

        #total time steps calculated from dt and total simulation time: formula only,evaluated in "update_para"
        #self.nt = self.nx_full_wave*self.nr_crossings*2

        #flags to couple with reflect main program
        #read paraxial output from main program or not. Normally True.
        self.read_paraxial = True
        #use specified mesh or not. If set False, will use the mesh defined in main program. Normally False. 
        self.submesh = False

        #absorbing boundary condition parameters
        #These parameters controls the numerical boundary condition of the full wave solver. If the absorption switch is turned off, a simple E=0 boundary condition will be used. If absorption is on, then a finite width layer will be placed right inside the boundary, and the field will gradually decay to zero due to artificially added numerical collision damping. This means in the wave equation, an imaginary part of frequency is added, denoted as Nu.  
        #artificial collision damping rate, in the unit of wave real frequency
        self.Nu_ampl = 1
        #absorption layer width in x,y,z direction, in the unit of vacuum wave length, symmetric on both ends assumed
        self.Nu_width_x = 0.5
        self.Nu_width_y = 0.5
        self.Nu_width_z = 0.5
        #switchs to turn on the absorbing layers
        #Normally, all the 6 boundaries should all be set to True . However, since the reflection layer SHOULD be inside the calculation area, ideally all the wave power will be reflected back, thus no wave touches x_min boundary. It is then reasonable to set the x_min boundary to be False, so there will be no artificial damping. When everything is set correctly, this should not significantly change the solution. If something went wrong, a non zero field may appear at x_min boundary. This can be used as a warning. 
        self.absorb_x_min = False
        self.absorb_x_max = True
        self.absorb_y_min = True
        self.absorb_y_max = True
        self.absorb_z_min = True
        self.absorb_z_max = True

        #source turn on time, the source will be turned on gradually to avoid pulse-like perturbations. In the unit of wave peroid
        self.src_ton = 5

        #source location for full wave solver
        #in FWR3D, source is set inside the full wave region, close to the paraxial boundary. Incidental wave paterns obtained by paraxial solution will be specified on two closely located planes, and a equivalent source will be generated there based on this information. We can control the source location with the following two parameters:
        # source location (value in range [0,1) ) specifies the inner source plane location, the value equals the length from left boundary of full wave region to the source location divided by total length of full wave region.
        self.source_location = 0.9
        # the outer (closer to paraxial region) source plane is specified by the difference of indices between itself and the inner plane,normally 1 is fine
        self.outer_source_dnx = 1

        # the indices of the two plane in full wave x grid can then be calculated as follows:(formula only, evaluated in "update_para")
        # self.ixs = [int(self.nx_full_wave*self.source_location),int(self.nx_full_wave*self.source_location) + self.outer_source_dnx]

        #location to collect reflected field, specified by indices difference to outer source plane
        self.refl_dnx = 1

        #the index of reflection collection location is then calculated as follows (formula only, evaluated in "update_para")
        # self.ix_refl = self.ixs[1]+self.refl_dnx

        #total time step for output(Note that NX*NY*itime*3*16Byte should not exceed 4GB)
        self.itime = 1

        #skip steps is then calculated as follows:
        #self.iskip = int(self.nt/self.itime)

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
        self.with_fluctuations = False

        #fluctuation data format
        self.fluctuation_type = 'leishi_3d_dataset'
        #fluctuation file
        self.fluctuation_file = 'fluctuation1_0.cdf'

        # wave polarization (O or X) 
        self.polarization = 'O'

	# The (y,z) coordinates of main light path. Incidental wave is assumed mainly along x direction. The central ray is then assumed mainly along x direction inspite of possible refraction and defraction effects of plasma. Epsilon is fully calculated along x at the specified (y,z) coordinates. Then on each y,z plane, delta_epsilon is calculated to the first order in dn/n. The total epsilon is then the sum of the delta_epsilon on the (x,y,z) location and the main ray epsilon at the x value. 
	self.yz_cut = [0,0]  

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
                      'para':'paraxial.inp',
                      'pp':'pparaxial.inp',
                      'vac':'vacuum.inp',
                      'mod':'model.inp',
                      'bc_fft':'BC_fftk.inp',
                      'bc_ker':'BC_kernel.inp',
                      'fw':'VEC_FW_expl.inp',
                      'mpi':'MPI.inp'
                     }

        # dictionary for all the namelist objects contained in each file,  
        self.NML_Dicts = {'ant':NmlDict(ANTENNA_NML={}),
                     'eps':NmlDict(EPSILON_NML={}),
                     'geo':NmlDict(geometry_nested_nml={},geometry_spec_nml={}),
                             #specification = nested
                     'para':NmlDict(PARAXIAL_NML={}),
                     'pp':NmlDict(PARAXIAL_NML={}),
                     'vac':NmlDict(VACUUM_NML={}),
                     'mod':NmlDict(FW_expl_interface_nml = {},model_nml={}),
                     'bc_fft':NmlDict(BC_fftk_nml={}),
                     'bc_ker':NmlDict(BC_kernel_nml={}),
                     'fw':NmlDict(VEC_FW_EXPL_NML={}),
                     'mpi':NmlDict(MPI_nml={})
                     }

        # define names for easy access of each namelist dictionaries

        self.ANT_NML = self.NML_Dicts['ant']
        self.EPS_NML = self.NML_Dicts['eps']
        self.GEO_NML = self.NML_Dicts['geo']
        self.PARA_NML = self.NML_Dicts['para']
        self.PP_NML = self.NML_Dicts['pp']
        self.VAC_NML = self.NML_Dicts['vac']
        self.MOD_NML = self.NML_Dicts['mod']
        self.BCFFT_NML = self.NML_Dicts['bc_fft']
        self.BCKER_NML = self.NML_Dicts['bc_ker']
        self.FW_NML = self.NML_Dicts['fw']
        self.MPI_NML = self.NML_Dicts['mpi']
        


    def update_para(self):

        #calculate useful parameters:
        # full wave total grid points
        self.nx_full_wave = int((self.x_full_wave_paraxial_bnd-self.x_min_full_wave)*self.ant_freq/3e10 * self.nx_wavelength)
        # grid size in full wave region
        self.dx_fw = float(self.x_full_wave_paraxial_bnd-self.x_min_full_wave)/self.nx_full_wave
        # time step size
        self.omega_dt = self.dx_fw*self.ant_freq*2*np.pi/6e10/2
        # total time
        self.nt = self.nx_full_wave*self.nr_crossings*2
        # output skipping time 
        self.iskip = int(self.nt/self.itime)
        # source locations
        self.ixs = [int(self.nx_full_wave*self.source_location),int(self.nx_full_wave*self.source_location) + self.outer_source_dnx]
        #reflection collection index
        self.ix_refl = self.ixs[1]+self.refl_dnx


        self.ANT_NML['ANTENNA_NML'] = NmlDict({'read_code5_data' : self.read_code5_data,
                    'code5_datafile' : self.code5_datafile,   
                    'ANT_CENTER':[self.ant_center_y,self.ant_center_z],
                    'ANT_HEIGHT':[self.ant_height_y,self.ant_height_z],
                    'FOCAL_LENGTH':self.ant_focal,
                    'ANT_LAUNCH_ANGLE':[self.ant_angle_y,self.ant_angle_z],
                    'ANT_FREQUENCY':self.ant_freq,
                    'ANT_AMPLITUDE':self.ant_amp,
                    'ANT_N_IMAGES':self.ant_nimg,
                    'OUTPUT':self.ant_out
                    })
        self.EPS_NML['EPSILON_NML'] = NmlDict({'DATA_FILE':self.equilibrium_file,
                    'data_file_format':self.data_file_format,
                    'generator':self.generator,
                    'with_fluctuations':self.with_fluctuations,
                    'generate_fluctuations':True,
                    'fluctuation_type':self.fluctuation_type,
                    'fluctuation_file':self.fluctuation_file,
                    'POLARIZATION':self.polarization,            
                    'yz_cut':self.yz_cut,
                    'OUTPUT':self.eps_out,
                    'OUTPUT_EPS_1D':self.eps_1d_out
                    })

	self.GEO_NML['geometry_spec_nml'] = NmlDict({'specification':'nested'})

        self.GEO_NML['geometry_nested_nml'] = NmlDict({'x_min_full_wave':self.x_min_full_wave,
                    'x_full_wave_paraxial_bnd':self.x_full_wave_paraxial_bnd,
                    'x_paraxial_vacuum_bnd':self.x_paraxial_vacuum_bnd,
                    'x_antenna':self.x_antenna,
                    'nx_full_wave':self.nx_full_wave,
                    'nx_paraxial':self.nx_paraxial,
                    'nx_plasma':self.nx_full_wave + self.nx_paraxial,
                    'nz_overall':self.NZ,
                    'ny_overall':self.NY,
                    'z_limits_overall':[self.Zmin ,self.Zmax],
                    'z_limits_full_wave':[self.Zmin ,self.Zmax],
                    'z_limits_paraxial':[self.Zmin, self.Zmax],
                    'y_limits_overall':[self.Ymin , self.Ymax],
                    'y_limits_full_wave':[self.Ymin ,self.Ymax],
                    'y_limits_paraxial':[self.Ymin , self.Ymax]            
                    })

        self.FW_NML['VEC_FW_EXPL_NML'] = NmlDict({'read_paraxial' : self.read_paraxial,
                   'submesh' : self.submesh,
    #               'propagation_medium':propagation_medium,
                   'nt':self.nt,
                   'omega_dt':self.omega_dt,
                   'iskip':self.iskip,
                   'do_output':self.fullw_out,
                   'Nu_ampl' : self.Nu_ampl,
                   'Nu_width' : [self.Nu_width_x,self.Nu_width_y,self.Nu_width_z],
                   'absorbing_layer': [self.absorb_x_min,self.absorb_x_max,self.absorb_y_min,self.absorb_y_max,self.absorb_z_min,self.absorb_z_max],
                   'src_ton':self.src_ton
                   })

        self.MOD_NML['model_nml'] =NmlDict({
            'ANTENNA':True,
            'VACUUM' :True,
            'PARAXIAL':True,
            'FULL_WAVE':True,
            'full_wave_solver':'explicit',
            'timer': True,
            'detector': False,
            'spectrum': False,
            'zdim_is_unlimited': True,#set z-dimension to be unlimited in cdf_file record, get around with the 4GB array size limit. 
            'log':False
            })
        self.MOD_NML['FW_expl_interface_nml'] = NmlDict({
            'ixs':[self.ixs[0],self.ixs[1]],
            'ix_refl': self.ix_refl
            })
        self.PARA_NML['PARAXIAL_NML'] = NmlDict({
            'THETA':0.5,
            'OUTPUT':self.para_out,
            'OUTPUT_3D_FIELDS':self.para_out,
            'numerical_method':'fft'
            })
        self.PP_NML['PARAXIAL_NML'] = NmlDict({
            'THETA':0.5,
            'OUTPUT':self.pp_out,
            'CDF_PREFIX':'pp_',
            'output_3d_fields':self.pp_out,
            'numerical_method':'fft'
            })
        self.VAC_NML['VACUUM_NML'] = NmlDict({
            'OUTPUT':self.vac_out
            })
        self.MPI_NML['MPI_nml'] = NmlDict({
            'nproc':[self.nproc_x,self.nproc_y,self.nproc_z]
            })


    def make_input(self,ftype):
        """ create .inp file for given ftype
        ftype: string, see the keys of FILE_NAMES dict for different ftype strings         
        """
        fname = self.run_path + self.FILE_NAMES[ftype]
        self.NML_Dicts[ftype].write(fname,force = True)


    def create_all_input_files(self):
        """create all input files using parameters defined at the beginning.
        """
        if __name__ == 'main':
            print 'creating all the input files in:'+ self.run_path
        self.update_para()
        for ftype in self.FILE_NAMES.keys():
            self.make_input(ftype)

        if __name__ == 'main':
            print 'input files created.'

    def create_executable_links(self):
        """create links to executables
        """
        import subprocess as subp
        try:
            subp.check_call(['ln','-s',self.reflect3d,self.run_path+'reflect_O'])
            subp.check_call(['ln','-s',self.vec_fw,self.run_path+'mvfw_O'])
        except subp.CalledProcessError as e:
            clean = raw_input('Executable linknames already exist in {0}. Do you want to remove the existing ones and make new links?(y/n):  '.format(self.run_path))
            if 'n' in clean:
                print 'I take that as a NO. Process interupted.'
                raise e
            elif 'y' in clean:
                print 'This means YES.'
                subp.check_call(['rm',self.run_path+'reflect_O'])
                subp.check_call(['rm',self.run_path+'mvfw_O'])
                subp.check_call(['ln','-s',self.reflect3d,self.run_path+'reflect_O'])
                subp.check_call(['ln','-s',self.vec_fw,self.run_path+'mvfw_O'])


    
# run the script if executed from command line.

if __name__ == '__main__':
    maker = FWR3D_input_maker()
    maker.create_all_input_files()
    maker.create_executable_links()
   
        
