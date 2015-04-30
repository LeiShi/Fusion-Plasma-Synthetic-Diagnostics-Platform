#!/usr/bin/env python
"""FWR3D Driver Script for NSTX_139047 reflectometry analysis

This script is used to create working directories, creating and copying necessary files, and submit batch jobs
"""
import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import numpy as np
import subprocess as subp
import os
import imp

working_path = '/p/gkp/lshi/XGC1_NSTX_Case/'

#toroidal cross sactions used

n_cross_section = 16 #Total number of Cross Sections used

#Time slices parameters
time_start = 100
time_end = 220
time_inc = 8

time_arr = np.arange(time_start,time_end+1,time_inc)

#frequencies in GHz

freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]
freqs_chosen = [57.5]

#MPI process info
nproc_x = 2
nproc_y = 8
nproc_z = 2

nproc_total = nproc_x * nproc_y * nproc_z

#input file parameters

input_path = working_path+'inputs/'

incident_antenna_pattern_head = 'antenna_pattern_launch_nstx'
incident_antenna_link_name = 'antenna_pattern.txt'
receiver_antenna_pattern_head = 'antenna_pattern_receive_nstx'
receiver_antenna_link_name = 'receiver_pattern.txt'

Make_inps_filehead = 'Make_inps'



#plasma file parameters
fluc_path = working_path + 'new_3D_fluctuations/time_step_8/'
equilibrium_file = 'equilibrium.cdf'
fluc_head = 'fluctuation'
equilibrium_link_name = 'equilibrium.cdf'
fluc_link_name = 'fluctuation.cdf'

#executable parameters
bin_path = working_path + 'bin/reflect3d/'
exe_reflect = 'reflect'
exe_FW = 'mpi_vec_fw'
link_reflect = 'reflect_O'
link_FW = 'mvfw_O'
#Start creating directories and files

#The tag of RUN. Each new run should be assigned a new number.
run_No = '_NEWAll_16_cross_16_time_57.5GHz'

full_output_path = working_path + 'Correlation_Runs/3DRUNS/RUN'+str(run_No)+'/'

#Boolean controls the output
all_output = True


def make_dirs(f_arr = freqs_chosen,t_arr = time_arr, nc = n_cross_section):
    
    os.chdir(working_path+'Correlation_Runs/3DRUNS/')
    #create the RUN directory for the new run
    try:
        subp.check_call(['mkdir','RUN'+str(run_No)])
    except subp.CalledProcessError as e:
        clean = raw_input('RUN Number:'+str(run_No)+' already existed!\n Do you want to overwrite it anyway? This will erase all the data under the existing directory, please make sure you have saved all the useful data or simply change the run_No in the script and try again.\n Now, do you REALLY want to overwrite the existing data?(y/n):  ')
    
        if 'n' in clean:
            print 'I take that as a NO, process interupted.'
            raise e
        elif 'y' in clean:
            print 'This means YES.'
            
            try:
                subp.check_call(['rm','-r','RUN'+str(run_No)])
            except:
                print 'Got problem deleting the directory: RUN'+str(run_No)+'.\n Please check the permission and try again.'
                raise
            subp.call(['mkdir','RUN'+str(run_No)])
            print 'Old data deleted, new directory built. Go on with preparing the new run.'
        else:
            print 'Not a valid option, but I\'ll take that as a NO. process interupted.'
            raise e

    os.chdir(working_path+'Correlation_Runs/3DRUNS/RUN'+str(run_No))
        
    #create the subdirectories for each detector(frequency) and plasma realization, add necessary links and copies of corresponding files.

    for f in f_arr:
        try:
            subp.check_call(['mkdir',str(f)])
            os.chdir(str(f))
            #make link to the Make_inps file for this frequency
            subp.check_call(['ln','-s',input_path+Make_inps_filehead+str(int(f*10))+'.py','Make_inps.py'])
            #import the Make_inps file for later use
            Make_inps = imp.load_source('Make_inps','./Make_inps.py')
            os.chdir('../')
            for t in t_arr:
                subp.check_call(['mkdir',str(f)+'/'+str(t)])
                for j in range(nc):
                    subp.check_call(['mkdir',str(f)+'/'+str(t)+'/'+str(j)])
                    os.chdir(str(f)+'/'+str(t)+'/'+str(j))

                    # make link to plasma equilibrium file
                    subp.check_call(['ln','-s',fluc_path+equilibrium_file,equilibrium_link_name])
                    # make link to plasma perturbation file
                    subp.check_call(['ln','-s',fluc_path+fluc_head+str(t)+'_'+str(j)+'.cdf',fluc_link_name])
                    #make links to the executable
                    subp.check_call(['ln','-s',bin_path+exe_reflect,link_reflect])
                    subp.check_call(['ln','-s',bin_path+exe_FW,link_FW])
                    #make links to  the antenna pattern files
                    subp.check_call(['ln','-s',input_path + incident_antenna_pattern_head + str(int(f*10))+'.txt',incident_antenna_link_name])
                    subp.check_call(['ln','-s',input_path + receiver_antenna_pattern_head + str(int(f*10))+'.txt',receiver_antenna_link_name])
                    
                    #call functions from Make_inps to create necessary .inp files
                    #modify corresponding parameters in Make_inps script
                    if(all_output):
                        full_out = '.TRUE.'
                    else:
                        full_out = '.FALSE.'

                    mi = Make_inps.FWR3D_input_maker()
                    
                    mi.eps_out = full_out
                    mi.eps_1d_out = full_out
                    mi.vac_out = full_out
                    mi.para_out = full_out
                    mi.pp_out = full_out
                    #mi.fullw_out = full_out

                    mi.nproc_x = nproc_x
                    mi.nproc_y = nproc_y
                    mi.nproc_z = nproc_z
                    mi.ant_freq = f*1e9
                    mi.equilibrium_file = equilibrium_link_name
                    mi.fluctuation_file = fluc_link_name

                    if(f>=70):
                        mi.nr_crossings = 8
                    else:
                        mi.nr_crossings = 4
                    

                    mi.create_all_input_files()
                    os.chdir('../../..')
        except subp.CalledProcessError:
            print 'Something is wrong, check the running environment.'
            raise
        
def make_batch(f_arr=freqs_chosen,t_arr=time_arr,nc = n_cross_section):
    """write batch job files for chosen frequencies and time slices
    """
    os.chdir(working_path+'Correlation_Runs/3DRUNS/RUN'+str(run_No))
    for f in f_arr:
        for t in t_arr:
            for j in range(nc):
                os.chdir(str(f)+'/'+str(t)+'/'+str(j))
                batch_file = open('batch','w')
                batch_file.write('#PBS -N reflect3d_'+str(f)+'_'+str(t)+'_'+str(j)+'\n')
                batch_file.write('#PBS -m a\n')
                batch_file.write('#PBS -M lshi@pppl.gov\n')
                batch_file.write('#PBS -l nodes={0}:ppn=8\n'.format(nproc_total/8) )  #NOTE: here assume total process number is a multiple of 8. If not, this line must be modified to agree with the situation. The total processors requested must equal to the total process number set.
                batch_file.write('#PBS -l mem={0}gb\n'.format(nproc_total*2))
                batch_file.write('#PBS -l walltime=1:00:00\n')
                batch_file.write('#PBS -r n\n')
                batch_file.write('cd $PBS_O_WORKDIR\n\n')
                batch_file.write('./reflect_O\n')
                if(not all_output):
                    batch_file.write('rm ./epsilon.cdf\n')
                    batch_file.write('rm ./parax.cdf\n')
                    batch_file.write('rm ./FW_expl.cdf\n')
                    batch_file.write('rm ./3dout.cdf\n')
                batch_file.close()
                os.chdir('../../..')
    
    

def submit(f_arr=freqs_chosen,t_arr=time_arr,nc = n_cross_section):
    """ submit the batch jobs
    """
    os.chdir(working_path+'Correlation_Runs/3DRUNS/RUN'+str(run_No))
    for f in f_arr:
        for t in t_arr:
            for j in range(nc):
                os.chdir(str(f)+'/'+str(t)+'/'+str(j))
                subp.check_call(['qsub','./batch'])
                os.chdir('../../..')


#run the functions:

if __name__ == "__main__":

    t_use = np.arange(100,107,8)
    f_use = [57.5]#[30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]
    nc_use = 1
    #make_dirs()#(t_arr = t_use,f_arr = f_use,nc = nc_use)
    #make_batch()#(t_arr = t_use,f_arr = f_use,nc = nc_use)
    submit()#()(t_arr = t_use,f_arr = f_use, nc = nc_use)
