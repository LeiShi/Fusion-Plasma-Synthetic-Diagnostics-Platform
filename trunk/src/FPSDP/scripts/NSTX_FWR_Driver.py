#!/usr/bin/env python
"""Temporary Driver Script for NSTX_139047 reflectometry analysis

This script is used to create working directories, creating and copying necessary files, and submit batch jobs
"""
import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import numpy as np
import subprocess as subp
import os

working_path = '/p/gkp/lshi/XGC1_NSTX_Case/'

#toroidal cross sactions used

n_cross_section = 16 #Total number of Cross Sections used

#Time slices parameters
time_start = 100
time_end = 220
time_inc = 10

time_arr = np.arange(time_start,time_end+1,time_inc)

#frequencies in GHz

freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]

#input file parameters

input_path = working_path+'inputs/'
FWR_driver_input_head = 'input'
FWR_driver_link_name = 'input.txt'
incident_antenna_pattern_head = 'antenna_pattern_launch_nstx'
incident_antenna_link_name = 'antenna_pattern.txt'
receiver_antenna_pattern_head = 'antenna_pattern_receive_nstx'
receiver_antenna_link_name = 'receiver_pattern.txt'

#fluctuation file parameters
fluc_path = working_path + '2D_fluctuations/'
fluc_head = 'fluctuation'
fluc_link_name = 'plasma.cdf'

#executable parameters
bin_path = working_path + 'bin/'
exe = 'drive_FWR2D'

#Start creating directories and files

#The tag of RUN. Each new run should be assigned a new number.
run_No = '_FullF_multi_cross'

full_output_path = working_path + 'Correlation_Runs/RUNS/RUN'+str(run_No)+'/'


def make_dirs(f_arr = freqs,t_arr = time_arr, nc = n_cross_section):
    
    os.chdir(working_path+'Correlation_Runs/RUNS/')
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

    os.chdir(working_path+'Correlation_Runs/RUNS/RUN'+str(run_No))
        
    #create the subdirectories for each detector(frequency) and plasma realization, add necessary links and copies of corresponding files.

    for f in f_arr:
        try:
            subp.check_call(['mkdir',str(f)])
            for t in t_arr:
                subp.check_call(['mkdir',str(f)+'/'+str(t)])
                for j in range(nc):
                    subp.check_call(['mkdir',str(f)+'/'+str(t)+'/'+str(j)])
                    os.chdir(str(f)+'/'+str(t)+'/'+str(j))
                    # make link to plasma perturbation file
                    subp.check_call(['ln','-s',fluc_path+fluc_head+str(t)+'_'+str(j)+'.cdf',fluc_link_name])
                    #make link to the executable
                    subp.check_call(['ln','-s',bin_path+exe,exe])
                    #make links to  the input files
                    subp.check_call(['ln','-s',input_path+FWR_driver_input_head+str(int(f*10))+'.txt',FWR_driver_link_name])
                    subp.check_call(['ln','-s',input_path + incident_antenna_pattern_head + str(int(f*10))+'.txt',incident_antenna_link_name])
                    subp.check_call(['ln','-s',input_path + receiver_antenna_pattern_head + str(int(f*10))+'.txt',receiver_antenna_link_name])               
                    os.chdir('../../..')
        except subp.CalledProcessError:
            print 'Something is wrong, check the running environment.'
            raise
        
def make_batch(f_arr=freqs,t_arr=time_arr,nc = n_cross_section):
    """write batch job files for chosen frequencies and time slices
    """
    os.chdir(working_path+'Correlation_Runs/RUNS/RUN'+str(run_No))
    for f in f_arr:
        for t in t_arr:
            for j in range(nc):
                os.chdir(str(f)+'/'+str(t)+'/'+str(j))
                batch_file = open('batch','w')
                batch_file.write('#PBS -N reflect_'+str(f)+'_'+str(t)+'_'+str(j)+'\n')
                batch_file.write('#PBS -m a\n')
                batch_file.write('#PBS -M lshi@pppl.gov\n')
                batch_file.write('#PBS -l nodes=1:ppn=1\n')
                batch_file.write('#PBS -l mem=1000mb\n')
                batch_file.write('#PBS -l walltime=1:00:00\n')
                batch_file.write('#PBS -r n\n')
                batch_file.write('cd $PBS_O_WORKDIR\n\n')
                batch_file.write(exe+' '+FWR_driver_link_name+'\n')
                batch_file.close()
                os.chdir('../../..')
    
    

def submit(f_arr=freqs,t_arr=time_arr,nc = n_cross_section):
    """ submit the batch jobs
    """
    os.chdir(working_path+'Correlation_Runs/RUNS/RUN'+str(run_No))
    for f in f_arr:
        for t in t_arr:
            for j in range(nc):
                os.chdir(str(f)+'/'+str(t)+'/'+str(j))
                subp.check_call(['qsub','./batch'])
                os.chdir('../../..')


#run the functions:

if __name__ == "__main__":

    t_use = [100]
    f_use = [32.5,55,72.5]
    nc_use = 1
    make_dirs()#(t_arr = t_use,f_arr = f_use,nc = nc_use)
    make_batch()#(t_arr = t_use,f_arr = f_use,nc = nc_use)
    submit()#(t_arr = t_use,f_arr = f_use, nc = nc_use)
