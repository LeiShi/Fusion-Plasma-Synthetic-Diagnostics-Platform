#!/usr/bin/env python
"""Temporary Driver Script for NSTX_139047 reflectometry analysis

This script is used to create working directories, creating and copying necessary files, and start FWR2D runs

Multiprocess version: using IPython multiprocessing method to drive multiple FWR2D runs on cluster
Updated Mar 16,2015 by Lei Shi

NOTE: Before running this script, you need to start IPython cluster service on your local network

example:
$ipcluster start -n 64

This will start a cluster with 64 processes

Google "IPython start cluster" for more information

"""

from IPython.parallel import Client
c = Client()
dv = c[:]
print('Multiprocess FWR2D run started: using {} processes.'.format(len(c.ids)))


with dv.sync_imports():
    import numpy as np
    import subprocess as subp
    import os
    import sys
    import getopt

dv.execute('np=numpy')
dv.execute('subp=subprocess')

working_path = '/p/gkp/lshi/XGC1_NSTX_Case/'

#toroidal cross sactions used

n_cross_section = 1 #Total number of Cross Sections used

#Time slices parameters
time_start = 1
time_end = 220
time_inc = 1

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
fluc_path = working_path + 'new_2D_fluctuations/Amp1_All/'
fluc_head = 'fluctuation'
fluc_link_name = 'plasma.cdf'

#executable parameters
bin_path = working_path + 'bin/'
exe = 'drive_FWR2D'

#Start creating directories and files

#The tag of RUN. Each new run should be assigned a new number.
run_No = '_TEST_MULTIPROC_batch'#'_NSTX_139047_All_Channel_All_Time'

full_output_path = working_path + 'Correlation_Runs/RUNS/RUN'+str(run_No)+'/'

dv.push(dict(working_path = working_path,
             n_cross_section = n_cross_section,
             time_arr = time_arr,
             freqs = freqs,
             FWR_driver_link_name = FWR_driver_link_name,
             exe = 'drive_FWR2D',
             run_No = run_No))

def make_dirs(f_arr = freqs,t_arr = time_arr, nc = n_cross_section, ask=True):
    
    os.chdir(working_path+'Correlation_Runs/RUNS/')
    #create the RUN directory for the new run
    try:
        subp.check_call(['mkdir','RUN'+str(run_No)])
    except subp.CalledProcessError as e:
        if(ask):
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
        else:
            raise e

    os.chdir(working_path+'Correlation_Runs/RUNS/RUN'+str(run_No))
    subp.check_call(['cp','../SCRIPT/NSTX_FWR_Driver.py','./'])    
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

        
#make_batch method is depracated in Multiprocess version
'''
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
                batch_file.write('#PBS -l walltime=3:00:00\n')
                batch_file.write('#PBS -r n\n')
                batch_file.write('cd $PBS_O_WORKDIR\n\n')
                batch_file.write(exe+' '+FWR_driver_link_name+'\n')
                batch_file.close()
                os.chdir('../../..')
    
    
#submit method is deprecated in Multiprocess version
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
'''

def start_run(params):
    """ start an FWR2D run with specific frequency, time and cross-section.
    """
    f,t,nc = params
    os.chdir(working_path+'Correlation_Runs/RUNS/RUN'+str(run_No)+'/'+str(f)+'/'+str(t)+'/'+str(nc))
    with open('./output.txt','w') as output:
        subp.check_call([exe,FWR_driver_link_name,],stdout=output) 

def launch(f_arr,t_arr,nc):
    """launch multiple tasks of FWR2D runs for given freqs, t_arr and total cross-section number
    """
    param_list = [(f,t,ic) for f in f_arr for t in t_arr for ic in range(nc)]
    ar = dv.map_async(start_run,param_list)    
    return ar
    
#run the functions:

def main(argv):
    """main routine that runs the driver:
    accepts arguments to modify frequencies, time_array, and number of cross-section
    options accepted:
        -f: followed by a list of frequencies, [...]
        -t: followed by a tuple setting (tstart, tend,tstep)
        -c: followed by an int to specify number of cross-sections
        -n: create new dir
        -a: ask when running directory already exists.
        -h: showing help information
    """
    try:
        opts,args = getopt.getopt(argv,'hnaf:t:c:')
    except(getopt.GetoptError):
        print 'unexpected option. use -h to see allowed options.'
        sys.exit(2)
    
    t_use =[1]#time_arr
    f_use = freqs[0]
    nc_use = 1
    
    for opt,arg in opts:
        if(opt == '-h'):
            print('main routine that runs the driver:\naccepts arguments to modify frequencies, time_array, and number of cross-section\noptions accepted:\n-f: followed by a list of frequencies, [...]\n-t: followed by a tuple setting (tstart, tend,tstep)\n-c: followed by an int to specify number of cross-sections\n-n: create new dir\n-a: ask when running directory already exists.\n-h: showing help information')
            sys.exit(1)
        if(opt == '-f'):
            f_use = eval(arg)
        if(opt == '-t'):
            tstart,tend,tstep = eval(arg)            
            t_use = range(tstart,tend+1,tstep)
        if(opt == '-c'):
            nc_use = eval(arg)
        if(opt == '-n'):
            if '-a' in [o[0] for o in opts]:
                ask = True
            else:
                ask = False
            try:
                make_dirs(t_arr = t_use,f_arr = f_use,nc = nc_use,ask=ask)
            except subp.CalledProcessError:
                print 'Creating new directory fails. Run_No may already exist. Resolve the problem and run again.'
                sys.exit(3)
    
    print('Running in directory:{0}\nWith parameters:\nf_use={1}\nt_use={2}\nnc_use={3}\n'.format(full_output_path,str(f_use),str(t_use),str(nc_use)))
                
    ar = launch(f_arr = f_use,t_arr = t_use,nc = nc_use)
    print('FWR processes launched.')
    ar.wait()
    print('All FWR2D runs finished. Check detailed output at output.txt in every run directory.')
        
    

if __name__ == "__main__":
    main(sys.argv[1:])
