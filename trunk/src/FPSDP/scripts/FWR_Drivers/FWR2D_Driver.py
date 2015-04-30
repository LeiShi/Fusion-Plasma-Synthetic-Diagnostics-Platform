#!/usr/bin/env python
"""Temporary Driver Script for reflectometry analysis

This script is used to create working directories, creating and copying necessary files, and start FWR2D runs

Multiprocess version: using IPython multiprocessing method to drive multiple FWR2D runs on cluster
Updated Mar 16,2015 by Lei Shi

NOTE: Before running this script, you need to start IPython cluster service on your local network

example:
$ipcluster start -n 64

This will start a cluster with 64 processes

For use with multiple nodes communicating via MPI, check out the IPython documentation for details of initiating a cluster. ipcontroller and ipengine may need to be called manually.

Google 'IPython start cluster' for more information

==========================================================
PREPARATION:

Before using this script, a working path must be set up by hand. The directory shoud have the following subirectories:

inputs/ : this directory should have 2 kinds of input files:
    1) input files for FWR2D executable: input file specifying FWR2D parameters.
       Filename convention: 'input' + {$frequency}*10 + '.txt'
    2) launching and receiving antenna pattern Code5 files.
       Filename convention: 'antenna_pattern_launch_' + {$frequency}*10 + '.txt'
                            'antenna_pattern_receive_'+ {$frequency}*10 + '.txt'
2D_fluctuations/ : this directory contains plasma cdf files created by Plasma package modules

bin/ : this directory contains executable: drive_FWR2D

RUNS/ : this is where new runs will be placed

SCRIPTS/: this script is recommended to be placed here.
==========================================================

USAGE:

$FWR2D_Driver.py -h

This will print out the available options.

Make sure you modify 'run_No' in this script before submit a new run. Otherwise the result from the last run may be overwritten.
"""

#The tag of RUN. Each new run should be assigned a new number.
run_No = '_140GHz_275t'

import time
from IPython.parallel import Client
c = Client(profile='pbs')

#the engine needs time to start, so check when all the engines are connected before take a direct view of the cluster.

desired_engine_num = 128 #Make sure this number is EXACTLY the same as the engine number you initiated with ipengine

waiting=0
while(len(c) < desired_engine_num and waiting<=86400):#check if the engines are ready, if the engines are not ready after 1 min, something might be wrong. Exit and raise an exception.
    time.sleep(10)
    waiting += 10

if(len(c) != desired_engine_num):
    raise Exception('usable engine number is not the same as the desired engine number! usable:{0}, desired:{1}.\nCheck your cluster status and the desired number set in the Driver script.'.format(len(c),desired_engine_num))


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

working_path = '/p/gkp/lshi/GTS_ALCATOR_Case/L_mode/'

#toroidal cross sactions used

n_cross_section = 1 #Total number of Cross Sections used

#Time slices parameters
time_start = 1
time_end = 275
time_inc = 1

time_arr = np.arange(time_start,time_end+1,time_inc)

#frequencies in GHz

freqs = [140]

#input file parameters

input_path = working_path+'inputs/'
FWR_driver_input_head = 'input'#followed by '{$frequency*10}.txt'
FWR_driver_link_name = 'input.txt'

#following names are default to be used in other scripts. DO NOT CHANGE THEM UNLESS YOU KNOW WHAT ELSE YOU NEED TO MODIFY. (list of dependent scripts:FWR_driver_input_file,Postprocess script using FPSDP.Diagnostics.Reflectometry.FWR2D.Postprocess)
incident_antenna_pattern_head = 'antenna_pattern_launch_'#followed by '{$frequency*10}.txt'
receiver_antenna_pattern_head = 'antenna_pattern_receive_'#followed by '{$frequency*10}.txt'
incident_antenna_link_name = 'antenna_pattern.txt'
receiver_antenna_link_name = 'receiver_pattern.txt'
#END of the default names


#fluctuation file parameters
fluc_path = working_path + '2D_fluctuations/'
fluc_head = 'fluctuation'
fluc_link_name = 'plasma.cdf'

#executable parameters
bin_path = working_path + 'bin/'
exe = 'drive_FWR2D'

full_output_path = working_path + 'RUNS/RUN'+str(run_No)+'/'

#Broadcast useful names to all worker nodes
dv.push(dict(working_path = working_path,
             full_output_path = full_output_path,
             n_cross_section = n_cross_section,
             time_arr = time_arr,
             freqs = freqs,
             FWR_driver_link_name = FWR_driver_link_name,
             exe = 'drive_FWR2D',
             run_No = run_No))

def make_dirs(f_arr = freqs,t_arr = time_arr, nc = n_cross_section, ask=True):
    
    os.chdir(working_path+'RUNS/')
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

    os.chdir(full_output_path)
    subp.check_call(['cp',working_path+'SCRIPTS/ALCATOR_FWR2D_Driver.py','./'])    
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


def start_run(params):
    """ start an FWR2D run with specific frequency, time and cross-section.
    """
    f,t,nc = params
    os.chdir(full_output_path+'/'+str(f)+'/'+str(t)+'/'+str(nc))
    with open('./output.txt','w') as output:
        try:
	    subp.check_call([exe,FWR_driver_link_name,],stdout=output)
	except Exception as e:
	    print >>output, 'Exception catched:{0}. from Case f={1},t={2},nc={3}. This run is skipped, others will continue, assuming this error is because you are trying to continue running the Script to finish a prematurely stopped run. Be sure this is the case to go on to use the result.'.format(str(e),f,t,nc)
	else:
	    print >>output, 'Case f={0},t={1},nc={2} finished without exceptions.'.format(f,t,nc)
         

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
    
    t_use = time_arr
    f_use = freqs #[freqs[i] for i in range(1)]
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
    ar.wait_interactive()
    print('All FWR2D runs finished. Check detailed output at output.txt in every run directory.')
        
    

if __name__ == "__main__":
    main(sys.argv[1:])
