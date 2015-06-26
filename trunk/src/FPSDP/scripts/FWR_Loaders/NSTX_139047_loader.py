import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid
import numpy as np
from IPython.parallel import Client
import time

xgc_path = '/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/'
output_path = '/p/gkp/lshi/XGC1_NSTX_Case/new_2D_fluctuations/no_dn_cancel/'

grid2D = Grid.Cartesian2D(DownLeft = (-0.5,1.25),UpRight = (0.5,1.6),NR = 100, NZ = 300)

grid3D = Grid.Cartesian3D(Xmin = 1.25,Xmax = 1.6,Ymin = -0.5, Ymax = 0.5, Zmin = -0.35, Zmax = 0.35, NX = 100,NY = 300,NZ = 100)

def load(dimension,tstart,tend,tstep,full_load,fluc_only,eq_only):
    if dimension == 3:
        xgc_nstx_139047 = xgc.XGC_Loader(xgc_path,grid3D,tstart,tend,tstep,dn_amplifier = 1,n_cross_section = 16, Equilibrium_Only = eq_only,Full_Load = full_load, Fluc_Only = fluc_only)
    elif dimension == 2:
        xgc_nstx_139047 = xgc.XGC_Loader(xgc_path,grid2D,tstart,tend,tstep,dn_amplifier = 1,n_cross_section = 1, Full_Load = full_load, Fluc_Only = fluc_only)
        
    return xgc_nstx_139047

def cluster_initialize(desired_engine_num,profile = 'default'):
    client_started = False
    while(not client_started):
        try:
            c = Client(profile = profile)
            client_started = True
            print 'Client connected!'
        except:
            pass
    
    print desired_engine_num   #Make sure this number is EXACTLY the same as the engine number you initiated with ipcluster
    
    waiting=0
    while(len(c) < desired_engine_num and waiting<=120):#check if the engines are ready, if the engines are not ready after 2 min, something might be wrong. Exit and raise an exception.
        time.sleep(10)
        waiting += 10
    
    if(len(c) != desired_engine_num):
        raise Exception('usable engine number is not the same as the desired engine number! usable:{0}, desired:{1}.\nCheck your cluster status and the desired number set in the Driver script.'.format(len(c),desired_engine_num))
    
    print 'engine number checked:{0}'.format(desired_engine_num)
    dv = c[:]
    print('Multiprocess FWR2D run started: using {} processes.'.format(len(c.ids)))
    
    dv.push(dict(xgc_path = xgc_path,
                 output_path = output_path))
                 
    return dv

def single_load(t):
    try:
        import sys
        sys.path.append('/p/gkp/lshi/')
        import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
        import FPSDP.Geometry.Grid as Grid
        grid3D = Grid.Cartesian3D(Xmin = 1.25,Xmax = 1.6,Ymin = -0.5, Ymax = 0.5, Zmin = -0.35, Zmax = 0.35, NX = 100,NY = 300,NZ = 100)
        xgc_nstx = xgc.XGC_Loader(xgc_path,grid3D,t,t+1,2,dn_amplifier = 1,n_cross_section = 16, Equilibrium_Only = False,Full_Load = True, Fluc_Only = False)
        xgc_nstx.cdf_output(output_path,eq_file = 'eqfile{0}.cdf'.format(t))
    except Exception as e:
        return str(e)
    return 0
    
    

def launch(t_arr, dv):
    print t_arr
    param_list = [t for t in t_arr]
    print param_list
    ar = dv.map_async(single_load,param_list)
    return ar

def single_load_2d(t):
    try:
        import sys
        sys.path.append('/p/gkp/lshi/')
        import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
        import FPSDP.Geometry.Grid as Grid
        grid2D = Grid.Cartesian2D(DownLeft = (-0.5,1.25),UpRight = (0.5,1.6),NR = 256, NZ = 512)
        xgc_nstx = xgc.XGC_Loader(xgc_path,grid2D,t,t+1,2,dn_amplifier = 1,n_cross_section = 32, Equilibrium_Only = False,Full_Load = True, Fluc_Only = False)
        xgc_nstx.cdf_output(output_path,eq_file = 'eqfile{0}.cdf'.format(t))
    except Exception as e:
        return str(e)
    return 0
    
def launch_2d(t_arr, dv):
    print t_arr
    param_list = [t for t in t_arr]
    print param_list
    ar = dv.map_async(single_load_2d,param_list)
    return ar
    
def main():
    t_arr = np.arange(100,221,1)
    dv = cluster_initialize(28,'pbs')
    ar = launch_2d(t_arr,dv)
    print('FWR Loaders launched.')
    ar.wait_interactive()
    print('All FWR Loaders finished. Check {} for the outputs.'.format(output_path))
    print('Return values from processes are:')
    for i in range(16):
        print '#{0}:{1}'.format(i,ar.get()[i])

if (__name__=='__main__'):
    
    main()

#freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]

