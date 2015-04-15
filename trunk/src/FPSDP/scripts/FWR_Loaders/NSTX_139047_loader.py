import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid

xgc_path = '/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/'

grid2D = Grid.Cartesian2D(DownLeft = (-0.5,1.25),UpRight = (0.5,1.6),NR = 256, NZ = 512)

grid3D = Grid.Cartesian3D(Xmin = 1.25,Xmax = 1.6,Ymin = -0.5, Ymax = 0.5, Zmin = -0.35, Zmax = 0.35, NX = 100,NY = 300,NZ = 80)

def load(dimension,tstart,tend,tstep,full_load,fluc_only,eq_only):
    if dimension == 3:
        xgc_nstx_139047 = xgc.XGC_Loader(xgc_path,grid3D,tstart,tend,tstep,dn_amplifier = 1,n_cross_section = 1, Equilibrium_Only = eq_only,Full_Load = full_load, Fluc_Only = fluc_only)
    elif dimension == 2:
        xgc_nstx_139047 = xgc.XGC_Loader(xgc_path,grid2D,tstart,tend,tstep,dn_amplifier = 1,n_cross_section = 32, Full_Load = full_load, Fluc_Only = fluc_only)
        
    return xgc_nstx_139047

if (__name__=='__main__'):
    for t in range(100,220,20):
        xgc_nstx_139047 = load(2,t,t+20,1,True,False,False)
        #xgc_nstx_139047.cdf_output('/p/gkp/lshi/XGC1_NSTX_Case/new_2D_fluctuations/Amp1_All/')
        xgc_nstx_139047.save('xgc_prof2D_all_time{0}.sav'.format(t))

freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]

