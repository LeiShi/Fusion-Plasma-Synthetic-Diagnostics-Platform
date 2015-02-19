import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid

xgc_path = '/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/'

grid2D = Grid.Cartesian2D(DownLeft = (-0.5,1.25),UpRight = (0.5,1.6),NR = 256, NZ = 512)

grid3D = Grid.Cartesian3D(Xmin = 1.3,Xmax = 1.6,Ymin = -0.5, Ymax = 0.5, Zmin = -0.3, Zmax = 0.3, NX = 256,NY = 512,NZ = 80)

time_start = 1
time_end = 220
time_step = 1

def load(dimension,full_load,fluc_only):
    if dimension == 3:
        xgc_nstx_139047 = xgc.XGC_Loader(xgc_path,grid3D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 1, Full_Load = full_load, Fluc_Only = fluc_only)
    elif dimension == 2:
        xgc_nstx_139047 = xgc.XGC_Loader(xgc_path,grid2D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 1, Full_Load = full_load, Fluc_Only = fluc_only)
        
    return xgc_nstx_139047

if (__name__=='__main__'):
    xgc_nstx_139047 = load(3,True,False)
    xgc_nstx_139047.cdf_output('/p/gkp/lshi/XGC1_NSTX_Case/new_3D_fluctuations/fulltime/')
    xgc_nstx_139047.save('/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/xgc_prof3D_all_fulltime.sav')

freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]

