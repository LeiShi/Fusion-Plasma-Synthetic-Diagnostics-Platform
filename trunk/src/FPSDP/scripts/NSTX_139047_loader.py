import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid

xgc_path = '/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/'

grid2D = Grid.Cartesian2D(DownLeft = (-0.5,1.25),UpRight = (0.5,1.6),NR = 256, NZ = 512)

grid3D = Grid.Cartesian3D(Xmin = 1.25,Xmax = 1.6,Ymin = -0.5, Ymax = 0.5, Zmin = -0.35, Zmax = 0.35, NX = 100,NY = 300,NZ = 80)

time_start = 100
time_end = 220
time_step = 10

def load(full_load):
    xgc_nstx_139047 = xgc.XGC_loader(xgc_path,grid3D,time_start,time_end,time_step,dn_amplifier = 1,n_cross_section = 16, Full_Load = full_load)
    return xgc_nstx_139047

if (__name__=='__main__'):
    xgc_nstx_139047 = load(True)
    xgc_nstx_139047.cdf_output('/p/gkp/lshi/XGC1_NSTX_Case/new_3D_fluctuations/Amp1/')

freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]

