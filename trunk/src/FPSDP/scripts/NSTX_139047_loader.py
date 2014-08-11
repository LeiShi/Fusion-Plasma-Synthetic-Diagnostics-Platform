import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid

xgc_path = '/p/gkp/lshi/XGC1_NSTX_Case/FullF_XGC_ti191_output/'

grid2D = Grid.Cartesian2D(DownLeft = (-0.4,1.25),UpRight = (0.4,1.6),NR = 256, NZ = 256)

grid3D = Grid.Cartesian3D(Xmin = 1.25,Xmax = 1.6,Ymin = -0.4, Ymax = 0.4, Zmin = -0.4, Zmax = 0.4, NX = 128,NY = 256,NZ = 64)

time_start = 100
time_end = 220
time_step = 10

def load():
    xgc_nstx_139047 = xgc.XGC_loader(xgc_path,grid2D,time_start,time_end,time_step,dn_amplifier = 0.1,n_cross_section = 16, Full_Load = True)
    return xgc_nstx_139047

if (__name__=='__main__'):
    xgc_nstx_139047 = load()
    xgc_nstx_139047.cdf_output('/p/gkp/lshi/XGC1_NSTX_Case/2D_fluctuations/Amp01/')

freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]

