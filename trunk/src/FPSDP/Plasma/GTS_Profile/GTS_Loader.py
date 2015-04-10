"""Mapping functions that get values on a prescribed Cartesian coordinates grids from GTS output data files which are in flux coordinates.
"""
import Map_Mod_C as mmc
import numpy as np
from FPSDP.Geometry import Grid
import scipy.io.netcdf as nc
from scipy.interpolate import NearestNDInterpolator
from time import clock

class GTS_loader_Error(Exception):
    """Exception class for handling GTS loading errors
    """
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class GTS_Loader:
    """GTS Loading class
    For each GTS run case, setup all the loading parameters, read out necessary data, and output to suited format.
    """

    def __init__(self, grid, t0,dt,nt, fluc_file_path,eq_fname,prof_fname,gts_file_path, n_cross_section = 1, phi_fname_head = 'PHI.', den_fname_head = 'DEN.', n_boundary = 1001, amplification = 1):
       """Initialize Loading Parameters:
        grid: FPSDP.Geometry.Grid.Cartesian2D or Cartesian3D object, contains the output grid information.
        t0: int; Starting time of the sampling series, in simulation record step counts.
        dt: int; The interval between two sample points, in unit of simulation record step counts.
        nt: int; The total number of time_steps.
        n_cross_section: int; total cross-sections used for enlarging the ensemble
        n_boundary: int; The total number of grid points resolving the plasma last closed flux surface. Normally not important.
        fluc_file_path: string; directory where to store the output fluctuation files
        eq_fname: string; filename of the equalibrium file, either absolute or relative path.
        phi_fname_head: string; The header letters of the phi record file before the toroidal plane number, usually "PHI."
        den_fname_head: string; The header letters of the density record file before the toroidal plane number, usually "DEN."
        gts_file_path: string; the directory where the PHI data files are stored.
       """
       self.grid = grid

       if(isinstance(grid,Grid.Cartesian2D)):
           self.dimension = 2
           self.xmin,self.xmax,self.nx = grid.Rmin,grid.Rmax,grid.NR
           self.ymin,self.ymax,self.ny = grid.Zmin,grid.Zmax,grid.NZ
           self.zmin,self.zmax,self.nz = 0,0,1

       elif(isinstance(grid,Grid.Cartesian3D)):
           self.dimension = 3
           self.xmin,self.xmax,self.nx = grid.Xmin,grid.Xmax,grid.NX
           self.ymin,self.ymax,self.ny = grid.Ymin,grid.Ymax,grid.NY
           self.zmin,self.zmax,self.nz = grid.Zmin,grid.Zmax,grid.NZ

       else:
           raise GTS_loader_Error('grid not valid. Right now GTS loader only support Cartesian2D or Cartesian3D grid.')
       

       self.t0,self.dt,self.nt = t0,dt,nt
       self.time_steps = self.t0 + np.arange(self.nt) *self.dt
       self.n_cross_section = n_cross_section

       self.fluc_file_path = fluc_file_path
       self.eq_fname = eq_fname
       self.prof_fname = prof_fname
       self.phi_fname_head = phi_fname_head
       self.den_fname_head = den_fname_head
       self.gts_file_path = gts_file_path
       
       self.n_boundary = n_boundary
       self.amplification = 1

       mmc.set_para_(Xmin=self.xmin,Xmax=self.xmax,NX=self.nx,
                     Ymin=self.ymin,Ymax=self.ymax,NY=self.ny,
                     Zmin=self.zmin,Zmax=self.zmax,NZ=self.nz,
                     NBOUNDARY=self.n_boundary,
                     TStart=self.t0,TStep=self.dt,NT=self.nt,
                     Fluc_Amplification=self.amplification,
                     FlucFilePath=self.fluc_file_path,
                     EqFileName=self.eq_fname,
                     NTFileName=self.prof_fname,
                     PHIFileNameStart=self.phi_fname_head,
                     DENFileNameStart = self.den_fname_head,
                     GTSDataDir=self.gts_file_path)
       mmc.show_para_()
       self.get_fluctuations_from_GTS()

       if (self.dimension == 3):
           self.dne_on_grid = self.ne0_on_grid[np.newaxis,np.newaxis,:,:,:] * (self.dne_ad_on_grid + self.nane_on_grid)
           self.B_2d = self.Btol_2d
       elif (self.dimension == 2):
           self.ne_on_grid = self.ne0_on_grid * (1 + self.dne_ad_on_grid + self.nane_on_grid)
           self.B_on_grid = self.Bt_on_grid

 
    def show_para(self):
        mmc.show_para_()

    def get_fluctuations_from_GTS(self):
        """load fluctuations on grid using C_function
        Create variables:
        equilibrium quantities:
            ne0_on_grid: double ndarray (1,ny,nx), equilibrium electron density.
            Te0_on_grid: double ndarray (1,ny,nx), equilibrium electron temperature.
            Bt_on_grid,Bp_on_grid,BR_on_grid,BZ_on_grid: double ndarray (1,ny,nx), equilibrium toroidal and poloidal magnetic field, and R,Z component of Bpol.
            
        fluctuations:
            dne_ad_on_grid: double ndarray (nt,nz,ny,nx), adiabatic electron density, calculated from fluctuating potential phi: dne_ad_on_grid/ne0_on_grid = e*phi/Te0_on_grid
            nane_on_grid : double ndarray (nt,nz,ny,nx), non-adiabatic electron density normalized to local equilibrium density, read from file. 
            nate_on_grid : double ndarray (nt,nz,ny,nx), non-adiabatic electron temperature normalized to equilibrium temperature at a reference radius, read from file. 
        """
        t0 = clock()
        if(self.dimension == 3):
            x1d = self.grid.X1D
            y1d = self.grid.Y1D
            
            self.x2d = np.zeros((1,self.ny,self.nx))+ x1d[np.newaxis,np.newaxis,:] 
            self.y2d = np.zeros((1,self.ny,self.nx))+ y1d[np.newaxis,:,np.newaxis]
            z2d = np.zeros((1,self.ny,self.nx))

            x3d = self.grid.X3D
            y3d = self.grid.Y3D
            z3d = self.grid.Z3D

            self.dne_ad_on_grid = np.zeros((self.n_cross_section,self.nt,self.nz,self.ny,self.nx))
            self.nane_on_grid = np.zeros((self.n_cross_section,self.nt,self.nz,self.ny,self.nx))
            self.nate_on_grid = np.zeros_like(self.nane_on_grid)
        
            #Note that new equilibrium loading convention needs only 2D equilibrium data. 
            self.ne0_2d = np.zeros((1,self.ny,self.nx))
            self.Te0_2d = np.zeros((1,self.ny,self.nx))
            self.Btol_2d = np.zeros((1,self.ny,self.nx))
            self.Bp_2d = np.zeros((1,self.ny,self.nx))
            self.BR_2d = np.zeros((1,self.ny,self.nx))
            self.BZ_2d = np.zeros((1,self.ny,self.nx))
            mismatched_eq = np.zeros_like(self.x2d,dtype = 'int32')
            
            fluc_2d = np.zeros((self.nt,1,self.ny,self.nx))

            mmc.set_para_(Xmin=self.xmin,Xmax=self.xmax,NX=self.nx,
                          Ymin=self.ymin,Ymax=self.ymax,NY=self.ny,
                          Zmin=0,Zmax=0,NZ=1,
                          NBOUNDARY=self.n_boundary,
                          TStart=1,TStep=1,NT=1,
                          Fluc_Amplification=self.amplification,
                          FlucFilePath=self.fluc_file_path,
                          EqFileName=self.eq_fname,
                          NTFileName=self.prof_fname,
                          PHIFileNameStart=self.phi_fname_head,
                          DENFileNameStart = self.den_fname_head,
                          GTSDataDir=self.gts_file_path)
            
            #one seperate 2D run to get all the equilibrium quantities
            mmc.get_GTS_profiles_(self.x2d,self.y2d,z2d,self.ne0_2d,self.Te0_2d,self.Btol_2d,self.Bp_2d, self.BR_2d, self.BZ_2d, fluc_2d,fluc_2d,fluc_2d,mismatched_eq,0)

            self._fill_mismatched_eq(mismatched_eq)

            #calculate B_toroidal based on B_total and B_poloidal
            self.BPhi_2d = np.sqrt(self.Btol_2d**2 - self.Bp_2d**2)

            mmc.set_para_(Xmin=self.xmin,Xmax=self.xmax,NX=self.nx,
                          Ymin=self.ymin,Ymax=self.ymax,NY=self.ny,
                          Zmin=self.zmin,Zmax=self.zmax,NZ=self.nz,
                          NBOUNDARY=self.n_boundary,
                          TStart=self.t0,TStep=self.dt,NT=self.nt,
                          Fluc_Amplification=self.amplification,
                          FlucFilePath=self.fluc_file_path,
                          EqFileName=self.eq_fname,
                          NTFileName=self.prof_fname,
                          PHIFileNameStart=self.phi_fname_head,
                          DENFileNameStart = self.den_fname_head,
                          GTSDataDir=self.gts_file_path)            

            #temporary arrays to hold 3D equilibrium quantities.
            self.ne0_on_grid = np.zeros_like(x3d)
            self.Te0_on_grid = np.zeros_like(x3d)
            self.Btol_on_grid = np.zeros_like(x3d)
            self.Bp_on_grid = np.zeros_like(x3d)
            self.BR_on_grid = np.zeros_like(x3d)
            self.BZ_on_grid = np.zeros_like(x3d)
            self.mismatch = np.zeros_like(x3d,dtype = 'int32')

            self.total_cross_section = mmc.get_GTS_profiles_(x3d,y3d,z3d,self.ne0_on_grid,self.Te0_on_grid,self.Btol_on_grid,self.Bp_on_grid,self.BR_on_grid,self.BZ_on_grid, self.dne_ad_on_grid[0,...],self.nane_on_grid[0,...],self.nate_on_grid[0,...],self.mismatch, 0)
            
            dcross = int(np.floor(self.total_cross_section / self.n_cross_section))
            self.center_cross_sections = np.arange(self.n_cross_section) * dcross

            for i in range(1,len(self.center_cross_sections)):
                mmc.get_GTS_profiles_(x3d,y3d,z3d,self.ne0_on_grid,self.Te0_on_grid,self.Btol_on_grid,self.Bp_on_grid,self.BR_on_grid,self.BZ_on_grid, self.dne_ad_on_grid[i,...],self.nane_on_grid[i,...],self.nate_on_grid[i,...],self.mismatch,self.center_cross_sections[i])
        
            self._fill_mismatched(self.mismatch)

        elif(self.dimension == 2):
            x1d = self.grid.R1D
            y1d = self.grid.Z1D
            x2d = np.zeros((1,self.ny,self.nx))+ x1d[np.newaxis,np.newaxis,:]
            y2d = np.zeros((1,self.ny,self.nx))+ y1d[np.newaxis,:,np.newaxis]
            z2d = np.zeros((1,self.ny,self.nx))

            self.dne_ad_on_grid = np.zeros((self.n_cross_section,self.nt,1,self.ny,self.nx))
            self.nane_on_grid = np.zeros((self.n_cross_section,self.nt,1,self.ny,self.nx))
            self.nate_on_grid = np.zeros_like(self.nane_on_grid)
        
            #Note that new equilibrium loading convention needs only 2D equilibrium data. 
            self.ne0_on_grid = np.zeros((1,self.ny,self.nx))
            self.Te0_on_grid = np.zeros((1,self.ny,self.nx))
            self.Bt_on_grid = np.zeros((1,self.ny,self.nx))
            self.Bp_on_grid = np.zeros((1,self.ny,self.nx))
            self.BR_on_grid = np.zeros((1,self.ny,self.nx))
            self.BZ_on_grid = np.zeros((1,self.ny,self.nx))
            self.mismatch = np.zeros_like(self.ne0_on_grid,dtype = 'int32')
        
            self.total_cross_section = mmc.get_GTS_profiles_(x2d,y2d,z2d,self.ne0_on_grid,self.Te0_on_grid,self.Bt_on_grid,self.Bp_on_grid,self.BR_on_grid,self.BZ_on_grid, self.dne_ad_on_grid[0,...],self.nane_on_grid[0,...],self.nate_on_grid[0,...],self.mismatch, 0)

            dcross = int(np.floor(self.total_cross_section / self.n_cross_section))
            self.center_cross_sections = np.arange(self.n_cross_section) * dcross

            for i in range(1,len(self.center_cross_sections)):
                mmc.get_GTS_profiles_(x2d,y2d,z2d,self.ne0_on_grid,self.Te0_on_grid,self.Bt_on_grid,self.Bp_on_grid,self.BR_on_grid,self.BZ_on_grid, self.dne_ad_on_grid[i,...],self.nane_on_grid[i,...],self.nate_on_grid[i,...],self.mismatch,self.center_cross_sections[i])
            t1 = clock() 
            self._fill_mismatched(self.mismatch)
            t2 = clock()
            print('Time used for interpolating mismatched points: {0}\nTotal time used:{1}'.format(t2-t1,t2-t0))

    def _fill_mismatched(self,mismatch):
        """interpolate upon correctly matched values, to get values on mismatched points
        """

        print('Start correcting mismatched points.')
        correct_idx = (mismatch == 0)
        mismatch_idx =  (mismatch == 1)
        if self.dimension == 3:
            x_correct = self.grid.X3D[correct_idx]
            y_correct = self.grid.Y3D[correct_idx]
            z_correct = self.grid.Z3D[correct_idx]
        
            xwant = self.grid.X3D[mismatch_idx]
            ywant = self.grid.Y3D[mismatch_idx]
            zwant = self.grid.Z3D[mismatch_idx]

            points = np.array([z_correct,y_correct,x_correct]).T
            points_want = np.array([zwant,ywant,xwant]).T
        
            self.ne0_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.ne0_on_grid[correct_idx])(points_want)
            self.Te0_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.Te0_on_grid[correct_idx])(points_want)
            self.Btol_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.Btol_on_grid[correct_idx])(points_want)
            self.Bp_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.Bp_on_grid[correct_idx])(points_want)
            self.BR_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.BR_on_grid[correct_idx])(points_want)
            self.BZ_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.BZ_on_grid[correct_idx])(points_want)
            for i in range(self.n_cross_section):
                for j in range(self.nt):
                    self.dne_ad_on_grid[i,j][mismatch_idx] = NearestNDInterpolator(points,self.dne_ad_on_grid[i,j][correct_idx])(points_want)
                    self.nane_on_grid[i,j][mismatch_idx] = NearestNDInterpolator(points,self.nane_on_grid[i,j][correct_idx])(points_want)
                    self.nate_on_grid[i,j][mismatch_idx] = NearestNDInterpolator(points,self.nate_on_grid[i,j][correct_idx])(points_want)
                print('Cross-section {0} finished.'.format(i))
        else:
            r_correct = self.grid.R2D[correct_idx[0,:,:]]
            z_correct = self.grid.Z2D[correct_idx[0,:,:]]
            rwant = self.grid.R2D[mismatch_idx[0,:,:]]
            zwant = self.grid.Z2D[mismatch_idx[0,:,:]]
            
            points = np.array([z_correct,r_correct]).T
            points_want = np.array([zwant,rwant]).T

            self.ne0_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.ne0_on_grid[correct_idx])(points_want)
            self.Te0_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.Te0_on_grid[correct_idx])(points_want)
            self.Bt_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.Bt_on_grid[correct_idx])(points_want)
            self.Bp_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.Bp_on_grid[correct_idx])(points_want)
            self.BR_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.BR_on_grid[correct_idx])(points_want)
            self.BZ_on_grid[mismatch_idx] = NearestNDInterpolator(points,self.BZ_on_grid[correct_idx])(points_want)
            for i in range(self.n_cross_section):
                for j in range(self.nt):
                    self.dne_ad_on_grid[i,j][mismatch_idx] = NearestNDInterpolator(points,self.dne_ad_on_grid[i,j][correct_idx])(points_want)
                    self.nane_on_grid[i,j][mismatch_idx] = NearestNDInterpolator(points,self.nane_on_grid[i,j][correct_idx])(points_want)
                    self.nate_on_grid[i,j][mismatch_idx] = NearestNDInterpolator(points,self.nate_on_grid[i,j][correct_idx])(points_want)
                print('Cross-section {0} finished.'.format(i)) 

    def _fill_mismatched_eq(self,mismatch):
        """Dedicated to interpolate equilibrium quantities on mismatched points for a 3D run. 
        """

        print('Start correcting mismatched points.')
        correct_idx = (mismatch == 0)
        mismatch_idx =  (mismatch == 1)
        
        r_correct = self.x2d[correct_idx]
        z_correct = self.y2d[correct_idx]
        rwant = self.x2d[mismatch_idx]
        zwant = self.y2d[mismatch_idx]
        
        points = np.array([z_correct,r_correct]).T
        points_want = np.array([zwant,rwant]).T
        
        self.ne0_2d[mismatch_idx] = NearestNDInterpolator(points,self.ne0_2d[correct_idx])(points_want)
        self.Te0_2d[mismatch_idx] = NearestNDInterpolator(points,self.Te0_2d[correct_idx])(points_want)
        self.Btol_2d[mismatch_idx] = NearestNDInterpolator(points,self.Btol_2d[correct_idx])(points_want)
        self.Bp_2d[mismatch_idx] = NearestNDInterpolator(points,self.Bp_2d[correct_idx])(points_want)
        self.BR_2d[mismatch_idx] = NearestNDInterpolator(points,self.BR_2d[correct_idx])(points_want)        
        self.BZ_2d[mismatch_idx] = NearestNDInterpolator(points,self.BZ_2d[correct_idx])(points_want)
                

    def cdf_output(self,output_path,eq_file = 'equilibrium.cdf',filehead = 'fluctuation', WithBp = True):
        """
        Wrapper for cdf_output_2D and cdf_output_3D.
        Determining 2D/3D by checking the grid property.
            
        """

        if ( self.dimension == 2 ):
            self.cdf_output_2D(output_path,filehead)
        elif (self.dimension == 3):
            self.cdf_output_3D(output_path,eq_file,filehead,WithBp)
        else:
            raise GTS_loader_Error('Wrong grid type! Grid should either be Cartesian2D or Cartesian3D.') 

    def cdf_output_2D(self,output_path,filehead='fluctuation'):
        """write out cdf files for old FWR2D code use

        Arguments:
        output_path: string, the full path to put the output files
        filehead: string, the starting string of all filenames

        CDF file format:
        Dimensions:
        r_dim: int, number of grid points in R direction.
        z_dim: int, number of grid points in Z direction
        
        Variables:
        rr: 1D array, coordinates in R direction, in Meter
        zz: 1D array, coordinates in Z direction, in Meter
        bb: 2D array, total magnetic field on grids, in Tesla, shape in (z_dim,r_dim)
        ne: 2D array, total electron density on grids, in m^-3
        ti: 2D array, total ion temperature, in keV
        te: 2D array, total electron temperature, in keV
        
        """
        file_start = output_path + filehead
        for i in range(self.n_cross_section):
            for j in range(len(self.time_steps)):
            
                fname = file_start + str(self.time_steps[j])+'_'+str(i) + '.cdf'
                f = nc.netcdf_file(fname,'w')
                f.createDimension('z_dim',self.grid.NZ)
                f.createDimension('r_dim',self.grid.NR)

                rr = f.createVariable('rr','d',('r_dim',))
                rr[:] = self.grid.R1D[:]
                zz = f.createVariable('zz','d',('z_dim',))
                zz[:] = self.grid.Z1D[:]
                rr.units = zz.units = 'Meter'

                bb = f.createVariable('bb','d',('z_dim','r_dim'))
                bb[:,:] = self.B_on_grid[0,:,:]
                bb.units = 'Tesla'
                
                ne = f.createVariable('ne','d',('z_dim','r_dim'))
                ne[:,:] = self.ne_on_grid[i,j,0,:,:]
                ne.units = 'per cubic meter'

                te = f.createVariable('te','d',('z_dim','r_dim'))
                te[:,:] = self.Te0_on_grid[0,:,:]
                te.units = 'keV'
                
                f.close()

    def cdf_output_3D(self,output_path = './',eq_filename = 'equilibrium.cdf',flucfilehead='fluctuation', WithBp = True):
        """write out cdf files for FWR3D code to use

        Arguments:
        output_path: string, the full path to put the output files
        eq_filename: string, the file name for the 2D equilibrium output
        flucfilehead: string, the starting string of all 3D fluctuation filenames

        CDF file format:

        Equilibrium file:
        
        Dimensions:
        nr: int, number of grid points in radial direction.
        nz: int, number of grid points in vetical direction
        
        Variables:
        rr: 1D array, coordinates in radial direction, in m
        zz: 1D array, coordinates in vertical direction, in m
        bb: 2D array, total magnetic field on grids, in Tesla, shape in (nz,nr)
        bpol: 2D array, magnetic field in poloidal direction, in Tesla
        b_r: 2D array, R component of B field
        b_z: 2D array, Z component of B field
        b_phi: 2D array, Phi component of B field
        ne: 2D array, total electron density on grids, in cm^-3
        ti: 2D array, total ion temperature, in keV
        te: 2D array, total electron temperature, in keV

        Fluctuation files:

        Dimensions:
        nx: number of grid points in radial direction
        ny: number of grid points in vertical direction
        nz: number of grid points in horizontal direction

        Variables:
        xx: 1D array, coordinates in radial direction
        yy: 1D array, coordinates in vertical direction 
        zz: 1D array, coordinates in horizontal direction
        dne: 3D array, (nz,ny,nx), adiabatic electron density perturbation, real value 
        """
        eqfname = output_path + eq_filename
        f = nc.netcdf_file(eqfname,'w')
        f.createDimension('nz',self.grid.NY)
        f.createDimension('nr',self.grid.NX)
        
        rr = f.createVariable('rr','d',('nr',))
        rr[:] = self.grid.X1D[:]
        zz = f.createVariable('zz','d',('nz',))
        zz[:] = self.grid.Y1D[:]
        rr.units = zz.units = 'm'
        
        bb = f.createVariable('bb','d',('nz','nr'))
        bb[:,:] = self.B_2d[0,:,:]
        bb.units = 'Tesla'

        bpol = f.createVariable('bpol','d',('nz','nr'))
        if(WithBp):
            bpol[:,:] = self.Bp_2d[0,:,:]
        else:
            bpol[:,:] = np.zeros_like(self.Bp_2d[0])
        bpol.units = 'Tesla'

        b_r = f.createVariable('b_r','d',('nz','nr'))
        if(WithBp):
            b_r[:,:] = self.BR_2d[0,:,:]
        else:
            b_r[:,:] = np.zeros_like(self.BR_2d[0])
        b_r.units = 'Tesla'

        b_z = f.createVariable('b_z','d',('nz','nr'))
        if(WithBp):
            b_z[:,:] = self.BZ_2d[0,:,:]
        else:
            b_z[:,:] = np.zeros_like(self.BZ_2d[0])
        b_z.units = 'Tesla'

        b_phi = f.createVariable('b_phi','d',('nz','nr'))
        if(WithBp):
            b_phi[:,:] = self.BPhi_2d[0,:,:]
        else:
            b_phi[:,:] = np.zeros_like(self.BPhi_2d[0])
        b_phi.units = 'Tesla'
        
        ne = f.createVariable('ne','d',('nz','nr'))
        ne[:,:] = self.ne0_2d[0,:,:]
        ne.units = 'm^-3'
        
        te = f.createVariable('te','d',('nz','nr'))
        te[:,:] = self.Te0_2d[0,:,:]
        te.units = 'keV'
                
        f.close()

        
        file_start = output_path + flucfilehead
        for j in range(self.n_cross_section):
            for i in range(len(self.time_steps)):
                fname = file_start + str(self.time_steps[i]) +'_'+ str(j)+ '.cdf'
                f = nc.netcdf_file(fname,'w')
                f.createDimension('nx',self.grid.NX)
                f.createDimension('ny',self.grid.NY)
                f.createDimension('nz',self.grid.NZ)

                xx = f.createVariable('xx','d',('nx',))
                xx[:] = self.grid.X1D[:]
                yy = f.createVariable('yy','d',('ny',))
                yy[:] = self.grid.Y1D[:]
                zz = f.createVariable('zz','d',('nz',))
                zz[:] = self.grid.Z1D[:]            
                xx.units = yy.units = zz.units = 'm'
            
                dne = f.createVariable('dne','d',('nz','ny','nx'))
                dne.units = 'm^-3'
                
                dne[:,:,:] = self.dne_on_grid[j,i,:,:,:]
                f.close()

#=======END of class GTS_Loader definition =======================================================
