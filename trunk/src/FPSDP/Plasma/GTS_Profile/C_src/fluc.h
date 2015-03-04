/*! \file fluc.h
    \brief Functions for getting the potential fluctuations

    Called by all the top-level functions that require fluctuation information
 */

#ifndef VERBOSE
#define VERBOSE 0
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

// files
extern char* NTFileName; // used by get_fluctuations
extern char* GTS_DATA_DIR; 

// linear interpolation between two points
#define N_NEAREST_TOROIDAL 2
#define N_NEAREST_PSIGRID 2
#define N_NEAREST_THETAGRID 2

// set the range of variables to be [0,2*pi) and [-pi,pi), respectively
#define SETRANGE_0_2PI(x) fmod(2.0*M_PI+fmod(x,2.0*M_PI),2.0*M_PI)
#define SETRANGE_MPI_PPI(x) -M_PI + fmod(3.0*M_PI+fmod(x,2.0*M_PI),2.0*M_PI)

// informative error output
#define REFLECTOMETER_INFO_FLUC_ERROR {fprintf(stderr,"error in reflectometer_info_fluc\n"); exit(ERRCODE);}

// external function declarations

extern REAL findMax(int n,REAL data[]);
extern REAL findMin(int n,REAL data[]);

// --- local function declarations
int fluc_info_file();
int get_fluctuations(int npts,size_t nTimeSteps,int* ntoroidal,REAL phi[],REAL ne[],REAL te[], REAL a[],REAL theta[],REAL zeta[],int timeSteps[],int flag[],int toroidal_startnum);
int total_fluctuations(int npts,size_t nTimeSteps,REAL ne_tilde[],REAL ne0[],REAL phi[],int* flag);
int interpolateTorPolPsi(int npts,REAL phi[npts],REAL a[npts],int ntoroidal,int mgrid,
			 REAL **potential,REAL deltazeta[npts][N_NEAREST_TOROIDAL],
			 int toroidalIdxs[npts][N_NEAREST_TOROIDAL],
			 int thetaGridIdxs[npts][N_NEAREST_PSIGRID][N_NEAREST_THETAGRID],
			 REAL deltaThetaGrid[npts][N_NEAREST_PSIGRID],int mtheta[],
			 size_t psiGridIdxs[npts][N_NEAREST_PSIGRID],REAL psiGrid[],
			 int flag[]);
int nearestThetaGridMap(int npts,int nearestThetaGridIdxs[npts][N_NEAREST_PSIGRID][N_NEAREST_THETAGRID],REAL deltaThetaGridVals[npts][N_NEAREST_PSIGRID],REAL theta[npts][N_NEAREST_PSIGRID],size_t psi_grid_idxs[npts][N_NEAREST_PSIGRID],int igrid[],int mtheta[],int flag[]);
int nearestPoloidalSlices(int npts,int ntoroidal,REAL zeta[],int toroidalIdxs[npts][N_NEAREST_TOROIDAL],REAL deltazeta[npts][N_NEAREST_TOROIDAL],int flag[]);
int traceBtoTorodialPosn(int npts,int ntoroidal,int mpsi,REAL a[],REAL thetaOut[npts][N_NEAREST_PSIGRID],size_t psi_idxs[npts][N_NEAREST_PSIGRID],REAL thetaIn[],REAL qtinv[],REAL psi_grid[],REAL zeta[npts],int flag[]);


// ---- local function definitions

//! wrapper for get_fluctuations when used with an existing fluctuation coord. file
/*! Writes the potential fluctuations to a netCDF file */
/*int fluc_info_file(){
  size_t nx,ny,nz,ntimesteps;
  int *timesteps;
  REAL *a,*theta,*zeta;

  int status = readFlucCoordFile(FLUC_FNAME,&nx,&ny,&nz,&ntimesteps,&a,&theta,&zeta,&timesteps);
  int n3d=nx*ny*nz;

  // Thus have the a,theta,zeta coords of the points.
  // Now we need to interpolate the fluctuation values to those points.
  REAL *phi = (REAL *)PyMem_Malloc(n3d*sizeof(REAL));
  int flag[n3d];
  status = get_fluctuations(n3d,ntimesteps,phi,a,theta,zeta,timesteps,flag);
  
  // write to netCDF file
  status = writeFlucFile(FLUC_FNAME,nx,ny,nz,phi);

  PyMem_Free(phi);

  return 0;
  }*/

//! Given flux coordinates (a,theta) performs all the steps to get the potential fluctuation at those points
int get_fluctuations(int npts,size_t nTimeSteps,int* ntoroidal_out,REAL phi[], REAL ne[],REAL te[],REAL a[],REAL theta[],REAL zeta[],int timeSteps[],int flag[], int toroidal_startnum){
  // We have the (a,theta,zeta) coords of the points.
  // Now we need to interpolate the fluctuation values to those points.
  
  // read in potential data from PHI.000xx files
  char fname[FNAME_MAX_CHARS];
  int phi_ints[N_PHI_INTEGERS];
  REAL phi_reals[N_PHI_REALS];
  int nsteps,*igrid=NULL,*mtheta=NULL,*itran=NULL;
  REAL *qtinv=NULL,*deltat=NULL,*vth_grid=NULL,***potential=NULL, ***ti_raw, ***ni_raw, ***te_raw, ***ne_raw;
  //  int timeSteps[N_TIME_STEPS] = TIME_STEPS;

  int status = readAllPhiFiles(GTS_DATA_DIR,nTimeSteps,timeSteps,phi_ints,phi_reals,&igrid,
			       &mtheta,&itran,&qtinv,&deltat,&vth_grid,&potential,&nsteps,npts,zeta,toroidal_startnum);
  printf("successfully read all PHI files.\n");
  status = readAllDenFiles(GTS_DATA_DIR,nTimeSteps, timeSteps,phi_ints,phi_reals,&igrid, &mtheta,
			   &itran, &qtinv, &deltat, &vth_grid, &ni_raw,&ti_raw,&ne_raw,&te_raw, &nsteps,npts,zeta,toroidal_startnum);
  *ntoroidal_out = phi_ints[5];
  printf("successfully read all DEN files.\n");
  // useful: phi_ints,igrid[mpsi],mtheta[mpsi],qtinv[mpsi]

  //  also gives potential[N_TIME_STEPS][ntoroidal][mgrid]
  //  nTimeSteps = 1 if so we're just looking at 1 time step, but can be > 1
  


//#if VERBOSE
  fprintf(stderr,"Number of fluctuation time steps:%d, using time step:%d to %d every %d.\n",nsteps,timeSteps[0],timeSteps[nTimeSteps-1],timeSteps[1]-timeSteps[0]);
//#endif
  // unpack the integer array
  int mpsi = phi_ints[2];	// igrid,mtheta,itran,qtinv,deltat,vth_grid have mpsi elements
  int mgrid = phi_ints[3];	// phi_steps[nsteps][mgrid]
  int ntoroidal = phi_ints[5];	// number of toroidal points (poloidal slices)

  
  // read in psi_grid data
  //Lei Shi temp off 
  REAL *psi_grid = readPsiGridFile(NTFileName,mpsi);
  /* REAL *psi_grid=(REAL*) PyMem_Malloc(sizeof(REAL)*mpsi);
  int idx;
  double dpsi=0.9/(mpsi-1);
  for(idx=0;idx<mpsi;idx++){
    psi_grid[idx]=dpsi*idx;
  }
  */
  REAL psi_grid_min=psi_grid[0];
  REAL psi_grid_max=psi_grid[mpsi-1];
  int idx2;
  for(idx2=0;idx2<npts;idx2++){
    if(a[idx2]>psi_grid_max || a[idx2]<psi_grid_min)
      flag[idx2]=0;
    else
      flag[idx2]=1;
  }
  // find the nearest toroidal points (poloidal slices) in the data
  int toroidalIdxs[npts][N_NEAREST_TOROIDAL];
  REAL deltazeta[npts][N_NEAREST_TOROIDAL];
  status = nearestPoloidalSlices(npts,ntoroidal,zeta,toroidalIdxs,deltazeta,flag);

  // find the value of theta at the lower toroidal pts by following field line
  //   also returns the nearest two psi grid indices
  REAL theta0[npts][N_NEAREST_PSIGRID];	// the value of theta mapped along field line to zeta=0 plane
  size_t psi_grid_idxs[npts][N_NEAREST_PSIGRID]; // nearest psi values with grid points
  status = traceBtoTorodialPosn(npts,ntoroidal,mpsi,a,theta0,
				psi_grid_idxs,theta,qtinv,psi_grid,zeta,flag);

  // find the indices of the nearest 2 theta grid points at for each psi grid point
  //   at the lower toroidal location (the indices are the same at the other location)
  // also find the distance theta_mapped - theta_grid_lower
  int theta_grid_idxs[npts][N_NEAREST_PSIGRID][N_NEAREST_THETAGRID];
  REAL delta_theta_grid[npts][N_NEAREST_PSIGRID]; // distance in theta from lower grid pt.
  status = nearestThetaGridMap(npts,theta_grid_idxs,delta_theta_grid,
			       theta0,psi_grid_idxs,igrid,mtheta,flag);

  // using the values at these (psi_grid,theta_grid,zeta_grid),
  //   interpolate first toroidally, then poloidally, then between flux surfaces
  // this function stores values in the "phi" array
  int i;
  for(i=0;i<nTimeSteps;i++){	// loop for each time step
    status = interpolateTorPolPsi(npts,&phi[i*npts],a,ntoroidal,mgrid,potential[i],deltazeta,toroidalIdxs,
				  theta_grid_idxs,delta_theta_grid,mtheta,psi_grid_idxs,psi_grid,flag);
    status = interpolateTorPolPsi(npts,&te[i*npts],a,ntoroidal,mgrid,te_raw[i],deltazeta,toroidalIdxs,
				  theta_grid_idxs,delta_theta_grid,mtheta,psi_grid_idxs,psi_grid,flag);
    status = interpolateTorPolPsi(npts,&ne[i*npts],a,ntoroidal,mgrid,ne_raw[i],deltazeta,toroidalIdxs,
				  theta_grid_idxs,delta_theta_grid,mtheta,psi_grid_idxs,psi_grid,flag); 
  }

  // free memory
  PyMem_Free(igrid);
  PyMem_Free(mtheta);
  PyMem_Free(itran);
  PyMem_Free(qtinv);
  PyMem_Free(deltat);
  PyMem_Free(vth_grid);
  PyMem_Free(potential);
  PyMem_Free(psi_grid);

  return 0;
}




//! Linearly interpolate the potential to the point of interest
/*! Interpolates toroidally, then poloidally, then radially (flux coordinate).*/
int interpolateTorPolPsi(int npts,REAL phi[npts],REAL a[npts],int ntoroidal,int mgrid,
			 REAL **potential,REAL deltazeta[npts][N_NEAREST_TOROIDAL],
			 int toroidalIdxs[npts][N_NEAREST_TOROIDAL],
			 int thetaGridIdxs[npts][N_NEAREST_PSIGRID][N_NEAREST_THETAGRID],
			 REAL deltaThetaGrid[npts][N_NEAREST_PSIGRID],int mtheta[],
			 size_t psiGridIdxs[npts][N_NEAREST_PSIGRID],REAL psiGrid[],
			 int flag[npts]){
  // potential needs to be passed as a REAL ** rather than as REAL [ntoroidal][mgrid] since this compiler is pre-C99 and doesn't support variable length arrays

  int point,psiPoint,thetaPoint;
  REAL dZetaGrid = 2.0*M_PI/ntoroidal;
  REAL dThetaGrid,psiUpper,psiLower;
  REAL phiA[N_NEAREST_PSIGRID][N_NEAREST_THETAGRID];
  REAL phiB[N_NEAREST_PSIGRID];


  for(point=0;point<npts;point++){
    if(flag[point]==1){
      for(psiPoint=0;psiPoint<N_NEAREST_PSIGRID;psiPoint++){
	for(thetaPoint=0;thetaPoint<N_NEAREST_THETAGRID;thetaPoint++){
	  phiA[psiPoint][thetaPoint] =
	    interpBetw2ptsDist(deltazeta[point][0],
			       potential[toroidalIdxs[point][0]][thetaGridIdxs[point][psiPoint][thetaPoint]],
			       potential[toroidalIdxs[point][1]][thetaGridIdxs[point][psiPoint][thetaPoint]],
			       dZetaGrid);
	}
	dThetaGrid = 2.0*M_PI/mtheta[psiGridIdxs[point][psiPoint]];
	phiB[psiPoint] = interpBetw2ptsDist(deltaThetaGrid[point][psiPoint],phiA[psiPoint][0],phiA[psiPoint][1],dThetaGrid);
      }
      psiUpper = psiGrid[psiGridIdxs[point][1]];
      psiLower = psiGrid[psiGridIdxs[point][0]];
      phi[point] = interpBetw2ptsDist(a[point]-psiLower,phiB[0],phiB[1],psiUpper-psiLower);
    }
    else{
      phi[point] = 0;
    }
  }

  return 0;
}


//! Find the nearest grid points in theta
/*! I need to change this to use theta at zeta = 0. */
int nearestThetaGridMap(int npts,int nearestThetaGridIdxs[npts][N_NEAREST_PSIGRID][N_NEAREST_THETAGRID],REAL deltaThetaGridVals[npts][N_NEAREST_PSIGRID],REAL theta[npts][N_NEAREST_PSIGRID],size_t psi_grid_idxs[npts][N_NEAREST_PSIGRID],int igrid[],int mtheta[],int flag[]){
  int i,j,psi_idx,itheta_grid;
  REAL this_theta,dthetagrid;
  for(i=0;i<npts;i++){
    if(flag[i]==1){
      for(j=0;j<N_NEAREST_PSIGRID;j++){
	psi_idx = psi_grid_idxs[i][j];
	this_theta = theta[i][j];	// range of theta is [0,2*pi)
	dthetagrid = 2.0*M_PI/((double)mtheta[psi_idx]);
	itheta_grid= (int)floor(this_theta/dthetagrid);
	nearestThetaGridIdxs[i][j][0] = itheta_grid + igrid[psi_idx];
	nearestThetaGridIdxs[i][j][1] = itheta_grid + igrid[psi_idx]+ 1; // since there is an extra point at 2pi, don't have to worry about exceeding the max idx for this psi 
	deltaThetaGridVals[i][j] = this_theta - dthetagrid*itheta_grid; // distance between our point of interest and the theta grid point just below it.
      }
    }
  }


  return 0;
}

//! Finds the nearest toroidal grid points and the toroidal distance
int nearestPoloidalSlices(int npts,int ntoroidal,REAL zeta[],int toroidalIdxs[npts][N_NEAREST_TOROIDAL],REAL deltazeta[npts][N_NEAREST_TOROIDAL],int flag[]){
  int i;
  REAL tor_pt,dzeta=2.0*M_PI/ntoroidal;
  REAL dzetainv=1.0/dzeta;
#if DEBUG >0
  fprintf(stderr,"nearestPoloidalSlices:\n");
  fprintf(stderr,"zeta:    z_lower:   dz_lower:   z_upper   dz_upper:\n");
#endif
  for(i=0;i<npts;i++){
    if(flag[i]==1){
      zeta[i] = SETRANGE_0_2PI(zeta[i]); // range of zeta set to [0,2*pi)
      // tor_pt = fmod(ntoroidal + zeta[i]*dzetainv -0.5,ntoroidal);
      // changed above line, since toroidal position is arbitrary to a const.
      //  now, 0 <= zeta[i] < dzeta will get mapped to toroidal idxs of 0,1
      tor_pt = fmod(ntoroidal + zeta[i]*dzetainv,ntoroidal);
      int lower_slice = (int)floor(tor_pt); // ranges from 0 to ntoroidal-1
      int upper_slice = (int)fmod(lower_slice+1,ntoroidal); // ranges from 0 to ntoroidal-1
      toroidalIdxs[i][0]=lower_slice;
      toroidalIdxs[i][1]=upper_slice;
      
      // again, these to lines are commented out to get the distances right
      //     deltazeta[i][0]=SETRANGE_MPI_PPI(zeta[i]-dzeta*(0.5+lower_slice));
      //     deltazeta[i][1]=SETRANGE_MPI_PPI(zeta[i]-dzeta*(0.5+upper_slice));
      deltazeta[i][0]=SETRANGE_MPI_PPI(zeta[i]-dzeta*lower_slice);
      deltazeta[i][1]=SETRANGE_MPI_PPI(zeta[i]-dzeta*upper_slice);
#if DEBUG >0
      fprintf(stderr,"%3g, %3g, %3g, %3g, %3g\n",zeta[i],dzeta*(lower_slice+0.5),deltazeta[i][0],dzeta*(upper_slice+0.5),deltazeta[i][1]);
#endif
    }
  }

  return 0;
}

//! Gives the nearest psi grid points, and the theta position traced along the field line to the zeta = 0 plane
int traceBtoTorodialPosn(int npts,int ntoroidal,int mpsi,REAL a[],REAL theta0[npts][N_NEAREST_PSIGRID],size_t psi_idxs[npts][N_NEAREST_PSIGRID],REAL thetaIn[],REAL qtinv[],REAL psi_grid[],REAL zeta[npts],int flag[]){
    // follow a field line: const = theta - zeta*qtinv
    int i,j,k;

    // output a warning if any points are outside the domain
#if VERBOSE >0
    REAL psi_grid_min = psi_grid[0];
    REAL psi_grid_max = psi_grid[mpsi-1];
    for(i=0;i<npts;i++){
      if(flag[i]==1){
	if(a[i] > psi_grid_max){
	  fprintf(stderr,"warning: point %d is outside fluctuation domain: a[%d]:%.3f, a[max]:%.3f\n",
		  i,i,a[i],psi_grid_max);
	}else if(a[i]< psi_grid_min)
	  fprintf(stderr,"warning: point %d is outside fluctuation domain: a[%d]:%.3f, a[min]:%.3f\n",
		  i,i,a[i],psi_grid_min);
      }
    }
#endif

    // 1st find the nearest two psi
    find_nearest1d(npts,mpsi,N_NEAREST_PSIGRID,1,psi_idxs,a,psi_grid);

#if DEBUG > 0
    fprintf(stderr,"traceBtoToroidalPosn:\n");
    fprintf(stderr,"a:      psi_-/+:    qtinv(psi_-/+):  zeta:   thetaIn:   theta0(-/+):\n");
#endif
    // 2nd use this psi with its corresponding qtinv to trace the field line toroidally to the plane zeta = 0
    for(i=0;i<npts;i++){
      //      for(j=0;j<N_NEAREST_TOROIDAL;j++)
      if(flag[i]==1){
	for(k=0;k<N_NEAREST_PSIGRID;k++){
	  //  thetaOut[i][j][k] = thetaIn[i] + qtinv[psi_idxs[i][k]]*deltazeta[i][j];
	  theta0[i][k] = thetaIn[i] - qtinv[psi_idxs[i][k]]*zeta[i];
	  theta0[i][k]=SETRANGE_0_2PI(theta0[i][k]); // range of theta is [0,2*pi)
#if DEBUG > 0
	  fprintf(stderr,"%.3g, %.3g, %.3g, %.3g,  %.3g, %.3g\n",a[i],psi_grid[psi_idxs[i][k]],qtinv[psi_idxs[i][k]],zeta[i],thetaIn[i],theta0[i][k]);
#endif
	}
      }
    }

    return 0;
  }


int total_fluctuations(int npts, size_t ntimesteps, REAL ne_tilde[],REAL ne0[],REAL phi[],int* flag){
  int i,j;
  for(i=0;i<ntimesteps;i++)
    for(j=0;j<npts;j++){
      if(flag[j]==1)
	ne_tilde[i*npts+j] = ne0[j]*phi[i*npts+j];
      else
	ne_tilde[i*npts+j] = 0;
    }
}
