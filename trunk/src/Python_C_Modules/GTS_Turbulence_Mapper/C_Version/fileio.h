/*! \file fileio.h
  \brief File read/write functions for reflectometery interface.

    Function definitions to read and write the netCDF files used by the reflectometry interface.

*/

#ifndef erikfileio_c
#define erikfileio_c
#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#include <netcdf.h>		// needed for netcdf files

#ifndef single			// compiler option determines variable size
#ifndef REAL_TYPEDEF
#define REAL_TYPEDEF
typedef double REAL;
#endif
#define NC_REAL NC_DOUBLE
#define nc_put_var_real nc_put_var_double
#define nc_get_var_real nc_get_var_double
#else
#ifndef REAL_TYPEDEF
#define REAL_TYPEDEF
typedef float REAL;
#endif
#define NC_REAL NC_FLOAT
#define nc_put_var_real nc_put_var_float
#define nc_get_var_real nc_put_var_float
#endif

#ifndef VERBOSE			// default to not verbose output
#define VERBOSE 0
#endif

#define SCALAR 0		// scalar data
#define ARRAY1D 1			// 1D data arrays
#define ARRAY2D 2			// 2D data arrays
#define ARRAY3D 3			// 3D data arrays
#define ARRAY4D 4			// 4D data arrays

// attribute string definitions
#define UNITS "units"
#define METERS "meters"
#define TESLA "tesla"
#define M3 "m^{-3}"
#define KEV "kev"

// definitions to read PHI.000xx files
#define N_PHI_INTEGERS 6
#define N_PHI_REALS 4
#define N_PHI_1DINT_ARR 3
#define N_PHI_1DREAL_ARR 3

// used by constructPhiFname
#define FNAME_MAX_CHARS 100


// simple error handler from netcdf examples
#define ERRCODE 2
#define ERR(e) {fprintf(stderr,"NetCDF Error: %s\n", nc_strerror(e)); exit(ERRCODE);}
#define READPHI_ERR(str) {fprintf(stderr,"error in phiread:\n%s\n",str); exit(ERRCODE);}
#define READPSI_ERR(fname,mpsi,mread) {fprintf(stderr,"error in readPsiGridFile reading file:%s\n given mpsi:%d, mpsi read:%d\n",fname,mpsi,mread); exit(ERRCODE);}
#define READALLPHIFILES_ERR(str) {fprintf(stderr,"error in readAllPhiFiles:\n%s\n",str); exit(ERRCODE);}
#define WRITE_FLUC_FILE_ERR(str) {fprintf(stderr,"error in writeFlucFile:\n %s\n",str); exit(ERRCODE);}
#define READ_SPECIFIED_3D_COORD_FILE_ERR(str) {fprintf(stderr,"error in readSpecified3dCoordFile:\n %s\n",str); exit(ERRCODE);}
#define READ_SPECIFIED_2D_COORD_FILE_ERR(str) {fprintf(stderr,"error in readSpecified2dCoordFile:\n %s\n",str); exit(ERRCODE);}

extern REAL findMax(int,REAL[]);
extern REAL findMin(int,REAL[]);
extern char* PHI_FNAME_START;

int writeTextFile(char *fname,int n,REAL Rwant[],REAL Zwant[],REAL a[],REAL theta[],REAL R[],REAL Z[],REAL Bm[],REAL Bpol[],REAL T[],REAL Te[],REAL P[],REAL ne[],REAL Rinitial[],REAL Zinitial[],REAL qprofile[]);
int write2dNetcdfFile(char *fname,int nr,int nz,size_t ntimesteps,int nboundary,REAL R1d[],REAL Z1d[],REAL a[],REAL theta[],REAL R[],REAL Z[],REAL Bm[],REAL Bpol[],REAL T[],REAL Te[],REAL P[],REAL ne[],REAL ne_tilde[],REAL phi[],int timesteps[],REAL Rinitial[],REAL Zinitial[],REAL qprofile[],REAL Rboundary[],REAL Zboundary[],REAL mag_axis_coords[]);
int write3dNetcdfFile(char *fname,int nx,int ny,int nz,size_t ntimesteps,int nboundary,REAL x1d[],REAL y1d[],REAL z1d[],REAL a[],REAL theta[],REAL zeta[],REAL Rwant[],REAL Zwant[],REAL Ract[],REAL Zact[],REAL Bm[],REAL Bpol[],REAL T[],REAL Te[],REAL P[],REAL ne[],REAL ne_tilde[],REAL phi[],int timesteps[],REAL Rinitial[],REAL Zinitial[],REAL qprofile[],REAL Rboundary[],REAL Zboundary[],REAL mag_axis_coords[]);
int readNetcdfFile(char *fname,int *nr,int *nz,REAL *R1d[],REAL *Z1d[],REAL *a[],REAL *theta[],REAL *R[],REAL *Z[],REAL *Bm[],REAL *Bpol[],REAL *T[],REAL *Te[],REAL *P[],REAL *ne[],REAL *Rinitial[],REAL *Zinitial[],REAL *qprofile[]);
int readSpecified2dCoordFile(char *fname,size_t *nr,size_t *nz,size_t *ntimesteps,REAL *zeta,REAL *R1d[],REAL *Z1d[],int *timesteps[]);
int readSpecified3dCoordFile(char *fname,size_t *nx,size_t *ny,size_t *nz,size_t *ntimesteps,REAL *x1d[],REAL *y1d[],REAL *z1d[],int *timesteps[]);

int writePhiNetcdfFile(char *fname,int nTimeSteps,int ntoroidal,int mgrid,REAL phi[]);
int writeFlucFile(char *fname,int nx,int ny,int nz,REAL phi[]);
int readPhiFile(char *fname,int phi_integers[N_PHI_INTEGERS],REAL phi_reals[N_PHI_REALS],int *igrid[],int *mtheta[],int *itran[],REAL *qtinv[],REAL *deltat[],REAL *vth_grid[],REAL **phi_steps[],int *nsteps);
int writeFlucCoordFile(char *fname,int nx,int ny,int nz,REAL x[],REAL y[],REAL z[],REAL Rwant[],REAL Zwant[],REAL zeta[],REAL Ract[],REAL Zact[],REAL a[],REAL theta[],REAL Rinitial[],REAL Zinitial[],REAL mag_axis_coords[]);
int readFlucCoordFile(char *fname,size_t *nx,size_t *ny,size_t *nz,size_t *ntimesteps,REAL *a[],REAL *theta[],REAL *zeta[],int *timesteps[]);
REAL *readPsiGridFile(char *fname,int mpsi); // reads file formatted like dsda_psi.dat to get psi_grid array

int loadNTProfiles(const char* fname,int** n_prof,REAL** a_p, REAL** n_p,REAL** Ti_p,REAL** Te_p){
  // netcdf identifiers
  int file_id; // id for file
  int a_dim_id;  // ids for dimensions
  int a_id,ne_id,Ti_id,Te_id; // ids for variables
  size_t *n;   
  int error_code;		// non-zero means netcdf error
     
  *n_prof=(int*)PyMem_Malloc(sizeof(int));
  n=(size_t *)PyMem_Malloc(sizeof(size_t));
  
  if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);
     
  if ((error_code = nc_inq_dimid(file_id, "profiles_radial_grid", &a_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id,a_dim_id,n)))
    ERR(error_code);

  if ((error_code = nc_inq_varid(file_id,"psi_grid_profiles",&a_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"ne_profile",&ne_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Ti_profile",&Ti_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Te_profile",&Te_id)))
    ERR(error_code);

  **n_prof = *n;
  *a_p=(REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  *n_p = (REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  *Ti_p = (REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  *Te_p = (REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  PyMem_Free(n);
  
  if ((error_code = nc_get_var_double(file_id,a_id,*a_p)))
    ERR(error_code);
  if ((error_code = nc_get_var_double(file_id,ne_id,*n_p)))
    ERR(error_code);
  if ((error_code = nc_get_var_double(file_id,Ti_id,*Ti_p)))
    ERR(error_code);
  if ((error_code = nc_get_var_double(file_id,Te_id,*Te_p)))
    ERR(error_code);

  return 0;
}

int clearNTProfiles(int** n_prof, REAL** a_p, REAL** n_p, REAL** Ti_p, REAL** Te_p){
  PyMem_Free(*n_prof);
  PyMem_Free(n_prof);
  PyMem_Free(*a_p);
  PyMem_Free(a_p);
  PyMem_Free(*n_p);
  PyMem_Free(n_p);
  PyMem_Free(*Ti_p);
  PyMem_Free(Ti_p);
  PyMem_Free(*Te_p);
  PyMem_Free(Te_p);
  return 0;
}

//! (Deprecated) Store profile information in a comma-separated text file.
/*! First row contains the column headers.
    Subsequent rows contain data in vertical columns, comma separated. */
int writeTextFile(char *fname,int n,REAL Rwant[],REAL Zwant[],REAL a[],REAL theta[],REAL R[],REAL Z[],REAL Bm[],REAL Bpol[],REAL T[],REAL Te[],REAL P[],REAL ne[],REAL Rinitial[],REAL Zinitial[],REAL qprofile[]){
  int i;

  FILE *outputFile = fopen(fname,"w");
  fprintf(outputFile,"%4s%11s%11s%11s%11s%11s%11s%11s%11s%11s%11s%11s%11s%11s%11s%11s\n","i,","Rwant","Zwant","a,","theta,","R,","Z,","Bm,","Bpol","T,","Te,","P,","ne,","Rinitial,","Zinitial,","qprofile");
  for(i=0;i<n;i++){
    fprintf(outputFile,"%4d, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g, %9g\n",i,Rwant[i],Zwant[i],a[i],theta[i],R[i],Z[i],Bm[i],Bpol[i],T[i],Te[i],P[i],ne[i],Rinitial[i],Zinitial[i],qprofile[i]);
  }
  fclose(outputFile);
  return 0;
}

//! Store potential data from gyrokinetic simulation in a single file.
/*! Potential data from gyrokinetic simulation is stored in a 3D array:
     phi[nTimeSteps][ntoroidal][mgrid]
     phi: potential in normalized units
     nTimeSteps: number of time steps of potential data
     ntoroidal: number of toroidal grid points (poloidal slices)
     mgrid: number of grid points in a poloidal plane
*/
int writePhiNetcdfFile(char *fname,int nTimeSteps,int ntoroidal,int mgrid,REAL phi[]){
// netcdf identifiers
  int file_id; // id for file
  int time_dim_id, toroidal_dim_id, mgrid_dim_id;
  int d3_dim_ids[ARRAY3D]; // ids for dimensions: time steps,toroidal,poloidal slice; also 3d array of these 3 dimensions
  int phi_id; // ids for 2D variable arrays
     
  int error_code;		// non-zero means netcdf error
  if ((error_code = nc_create(fname, NC_CLOBBER, &file_id))) // create & open the netcdf file
    ERR(error_code);

     // define the dimensions
  if ((error_code = nc_def_dim(file_id, "timesteps", nTimeSteps, &time_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "ntoroidal", ntoroidal, &toroidal_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "mgrid", mgrid, &mgrid_dim_id))) 
    ERR(error_code);

  d3_dim_ids[0] = time_dim_id;	// give lengths of each dimension
  d3_dim_ids[1] = toroidal_dim_id;
  d3_dim_ids[2] = mgrid_dim_id;

   // define the variables
  if ((error_code = nc_def_var(file_id,"potential",NC_REAL,ARRAY3D,d3_dim_ids,&phi_id)))
    ERR(error_code);
  
  // define the unit for each variable
 if ((error_code = nc_put_att_text(file_id, phi_id, UNITS, strlen(KEV), KEV)))
      ERR(error_code);

  if ((error_code = nc_enddef(file_id))) // end define mode
    ERR(error_code);
     
  // write the data to the file
  // 3D arrays
  if ((error_code = nc_put_var_real(file_id,phi_id,phi)))
    ERR(error_code);

  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);
  return 0;
}


//! Store 3D coordinates for reflectometer in cartesian, cylindrical, and flux coordinate systems
int writeFlucCoordFile(char *fname,int nx,int ny,int nz,REAL x[],REAL y[],REAL z[],REAL Rwant[],REAL Zwant[],REAL zeta[],REAL Ract[],REAL Zact[],REAL a[],REAL theta[],REAL Rinitial[],REAL Zinitial[],REAL mag_axis_coords[]){
  // netcdf identifiers
  int file_id; // id for file
  int x_dim_id, y_dim_id,z_dim_id,xyz_dim_ids[ARRAY3D];  // ids for dimensions: interior coords, boundary coords, 2d array of r,z dimensions
  int nx_id,ny_id,nz_id,r_mag_axis_id,z_mag_axis_id; // ids for scalar variables
  int x_id,y_id,z_id,Rwant_id,Zwant_id,zeta_id,Ract_id,Zact_id,a_id,theta_id,Rinitial_id,Zinitial_id; // ids for 3d arrays
     
  int error_code;		// non-zero means netcdf error
  
  if ((error_code = nc_create(fname, NC_CLOBBER, &file_id))) // create & open the netcdf file
    ERR(error_code);

     // define the dimensions
  if ((error_code = nc_def_dim(file_id, "x_dim", nx, &x_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "y_dim", ny, &y_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "z_dim", ny, &z_dim_id))) 
    ERR(error_code);

  xyz_dim_ids[0] = x_dim_id;	// give lengths of each dimension
  xyz_dim_ids[1] = y_dim_id;	// 
  xyz_dim_ids[2] = z_dim_id;	// 

    
  // define the variables
  // scalar variables
if ((error_code = nc_def_var(file_id,"nx",NC_INT,SCALAR,NULL,&nx_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ny",NC_INT,SCALAR,NULL,&ny_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"nz",NC_INT,SCALAR,NULL,&nz_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"R_mag_axis",NC_REAL,SCALAR,NULL,&r_mag_axis_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"Z_mag_axis",NC_REAL,SCALAR,NULL,&z_mag_axis_id)))
    ERR(error_code);
  // 3-D arrays 
  if ((error_code = nc_def_var(file_id,"a",NC_REAL,ARRAY3D,xyz_dim_ids,&a_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"theta",NC_REAL,ARRAY3D,xyz_dim_ids,&theta_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"zeta",NC_REAL,ARRAY3D,xyz_dim_ids,&zeta_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"x",NC_REAL,ARRAY3D,xyz_dim_ids,&x_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"y",NC_REAL,ARRAY3D,xyz_dim_ids,&y_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"z",NC_REAL,ARRAY3D,xyz_dim_ids,&z_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Rwant",NC_REAL,ARRAY3D,xyz_dim_ids,&Rwant_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Zwant",NC_REAL,ARRAY3D,xyz_dim_ids,&Zwant_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"R",NC_REAL,ARRAY3D,xyz_dim_ids,&Ract_id)))
    ERR(error_code);  
  if ((error_code = nc_def_var(file_id,"Z",NC_REAL,ARRAY3D,xyz_dim_ids,&Zact_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Rinitial",NC_REAL,ARRAY3D,xyz_dim_ids,&Rinitial_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Zinitial",NC_REAL,ARRAY3D,xyz_dim_ids,&Zinitial_id)))
    ERR(error_code);
  
  // define the unit for each variable
 if ((error_code = nc_put_att_text(file_id, x_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, y_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, z_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Rwant_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, Zwant_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Ract_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, Zact_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);

  if ((error_code = nc_enddef(file_id))) // end define mode
    ERR(error_code);
     
  // write the data to the file
  // scalars
if ((error_code = nc_put_var_int(file_id,nx_id,&nx)))
    ERR(error_code);
  if ((error_code = nc_put_var_int(file_id,ny_id,&ny)))
    ERR(error_code);
if ((error_code = nc_put_var_int(file_id,nz_id,&nz)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,r_mag_axis_id,&mag_axis_coords[0])))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,z_mag_axis_id,&mag_axis_coords[1])))
    ERR(error_code);
  // 3D arrays
  if ((error_code = nc_put_var_real(file_id,a_id,a)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,theta_id,theta)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,zeta_id,zeta)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,x_id,x)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,y_id,y)))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,z_id,z)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Rwant_id,Rwant)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Zwant_id,Zwant)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Ract_id,Ract)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Zact_id,Zact)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Rinitial_id,Rinitial)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Zinitial_id,Zinitial)))
    ERR(error_code);
  
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);

  return 0;
}

//! Read the 3D reflectometer coordinates file generated by writeFlucCoordFile
/*! Returns the coordinates in cartesian, cylindrical, and flux coordinate systems.
 */
int readFlucCoordFile(char *fname,size_t *nx,size_t *ny,size_t *nz,size_t *ntimesteps,REAL *a[],REAL *theta[],REAL *zeta[],int *timesteps[]){
  // netcdf identifiers
  int file_id; // id for file
  int x_dim_id, y_dim_id,z_dim_id, xyz_dim_ids, timesteps_dim_id;  // ids for dimensions
  int a_id,theta_id,zeta_id,timesteps_id; // ids for variables
     
  int error_code;		// non-zero means netcdf error
     
  
  if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);
     
  if ((error_code = nc_inq_dimid(file_id, "x_dim", &x_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, x_dim_id, nx))) // get the dimension length
    ERR(error_code);
if ((error_code = nc_inq_dimid(file_id, "y_dim", &y_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, y_dim_id, ny))) // get the dimension length
    ERR(error_code);
if ((error_code = nc_inq_dimid(file_id, "z_dim", &z_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, z_dim_id, nz))) // get the dimension length
    ERR(error_code);
  if ((error_code = nc_inq_dimid(file_id, "timesteps_dim", &timesteps_dim_id))) // get the dimension id
    ERR(error_code);

  int space3d = *nx*(*ny)*(*nz)*sizeof(REAL);
  // allocate arrays
  *a = (REAL *)PyMem_Malloc(space3d);
  *theta = (REAL *)PyMem_Malloc(space3d);
  *zeta = (REAL *)PyMem_Malloc(space3d);
  *timesteps = (int *)PyMem_Malloc(*ntimesteps*sizeof(int));
 
     
  // get the variable ids
    if ((error_code = nc_inq_varid(file_id,"a",&a_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"theta",&theta_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"zeta",&zeta_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"timesteps",&timesteps_id)))
    ERR(error_code);
  
  
  // read in the data to memory
    if ((error_code = nc_get_var_real(file_id,a_id,*a)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,theta_id,*theta)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,zeta_id,*zeta)))
    ERR(error_code);
  if ((error_code = nc_get_var_int(file_id,timesteps_id,*timesteps)))
    ERR(error_code);
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);
     

  return 0;
}


// //! Read the specified 3D cartesian (x,y,z) coordinates.
// /*! Used in order to read in the coordinates on which to get the profile and/or fluctuation
//     data.
//  */
// int readSpecified3dCoordFile(char *fname,size_t *nx,size_t*ny,size_t*nz,REAL *x[],REAL *y[],REAL *z[]){
//   // netcdf identifiers
//   int file_id; // id for file
//   size_t Nx,Ny,Nz;
//   int ndims,x_dim_id, y_dim_id,z_dim_id,npts_dim_id;  // id for dimension
//   int x_id,y_id,z_id; // id for variables
     
//   int error_code;		// non-zero means netcdf error
     
  
//   if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
//     ERR(error_code);

// int nc_inq_ndims    (int ncid, int *ndimsp);
//  if ((error_code = nc_inq_ndims(file_id, &ndims))) // get the # of dimensions
//     ERR(error_code);

//  switch(ndims){
//  // case 1:
// //    if ((error_code = nc_inq_dimid(file_id, "npts_dim", &npts_dim_id))) // get the dimension id
// //     ERR(error_code);
// //    if ((error_code = nc_inq_dimlen(file_id, npts_dim_id, npts))) // get the dimension id
// //     ERR(error_code);
// //    break;
//  case 3:
//    if ((error_code = nc_inq_dimid(file_id, "x_dim", &x_dim_id))) // get the dimension id
//      ERR(error_code);
//    if ((error_code = nc_inq_dimlen(file_id, x_dim_id, &Nx))) // get the dimension length
//      ERR(error_code);
//    if ((error_code = nc_inq_dimid(file_id, "y_dim", &y_dim_id))) // get the dimension id
//      ERR(error_code);
//    if ((error_code = nc_inq_dimlen(file_id, y_dim_id, &Ny))) // get the dimension length
//      ERR(error_code);
//    if ((error_code = nc_inq_dimid(file_id, "z_dim", &z_dim_id))) // get the dimension id
//      ERR(error_code);
//    if ((error_code = nc_inq_dimlen(file_id, z_dim_id, &Nz))) // get the dimension length
//      ERR(error_code);
//    //*npts = Nx*Ny*Nz;
//    break;
//  default:
//    READ_SPECIFIED_3D_COORD_FILE_ERR("error reading coordinate file; see documentation for proper structure.");
//  }

//  *nx = Nx;
//  *ny = Ny;
//  *nz = Nz;

//   // allocate arrays
//   *x = (REAL *)PyMem_Malloc(Nx*sizeof(REAL));
//   *y = (REAL *)PyMem_Malloc(Ny*sizeof(REAL));
//   *z = (REAL *)PyMem_Malloc(Nz*sizeof(REAL));
 
     
//   // get the variable ids
//     if ((error_code = nc_inq_varid(file_id,"xx",&x_id)))
//     ERR(error_code);
//   if ((error_code = nc_inq_varid(file_id,"yy",&y_id)))
//     ERR(error_code);
//   if ((error_code = nc_inq_varid(file_id,"zz",&z_id)))
//     ERR(error_code);
  
  
//   // read in the data to memory
//     if ((error_code = nc_get_var_real(file_id,x_id,*x)))
//     ERR(error_code);
//   if ((error_code = nc_get_var_real(file_id,y_id,*y)))
//     ERR(error_code);
//   if ((error_code = nc_get_var_real(file_id,z_id,*z)))
//     ERR(error_code);
     
//   // close the file 
//   if ((error_code = nc_close(file_id)))
//     ERR(error_code);
     
//   return 0;
// }

// //! Read the specified 2D cartesian (R,Z) coordinates.
// /*! Used in order to read in the coordinates on which to get the profile and/or fluctuation
//     data.
//  */
// int readSpecified2dCoordFile(char *fname,int *nr,int*nz,REAL *R[],REAL *Z[]){
//   // netcdf identifiers
//   int file_id; // id for file
//   size_t Nr,Nz;
//   int ndims,R_dim_id, Z_dim_id,npts_dim_id;  // id for dimension
//   int R_id,Z_id; // id for variables
     
//   int error_code;		// non-zero means netcdf error
     
  
//   if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
//     ERR(error_code);

//  if ((error_code = nc_inq_ndims(file_id, &ndims))) // get the # of dimensions
//     ERR(error_code);

//  switch(ndims){
// //  case 1:
// //    if ((error_code = nc_inq_dimid(file_id, "npts_dim", &npts_dim_id))) // get the dimension id
// //     ERR(error_code);
// //    if ((error_code = nc_inq_dimlen(file_id, npts_dim_id, npts))) // get the dimension id
// //     ERR(error_code);
// //    break;
//  case 2:
//    if ((error_code = nc_inq_dimid(file_id, "R_dim", &R_dim_id))) // get the dimension id
//      ERR(error_code);
//    if ((error_code = nc_inq_dimlen(file_id, R_dim_id, &Nr))) // get the dimension length
//      ERR(error_code);
//    if ((error_code = nc_inq_dimid(file_id, "Z_dim", &Z_dim_id))) // get the dimension id
//      ERR(error_code);
//    if ((error_code = nc_inq_dimlen(file_id, Z_dim_id, &Nz))) // get the dimension length
//      ERR(error_code);
//    *nr = Nr;
//    *nz = Nz;
//    //   *npts = Nr*Nz;
//    break;
//  default:
//    READ_SPECIFIED_2D_COORD_FILE_ERR("error reading coordinate file; see documentation for proper structure.");
//  }

  
//   // allocate arrays
//   *R = (REAL *)PyMem_Malloc(Nr*sizeof(REAL));
//   *Z = (REAL *)PyMem_Malloc(Nz*sizeof(REAL));
  
     
//   // get the variable ids
//     if ((error_code = nc_inq_varid(file_id,"rr",&R_id)))
//     ERR(error_code);
//    if ((error_code = nc_inq_varid(file_id,"zz",&Z_id)))
//     ERR(error_code);
  
  
//   // read in the data to memory
//     if ((error_code = nc_get_var_real(file_id,R_id,*R)))
//     ERR(error_code);
//    if ((error_code = nc_get_var_real(file_id,Z_id,*Z)))
//     ERR(error_code);
     
//   // close the file 
//   if ((error_code = nc_close(file_id)))
//     ERR(error_code);
     
//   return 0;
// }

//! Add the fluctuation potential to the 3D reflectometer coordinates file
/*! Uses the 3D coordinates file generated by writeFlucCoordFile
 */
int writeFlucFile(char *fname,int nx,int ny,int nz,REAL phi[]){
  // netcdf identifiers
  int file_id; // id for file
  int x_dim_id, y_dim_id,z_dim_id,xyz_dim_ids[ARRAY3D];  // ids for dimensions: interior coords, boundary coords, 2d array of r,z dimensions
  size_t Nx,Ny,Nz; // ids for scalar variables
  int phi_id; // ids for 3d arrays
     
  int error_code;		// non-zero means netcdf error
  
  if ((error_code = nc_open(fname, NC_WRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);


  // get the dimensions
 if ((error_code = nc_inq_dimid(file_id, "x_dim", &x_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, x_dim_id, &Nx))) // get the dimension length
    ERR(error_code);
if ((error_code = nc_inq_dimid(file_id, "y_dim", &y_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, y_dim_id, &Ny))) // get the dimension length
    ERR(error_code);
if ((error_code = nc_inq_dimid(file_id, "z_dim", &z_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, z_dim_id, &Nz))) // get the dimension length
    ERR(error_code);

  if((nx!=Nx) || (ny!=Ny) || (nz!=Nz))
    WRITE_FLUC_FILE_ERR("Coordinate dimensinos in fluctuations file do not match");

  xyz_dim_ids[0]=x_dim_id;
  xyz_dim_ids[1]=y_dim_id;
  xyz_dim_ids[2]=z_dim_id;


  if ((error_code = nc_redef(file_id))) // enter re-define mode to add a variable
    ERR(error_code);
      
  // define the variables
  // 3-D arrays 
  if ((error_code = nc_def_var(file_id,"phi",NC_REAL,ARRAY3D,xyz_dim_ids,&phi_id)))
    ERR(error_code);


  if ((error_code = nc_enddef(file_id))) // end define mode
    ERR(error_code);
     
  // write the data to the file
  // 3D arrays
  if ((error_code = nc_put_var_real(file_id,phi_id,phi)))
    ERR(error_code);
  
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);

  return 0;
}

int makeFlucFname(char* fname,char* fhead,int time){
  sprintf(fname,"%s%d.cdf",fhead,time);
  return 0;
}


int write2dSnapshot(char* fname, int nr,int nz, REAL R1d[],REAL Z1d[], REAL T[],REAL Te[],REAL B[],REAL ne[])
{
  int file_id;
  int r_dim_id,z_dim_id,rz_dim_ids[2];
  int rr_id,zz_id,ti_id,te_id,bb_id,ne_id;

  int error_code;

  if ((error_code = nc_create(fname, NC_CLOBBER, &file_id))) // create & open the netcdf file
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "r_dim", nr, &r_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "z_dim", nz, &z_dim_id))) 
    ERR(error_code);

  rz_dim_ids[0] = z_dim_id;	// give lengths of each dimension
  rz_dim_ids[1] = r_dim_id;

  if ((error_code = nc_def_var(file_id,"rr",NC_REAL,ARRAY1D,&r_dim_id,&rr_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"zz",NC_REAL,ARRAY1D,&z_dim_id,&zz_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ti",NC_REAL,ARRAY2D,&rz_dim_ids,&ti_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"te",NC_REAL,ARRAY2D,&rz_dim_ids,&te_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"bb",NC_REAL,ARRAY2D,&rz_dim_ids,&bb_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ne",NC_REAL,ARRAY2D,&rz_dim_ids,&ne_id)))
    ERR(error_code);

  if ((error_code = nc_enddef(file_id))) // end define mode
    ERR(error_code);

  if ((error_code = nc_put_var_real(file_id,rr_id,R1d)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,zz_id,Z1d)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,ti_id,T)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,te_id,Te)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,bb_id,B)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,ne_id,ne)))
    ERR(error_code);
  
  if ((error_code = nc_close(file_id)))
    ERR(error_code);

  return 0;
}


//! Stores profile data in a new NetCDF file
int write2dNetcdfFile(char *fname,int nr,int nz,size_t ntimesteps,int nboundary,REAL R1d[],REAL Z1d[],REAL a[],REAL theta[],REAL R[],REAL Z[],REAL Bm[],REAL Bpol[],REAL T[],REAL Te[],REAL P[],REAL ne[],REAL ne_tilde[],REAL phi[],int timesteps[],REAL Rinitial[],REAL Zinitial[],REAL qprofile[],REAL Rboundary[],REAL Zboundary[],REAL mag_axis_coords[]){
  // netcdf identifiers
  int file_id; // id for file
  int r_dim_id, z_dim_id,rbdy_dim_id, rz_dim_ids[ARRAY2D];  // ids for dimensions: interior coords, boundary coords, 2d array of r,z dimensions
  int timesteps_dim_id, rzt_dim_ids[ARRAY3D];
  int nr_id,nz_id,nbdy_id,r_mag_axis_id,z_mag_axis_id; // ids for scalar variables
  int R1d_id,Z1d_id,R_bdy_id,Z_bdy_id,timesteps_id; // ids for 1D variable arrays
  int a_id,theta_id,R_id,Z_id,Bm_id,Bpol_id,T_id,Te_id,P_id,ne_id,ne_tilde_id,phi_id,
    Rinitial_id,Zinitial_id,qprofile_id; // ids for 2D variable arrays
     
  int error_code;		// non-zero means netcdf error
  
  // if(nr!=nz){
//     fprintf(stderr,"writeNetcdfFile: nr (%d) and nz (%d) not equal\n",nr,nz);
//     exit(1);
//   }   
  
  if ((error_code = nc_create(fname, NC_CLOBBER, &file_id))) // create & open the netcdf file
    ERR(error_code);

     // define the dimensions
  if ((error_code = nc_def_dim(file_id, "r_dim", nr, &r_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "z_dim", nz, &z_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "timesteps_dim", ntimesteps, &timesteps_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "dim_404", nboundary, &rbdy_dim_id))) 
    ERR(error_code);

  rz_dim_ids[0] = z_dim_id;	// give lengths of each dimension
  rz_dim_ids[1] = r_dim_id;	//
  rzt_dim_ids[0] = timesteps_dim_id;	// for fluctuation data
  rzt_dim_ids[1] = z_dim_id;	//
  rzt_dim_ids[2] = r_dim_id;	//
  

//   if ((error_code = nc_def_dim(file_id, "dim_101_2", nr*nz, &dim_id2))) // define the dimensions for 2D arrays
//     ERR(error_code);
     
  // define the variables
  // scalar variables
if ((error_code = nc_def_var(file_id,"nr",NC_INT,SCALAR,NULL,&nr_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"nz",NC_INT,SCALAR,NULL,&nz_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"nbdy",NC_INT,SCALAR,NULL,&nbdy_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"r_mag_axis",NC_REAL,SCALAR,NULL,&r_mag_axis_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"z_mag_axis",NC_REAL,SCALAR,NULL,&z_mag_axis_id)))
    ERR(error_code);
  // 1-D arrays
  if ((error_code = nc_def_var(file_id,"rr",NC_REAL,ARRAY1D,&r_dim_id,&R1d_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"zz",NC_REAL,ARRAY1D,&z_dim_id,&Z1d_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"timesteps",NC_INT,ARRAY1D,&timesteps_dim_id,&timesteps_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"rbdy",NC_REAL,ARRAY1D,&rbdy_dim_id,&R_bdy_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"zbdy",NC_REAL,ARRAY1D,&rbdy_dim_id,&Z_bdy_id)))
    ERR(error_code);
  // 2-D arrays 
  if ((error_code = nc_def_var(file_id,"a",NC_REAL,ARRAY2D,rz_dim_ids,&a_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"theta",NC_REAL,ARRAY2D,rz_dim_ids,&theta_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"R",NC_REAL,ARRAY2D,rz_dim_ids,&R_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Z",NC_REAL,ARRAY2D,rz_dim_ids,&Z_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"bb",NC_REAL,ARRAY2D,rz_dim_ids,&Bm_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"bpol",NC_REAL,ARRAY2D,rz_dim_ids,&Bpol_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ti",NC_REAL,ARRAY2D,rz_dim_ids,&T_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"te",NC_REAL,ARRAY2D,rz_dim_ids,&Te_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"qrz",NC_REAL,ARRAY2D,rz_dim_ids,&qprofile_id)))
    ERR(error_code);  
  if ((error_code = nc_def_var(file_id,"P",NC_REAL,ARRAY2D,rz_dim_ids,&P_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ne",NC_REAL,ARRAY2D,rz_dim_ids,&ne_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Rinitial",NC_REAL,ARRAY2D,rz_dim_ids,&Rinitial_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Zinitial",NC_REAL,ARRAY2D,rz_dim_ids,&Zinitial_id)))
    ERR(error_code);
  // 3d arrays
if ((error_code = nc_def_var(file_id,"ne_tilde",NC_REAL,ARRAY3D,rzt_dim_ids,&ne_tilde_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"phi",NC_REAL,ARRAY3D,rzt_dim_ids,&phi_id)))
    ERR(error_code);
  
  // define the unit for each variable
 if ((error_code = nc_put_att_text(file_id, R1d_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Z1d_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Bm_id, UNITS, strlen(TESLA), TESLA)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Bpol_id, UNITS, strlen(TESLA), TESLA)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, T_id, UNITS, strlen(KEV), KEV)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Te_id, UNITS, strlen(KEV), KEV)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, ne_id, UNITS, strlen(M3), M3)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, ne_tilde_id, UNITS, strlen(M3), M3)))
      ERR(error_code);

  if ((error_code = nc_enddef(file_id))) // end define mode
    ERR(error_code);
     
  // write the data to the file
  // scalars
if ((error_code = nc_put_var_int(file_id,nr_id,&nr)))
    ERR(error_code);
  if ((error_code = nc_put_var_int(file_id,nz_id,&nz)))
    ERR(error_code);
if ((error_code = nc_put_var_int(file_id,nbdy_id,&nboundary)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,r_mag_axis_id,&mag_axis_coords[0])))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,z_mag_axis_id,&mag_axis_coords[1])))
    ERR(error_code);
  // 1D arrays
  if ((error_code = nc_put_var_real(file_id,R1d_id,R1d)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Z1d_id,Z1d)))
    ERR(error_code);
if ((error_code = nc_put_var_int(file_id,timesteps_id,timesteps)))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,R_bdy_id,Rboundary)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Z_bdy_id,Zboundary)))
    ERR(error_code);
  // 2D arrays
  if ((error_code = nc_put_var_real(file_id,a_id,a)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,theta_id,theta)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,R_id,R)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Z_id,Z)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Bm_id,Bm)))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,Bpol_id,Bpol)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,T_id,T)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Te_id,Te)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,P_id,P)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,ne_id,ne)))
    ERR(error_code);
   if ((error_code = nc_put_var_real(file_id,Rinitial_id,Rinitial)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Zinitial_id,Zinitial)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,qprofile_id,qprofile)))
    ERR(error_code);
  // 3d arrays
if ((error_code = nc_put_var_real(file_id,ne_tilde_id,ne_tilde)))
   ERR(error_code);
 if ((error_code = nc_put_var_real(file_id,phi_id,phi)))
   ERR(error_code);
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);

  return 0;
}


//! Stores profile data in a new NetCDF file
int write3dNetcdfFile(char *fname,int nx,int ny,int nz,size_t ntimesteps,int nboundary,REAL x1d[],REAL y1d[],REAL z1d[],REAL a[],REAL theta[],REAL zeta[],REAL Rwant[],REAL Zwant[],REAL Ract[],REAL Zact[],REAL Bm[],REAL Bpol[],REAL T[],REAL Te[],REAL P[],REAL ne[],REAL ne_tilde[],REAL phi[],int timesteps[],REAL Rinitial[],REAL Zinitial[],REAL qprofile[],REAL Rboundary[],REAL Zboundary[],REAL mag_axis_coords[]){
  // netcdf identifiers
  int file_id; // id for file
  int x_dim_id, y_dim_id,z_dim_id,rbdy_dim_id, xyz_dim_ids[ARRAY3D];  // ids for dimensions: interior coords, boundary coords, 3d array of x,y,z dimensions
  int timesteps_dim_id, xyzt_dim_ids[ARRAY4D];
  int nx_id,ny_id,nz_id,nbdy_id,r_mag_axis_id,z_mag_axis_id; // ids for scalar variables
  int x1d_id,y1d_id,z1d_id,timesteps_id,R_bdy_id,Z_bdy_id; // ids for 1D variable arrays
  int a_id,theta_id,zeta_id,Rwant_id,Zwant_id,Ract_id,Zact_id,Bm_id,Bpol_id,T_id,Te_id,
    P_id,ne_id,ne_tilde_id,phi_id,Rinitial_id,Zinitial_id,qprofile_id; // ids for 3D variable arrays
     
  int error_code;		// non-zero means netcdf error
     
  if ((error_code = nc_create(fname, NC_CLOBBER, &file_id))) // create & open the netcdf file
    ERR(error_code);

     // define the dimensions
  if ((error_code = nc_def_dim(file_id, "x_dim", nx, &x_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "y_dim", ny, &y_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "z_dim", nz, &z_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "timesteps_dim", ntimesteps, &timesteps_dim_id))) 
    ERR(error_code);
  if ((error_code = nc_def_dim(file_id, "boundary_dim", nboundary, &rbdy_dim_id))) 
    ERR(error_code);

  xyz_dim_ids[0] = z_dim_id;	// give lengths of each dimension
  xyz_dim_ids[1] = y_dim_id;	// 
  xyz_dim_ids[2] = x_dim_id;	// 
  xyzt_dim_ids[0] = timesteps_dim_id;	// for fluctuations
  xyzt_dim_ids[1] = z_dim_id;	// 
  xyzt_dim_ids[2] = y_dim_id;	// 
  xyzt_dim_ids[3] = x_dim_id;	// 
       
  // define the variables
  // scalar variables
if ((error_code = nc_def_var(file_id,"nx",NC_INT,SCALAR,NULL,&nx_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ny",NC_INT,SCALAR,NULL,&ny_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"nz",NC_INT,SCALAR,NULL,&nz_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"R_mag_axis",NC_REAL,SCALAR,NULL,&r_mag_axis_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"Z_mag_axis",NC_REAL,SCALAR,NULL,&z_mag_axis_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"nbdy",NC_INT,SCALAR,NULL,&nbdy_id)))
    ERR(error_code);
    // 1-D arrays
  if ((error_code = nc_def_var(file_id,"x",NC_REAL,ARRAY1D,&x_dim_id,&x1d_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"y",NC_REAL,ARRAY1D,&y_dim_id,&y1d_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"z",NC_REAL,ARRAY1D,&z_dim_id,&z1d_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"timesteps",NC_INT,ARRAY1D,&timesteps_dim_id,&timesteps_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"rbdy",NC_REAL,ARRAY1D,&rbdy_dim_id,&R_bdy_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"zbdy",NC_REAL,ARRAY1D,&rbdy_dim_id,&Z_bdy_id)))
    ERR(error_code);
  // 3-D arrays 
  if ((error_code = nc_def_var(file_id,"a",NC_REAL,ARRAY3D,xyz_dim_ids,&a_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"theta",NC_REAL,ARRAY3D,xyz_dim_ids,&theta_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"zeta",NC_REAL,ARRAY3D,xyz_dim_ids,&zeta_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Rwant",NC_REAL,ARRAY3D,xyz_dim_ids,&Rwant_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Zwant",NC_REAL,ARRAY3D,xyz_dim_ids,&Zwant_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"R",NC_REAL,ARRAY3D,xyz_dim_ids,&Ract_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Z",NC_REAL,ARRAY3D,xyz_dim_ids,&Zact_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"bb",NC_REAL,ARRAY3D,xyz_dim_ids,&Bm_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"bpol",NC_REAL,ARRAY3D,xyz_dim_ids,&Bpol_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ti",NC_REAL,ARRAY3D,xyz_dim_ids,&T_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"te",NC_REAL,ARRAY3D,xyz_dim_ids,&Te_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"qrz",NC_REAL,ARRAY3D,xyz_dim_ids,&qprofile_id)))
    ERR(error_code);  
  if ((error_code = nc_def_var(file_id,"P",NC_REAL,ARRAY3D,xyz_dim_ids,&P_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"ne",NC_REAL,ARRAY3D,xyz_dim_ids,&ne_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Rinitial",NC_REAL,ARRAY3D,xyz_dim_ids,&Rinitial_id)))
    ERR(error_code);
  if ((error_code = nc_def_var(file_id,"Zinitial",NC_REAL,ARRAY3D,xyz_dim_ids,&Zinitial_id)))
    ERR(error_code);
  // 4D arrays
if ((error_code = nc_def_var(file_id,"ne_tilde",NC_REAL,ARRAY4D,xyzt_dim_ids,&ne_tilde_id)))
    ERR(error_code);
if ((error_code = nc_def_var(file_id,"phi",NC_REAL,ARRAY4D,xyzt_dim_ids,&phi_id)))
    ERR(error_code);
  
  // define the unit for each variable
 if ((error_code = nc_put_att_text(file_id, x1d_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, y1d_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, z1d_id, UNITS, strlen(METERS), METERS)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Bm_id, UNITS, strlen(TESLA), TESLA)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Bpol_id, UNITS, strlen(TESLA), TESLA)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, T_id, UNITS, strlen(KEV), KEV)))
      ERR(error_code);
 if ((error_code = nc_put_att_text(file_id, Te_id, UNITS, strlen(KEV), KEV)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, ne_id, UNITS, strlen(M3), M3)))
      ERR(error_code);
if ((error_code = nc_put_att_text(file_id, ne_tilde_id, UNITS, strlen(M3), M3)))
      ERR(error_code);

  if ((error_code = nc_enddef(file_id))) // end define mode
    ERR(error_code);
     
  // write the data to the file
  // scalars
if ((error_code = nc_put_var_int(file_id,nx_id,&nx)))
    ERR(error_code);
  if ((error_code = nc_put_var_int(file_id,ny_id,&ny)))
    ERR(error_code);
  if ((error_code = nc_put_var_int(file_id,nz_id,&nz)))
    ERR(error_code);
if ((error_code = nc_put_var_int(file_id,nbdy_id,&nboundary)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,r_mag_axis_id,&mag_axis_coords[0])))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,z_mag_axis_id,&mag_axis_coords[1])))
    ERR(error_code);
  // 1D arrays
  if ((error_code = nc_put_var_real(file_id,x1d_id,x1d)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,y1d_id,y1d)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,z1d_id,z1d)))
    ERR(error_code);
if ((error_code = nc_put_var_int(file_id,timesteps_id,timesteps)))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,R_bdy_id,Rboundary)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Z_bdy_id,Zboundary)))
    ERR(error_code);
  // 3D arrays
  if ((error_code = nc_put_var_real(file_id,a_id,a)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,theta_id,theta)))
    ERR(error_code);
 if ((error_code = nc_put_var_real(file_id,zeta_id,zeta)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Rwant_id,Rwant)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Zwant_id,Zwant)))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,Ract_id,Ract)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Zact_id,Zact)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Bm_id,Bm)))
    ERR(error_code);
if ((error_code = nc_put_var_real(file_id,Bpol_id,Bpol)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,T_id,T)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Te_id,Te)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,P_id,P)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,ne_id,ne)))
    ERR(error_code);
   if ((error_code = nc_put_var_real(file_id,Rinitial_id,Rinitial)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,Zinitial_id,Zinitial)))
    ERR(error_code);
  if ((error_code = nc_put_var_real(file_id,qprofile_id,qprofile)))
    ERR(error_code);
  // 4d arrays
if ((error_code = nc_put_var_real(file_id,ne_tilde_id,ne_tilde)))
   ERR(error_code);
 if ((error_code = nc_put_var_real(file_id,phi_id,phi)))
   ERR(error_code);
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);

  return 0;
}


//! Reads profile data from NetCDF file generated by writeNetcdfFile
/*! Note that the array pointers need to be passed by reference:
    example usage:
    readNetcdfFile(NETCDF_FILE,&n,&a,&theta,&R,&Z,&Bm,&T,&Te,&P,&ne,&Rinitial,&Zinitial,&qprofile);
    where "int n;" and "REAL *a;" etc. have previously been defined
*/
int readNetcdfFile(char *fname,int *nr,int *nz,REAL *R1d[],REAL *Z1d[],REAL *a[],REAL *theta[],REAL *R[],REAL *Z[],REAL *Bm[],REAL *Bpol[],REAL *T[],REAL *Te[],REAL *P[],REAL *ne[],REAL *Rinitial[],REAL *Zinitial[],REAL *qprofile[]){
  // netcdf identifiers
  int file_id; // id for file
  int r_dim_id, z_dim_id, rz_dim_ids;  // id for dimension
  size_t m;
  int R1d_id,Z1d_id,a_id,theta_id,R_id,Z_id,Bm_id,Bpol_id,T_id,Te_id,P_id,ne_id,Rinitial_id,Zinitial_id,qprofile_id; // id for variables
     
  int error_code;		// non-zero means netcdf error
     
  
  if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);
     
  if ((error_code = nc_inq_dimid(file_id, "dim_101", &r_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, r_dim_id, &m))) // get the dimension length
    ERR(error_code);
  *nr = (int)m;
  *nz = (int)m;
//   if ((error_code = nc_inq_dimid(file_id, "dim_101_2", &dim_id2))) // get the dimension id
//     ERR(error_code);
//   if ((error_code = nc_inq_dimlen(file_id, dim_id2, &m))) // get the dimension length
//     ERR(error_code);
 
//   if(m != *nr*(*nz)){
//     fprintf(stderr,"readNetcdfFile Error: nr != nz\n  nr:%d,nz:%d,m:%d\n",*nr,*nz,m);
//     exit(1);
//   }
  int space1d = m*sizeof(REAL);
  int space2d = m*m*sizeof(REAL);
  // allocate arrays
  *R1d = (REAL *)PyMem_Malloc(space1d);
  *Z1d = (REAL *)PyMem_Malloc(space1d);
  *a = (REAL *)PyMem_Malloc(space2d);
  *theta = (REAL *)PyMem_Malloc(space2d);
  *R = (REAL *)PyMem_Malloc(space2d);
  *Z = (REAL *)PyMem_Malloc(space2d);
  *Bm = (REAL *)PyMem_Malloc(space2d);
*Bpol = (REAL *)PyMem_Malloc(space2d);
  *T = (REAL *)PyMem_Malloc(space2d);
  *Te = (REAL *)PyMem_Malloc(space2d);
  *P = (REAL *)PyMem_Malloc(space2d);
  *ne = (REAL *)PyMem_Malloc(space2d);
  *Rinitial = (REAL *)PyMem_Malloc(space2d);
  *Zinitial = (REAL *)PyMem_Malloc(space2d);
  *qprofile = (REAL *)PyMem_Malloc(space2d);
     
  // get the variable ids
  if ((error_code = nc_inq_varid(file_id,"rr",&R1d_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"zz",&Z1d_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"a",&a_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"theta",&theta_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"R",&R_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Z",&Z_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Bm",&Bm_id)))
    ERR(error_code);
if ((error_code = nc_inq_varid(file_id,"Bpol",&Bpol_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"T",&T_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Te",&Te_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"P",&P_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"ne",&ne_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Rinitial",&Rinitial_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Zinitial",&Zinitial_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"qrz",&qprofile_id)))
    ERR(error_code);
  
  // read in the data to memory
  if ((error_code = nc_get_var_real(file_id,R1d_id,*R1d)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,Z1d_id,*Z1d)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,a_id,*a)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,theta_id,*theta)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,R_id,*R)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,Z_id,*Z)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,Bm_id,*Bm)))
    ERR(error_code);
if ((error_code = nc_get_var_real(file_id,Bpol_id,*Bpol)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,T_id,*T)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,Te_id,*Te)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,P_id,*P)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,ne_id,*ne)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,Rinitial_id,*Rinitial)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,Zinitial_id,*Zinitial)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,qprofile_id,*qprofile)))
    ERR(error_code);
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);
     

  return 0;
}



//! Reads specified 2d coordinate NetCDF file
/*! Note that the array pointers need to be passed by reference, as in other functions
*/
int readSpecified2dCoordFile(char *fname,size_t *nr,size_t *nz,size_t *ntimesteps,REAL *zeta,REAL *R1d[],REAL *Z1d[],int *timesteps[]){
  // netcdf identifiers
  int file_id; // id for file
  int r_dim_id, z_dim_id, timesteps_dim_id;  // ids for dimension
  int R1d_id,Z1d_id,zeta_id,timesteps_id; // ids for variables
     
  int error_code;		// non-zero means netcdf error
  
  #if VERBOSE > 0
   fprintf(stderr,"Reading coordinate file: %s\n",fname);
  #endif
   
  
  if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);
     
  if ((error_code = nc_inq_dimid(file_id, "r_dim", &r_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, r_dim_id, nr))) // get the dimension length
    ERR(error_code);
  if ((error_code = nc_inq_dimid(file_id, "z_dim", &z_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, z_dim_id, nz))) // get the dimension length
    ERR(error_code);
  if ((error_code = nc_inq_dimid(file_id, "timesteps_dim", &timesteps_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, timesteps_dim_id, ntimesteps))) // get the dimension length
    ERR(error_code);
if ((error_code = nc_inq_dimlen(file_id, timesteps_dim_id, ntimesteps))) // get the dimension length
    ERR(error_code);

  // allocate arrays
  *R1d = (REAL *)PyMem_Malloc(*nr*sizeof(REAL));
  *Z1d = (REAL *)PyMem_Malloc(*nz*sizeof(REAL));
  *timesteps = (int *)PyMem_Malloc(*ntimesteps*sizeof(int));
     
  // get the variable ids
  if ((error_code = nc_inq_varid(file_id,"r",&R1d_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"z",&Z1d_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"timesteps",&timesteps_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"zeta",&zeta_id)))
    ERR(error_code);
  
  // read in the data to memory
  if ((error_code = nc_get_var_real(file_id,R1d_id,*R1d)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,Z1d_id,*Z1d)))
    ERR(error_code);
  if ((error_code = nc_get_var_int(file_id,timesteps_id,*timesteps)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,zeta_id,zeta)))
    ERR(error_code);
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);
     
  return 0;
}


//! Reads specified 3d coordinate NetCDF file
/*! Note that the array pointers need to be passed by reference, as in other functions
*/
int readSpecified3dCoordFile(char *fname,size_t *nx,size_t *ny,size_t *nz,size_t *ntimesteps,REAL *x1d[],REAL *y1d[],REAL *z1d[],int *timesteps[]){
  // netcdf identifiers
  int file_id; // id for file
  int x_dim_id, y_dim_id, z_dim_id, timesteps_dim_id;  // ids for dimension
  int x1d_id,y1d_id,z1d_id,timesteps_id; // ids for variables
     
  int error_code;		// non-zero means netcdf error
     
  
  if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);
     
  if ((error_code = nc_inq_dimid(file_id, "x_dim", &x_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, x_dim_id, nx))) // get the dimension length
    ERR(error_code);
 if ((error_code = nc_inq_dimid(file_id, "y_dim", &y_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, y_dim_id, ny))) // get the dimension length
    ERR(error_code);
  if ((error_code = nc_inq_dimid(file_id, "z_dim", &z_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, z_dim_id, nz))) // get the dimension length
    ERR(error_code);
  if ((error_code = nc_inq_dimid(file_id, "timesteps_dim", &timesteps_dim_id))) // get the dimension id
    ERR(error_code);
if ((error_code = nc_inq_dimlen(file_id, timesteps_dim_id, ntimesteps))) // get the dimension length
    ERR(error_code);

  // allocate arrays
  *x1d = (REAL *)PyMem_Malloc(*nx*sizeof(REAL));
  *y1d = (REAL *)PyMem_Malloc(*ny*sizeof(REAL));
  *z1d = (REAL *)PyMem_Malloc(*nz*sizeof(REAL));
  *timesteps = (int *)PyMem_Malloc(*ntimesteps*sizeof(int));
     
  // get the variable ids
  if ((error_code = nc_inq_varid(file_id,"x",&x1d_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"y",&y1d_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"z",&z1d_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"timesteps",&timesteps_id)))
    ERR(error_code);
  
  // read in the data to memory
  if ((error_code = nc_get_var_real(file_id,x1d_id,*x1d)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,y1d_id,*y1d)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,z1d_id,*z1d)))
    ERR(error_code);
  if ((error_code = nc_get_var_int(file_id,timesteps_id,*timesteps)))
    ERR(error_code);
     
  // close the file 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);
     
  return 0;
}


//! Reads a binary record byte count generated by Fortan when writing binary files
/*! Use when reading a binary file created by Fortran
    returns 0 if the number of bytes passed is equal to the byte count from the fortran record
    returns 1 if not
    returns -1 if reached end-of-file
*/
int advanceFortranRecord(FILE *inputFile,int nbytes){
  int fortranRecordByteCount=0;
  if(fread(&fortranRecordByteCount,sizeof(int),1,inputFile) != 1) return -1;
  if(nbytes == fortranRecordByteCount){
    return 0;
  }else{
    fprintf(stderr,"advanceFortranRecord: Bytes read doesn't match Fortran record length\n");
    return 1;
  }
}


//! Remove the duplicate point in the potential array
/*! Use if all the data points must be unique.  The potential data array generated by the
    gyrokinetic code has a duplicate point on each flux surface at theta=2*pi which is identical
    to the value on the same flux surface at theta=0.  The resulting 3D array has mpsi fewer
    elements in each column, where mpsi is the number of "psi" grid points (flux surfaces).
 */
int phiArrRemoveDuplicates(int nTimeSteps, int ntoroidal,int mgrid,int mpsi,int mtheta[],REAL ***phiOld,REAL ****phiNew,int *mgridNew){
  int *sumTheta = (int*)PyMem_Malloc((mpsi+1)*sizeof(int));
  sumTheta[0]=0;
  int i,j,k,m;
  for(i=0;i<mpsi;i++) sumTheta[i+1] = sumTheta[i] + mtheta[i];
//   int mgrid_no_duplicates=0;
//   for(i=0;i<mpsi;i++) mgrid_no_duplicates = mgrid_no_duplicates + mtheta[i];

  REAL ***phi = realAllocContiguous3d(nTimeSteps,ntoroidal,sumTheta[mpsi]);
  for(i=0;i<nTimeSteps;i++)
    for(j=0;j<ntoroidal;j++)
      for(k=0;k<mpsi;k++)
	for(m=0;m<mtheta[k];m++)
	  phi[i][j][sumTheta[k] + m] = phiOld[i][j][sumTheta[k]+m+k];

    *phiNew = phi;
//     *mgridNew = mgrid_no_duplicates;
    *mgridNew = sumTheta[mpsi];
  return 0;
}

//! Reads all potential files in a given directory
/*! Reads all potential files in a given directory, of the form PHI.xxxxx , where "xxxxx" is 
    the toroidal index (poloidal slice) for the potential fluctuations it contains.  This
    function can read in data from as many time steps as desired, and also reads in other
    useful information:
    qtinv, mtheta, igrid, ntoroidal, mpsi
 */
int readAllPhiFiles(char *path,int nTimeSteps,int timeSteps[],int phi_integers[],REAL phi_reals[],int *igrid[],int *mtheta[],int *itran[],REAL *qtinv[],REAL *deltat[],REAL *vth_grid[],REAL ***phi[],int *nsteps,int npts,REAL zeta[]){

  if(nTimeSteps < 1){		// need to read in at least one time step
    READALLPHIFILES_ERR("nTimeSteps must be >= 1");
  }

  // read in the first potential file PHI.00000 to determine the number of files
  //   and get other information

  char fname[FNAME_MAX_CHARS];
  int itoroidal = 0;
  REAL **phi_steps=NULL;
  int status = constructPhiFname(fname,path,itoroidal);
  status = readPhiFile(fname,phi_integers,phi_reals,igrid,mtheta,itran,qtinv,deltat,vth_grid,&phi_steps,nsteps);

  int mgrid = phi_integers[3];	   // unpack # of grid points at each toroidal location
  int ntoroidal = phi_integers[5]; // unpack the number of toroidal grid points (number of PHI.000xx files)
  
  #if DEBUG > 0
  ntoroidal = 16;		// for development, so we don't need to read as many files
  #endif

  // allocate 3d array to store the potential fluctuations: phi[nTimeSteps][ntoroidal][mgrid]
  int i,j;
  //Lei Shi temp
  REAL MaxZeta;
  MaxZeta=findMax(npts,zeta);
  REAL dzeta=2*M_PI/ntoroidal;
  int MaxNtor=(int)floor(MaxZeta/dzeta)+2;//the actual number of toroidal slices we need to read is the number of zeta pieces fully covered by our zeta region + 2,note that we need an additional zeta piece to cover our zeta region and that means +2 toroidal slices.
  if(MaxNtor>ntoroidal)// if the total number exceeds ntoroidal, only happens if we need the full region, and we count the starting plane twice since we didn't consider the toroidal periodicity above.
    MaxNtor=ntoroidal;
  REAL ***phi_arr = realAllocContiguous3d(nTimeSteps,MaxNtor,mgrid);
  *phi = phi_arr;
  
  // copy the potential fluctuations from the timesteps of interest to this new array
  for(i=0;i<nTimeSteps;i++)
    for(j=0;j<mgrid;j++) phi_arr[i][0][j] = phi_steps[timeSteps[i]][j];
  // free memory associated with phi_steps, since another block will be assigned with each call to readPhiFile
  realFreeDiscontiguous2d(*nsteps,&phi_steps);
  

  // loop reading all the other potential files, keeping only the potential fluc. for the timestep of interest
  //Lei Shi temp
  for(itoroidal=1;itoroidal<MaxNtor;itoroidal++){
    status = constructPhiFname(fname,path,itoroidal);
    status = readPhiFile(fname,phi_integers,phi_reals,igrid,mtheta,itran,qtinv,deltat,vth_grid,&phi_steps,nsteps);
    for(i=0;i<nTimeSteps;i++)
      for(j=0;j<mgrid;j++) phi_arr[i][itoroidal][j] = phi_steps[timeSteps[i]][j];
    realFreeDiscontiguous2d(*nsteps,&phi_steps); // free memory since it will be reassigned on the next loop
  }
#if DEBUG > 0
  phi_integers[5] = ntoroidal;	// for development
#endif
  return 0;
}

//! Read a single potential file generated from a gyrokinetic simulation.
/*! Reads a single potential file (typically of the form PHI.xxxxx) which contains the
    potential fluctuation at a single toroidal location (poloidal slice).
 */
int readPhiFile(char *fname,int phi_integers[],REAL phi_reals[N_PHI_REALS],int *igrid[],int *mtheta[],int *itran[],REAL *qtinv[],REAL *deltat[],REAL *vth_grid[],REAL **phi_steps[],int *nsteps){

  int status=0;
  int fortranRecordByteCount=sizeof(int)*(int)N_PHI_INTEGERS + sizeof(REAL)*(int)N_PHI_REALS;
#if VERBOSE > 0
  fprintf(stderr,"Reading: %s\n",fname);
#endif
  FILE *inputFile = fopen(fname,"rb");	// open file for reading
  if(inputFile == NULL) READPHI_ERR("couldn't open file");
  if(advanceFortranRecord(inputFile,fortranRecordByteCount))
    READPHI_ERR("Fortran record byte count mismatch"); // begin record

  if(fread(phi_integers,sizeof(int),N_PHI_INTEGERS,inputFile) != N_PHI_INTEGERS)
    READPHI_ERR("integer header data read error");
  // phi_integers is an array in order: msnap,myrank_toroidal,mpsi,mgrid,mzetamax,ntoroidal

  if(fread(phi_reals,sizeof(REAL),N_PHI_REALS,inputFile) != N_PHI_REALS)
    READPHI_ERR("real header data read error");
  // phi_reals is an array in order: zetamin,zetamax,deltaz,vthc
  if(advanceFortranRecord(inputFile,fortranRecordByteCount))
    READPHI_ERR("Fortran record byte count mismatch"); // end record

  // integer scalars
  int mzeta =1 ;		// only reading 1 poloidal slice

  int msnap = phi_integers[0];
  int myrank_toroidal = phi_integers[1];
  phi_integers[2] += 1;//(phi_integers[2])++; // fortran uses array range from 0:mpsi, so it has mpsi+1 elements
  int mpsi = phi_integers[2];	// I change mpsi so the array 0:mpsi-1 with mpsi elems
  int mgrid = phi_integers[3];	// unlike the fortran convention, phi_steps[isteps][0:mgrid-1]
  int mzetamax = phi_integers[4];
  int ntoroidal = phi_integers[5];
  
  int space1dint = mpsi*sizeof(int);
  int space1dreal = mpsi*sizeof(REAL);
  int space2d = mzeta*mgrid*sizeof(REAL);

  // fprintf(stderr,"msnap:%d, ",msnap);
//   fprintf(stderr," myrank_toroidal:%d, ",myrank_toroidal);
//   fprintf(stderr," mpsi:%d, ",mpsi);
//   //  fprintf(stderr," mzeta:%d\n",mzeta); // mzeta=1 since we're only reading one toroidal location
//   fprintf(stderr,"mgrid:%d, ",mgrid);
//   fprintf(stderr," mzetamax:%d, ",mzetamax);
//   fprintf(stderr," ntoroidal:%d\n",ntoroidal);

//   fprintf(stderr,"phireals: %g, %g, %g, %g\n",phi_reals[0],phi_reals[1],phi_reals[2],phi_reals[3]);


 // allocate arrays
  *igrid = (int *)PyMem_Realloc((void*)*igrid,space1dint); // use realloc instead of malloc so that if this function
  *mtheta = (int *)PyMem_Realloc((void*)*mtheta,space1dint);	// is called several times with the same arguments
  *itran = (int *)PyMem_Realloc((void*)*itran,space1dint); // the memory assigned to each of these arrays
  *qtinv = (REAL *)PyMem_Realloc((void*)*qtinv,space1dreal); // is freed automatically and reassigned
  *deltat = (REAL *)PyMem_Realloc((void*)*deltat,space1dreal);
  *vth_grid = (REAL *)PyMem_Realloc((void*)*vth_grid,space1dreal);


  // read data into arrays
fortranRecordByteCount =
  space1dint*N_PHI_1DINT_ARR + space1dreal*N_PHI_1DREAL_ARR;
if(advanceFortranRecord(inputFile,fortranRecordByteCount))
  READPHI_ERR("Fortran record byte count mismatch"); // begin record
  if(fread(*igrid,sizeof(int),mpsi,inputFile) != mpsi)
    READPHI_ERR("igrid data read error");
if(fread(*deltat,sizeof(REAL),mpsi,inputFile) != mpsi)
  READPHI_ERR("deltat data read error");
  if(fread(*qtinv,sizeof(REAL),mpsi,inputFile) != mpsi)
    READPHI_ERR("qtinv data read error");
if(fread(*mtheta,sizeof(int),mpsi,inputFile) != mpsi)
  READPHI_ERR("mtheta data read error");
  if(fread(*itran,sizeof(int),mpsi,inputFile) != mpsi)
    READPHI_ERR("itran data read error");
  if(fread(*vth_grid,sizeof(REAL),mpsi,inputFile) != mpsi)
    READPHI_ERR("vth_grid data read error");
if(advanceFortranRecord(inputFile,fortranRecordByteCount))
  READPHI_ERR("Fortran record byte count mismatch"); // end record

  int i=0;
  int *istep=NULL;
  while(~feof(inputFile)){
//     fprintf(stderr,"i:%d\n",i);
    fortranRecordByteCount = sizeof(int);
    status = advanceFortranRecord(inputFile,fortranRecordByteCount);
    if(status == -1){// begin record
      break;			// break if at end of file
    }else if(status == 1)
      READPHI_ERR("Fortran record byte count mismatch"); // exit with error if the fortran record byte count is wrong
    istep = (int*) PyMem_Realloc ((void*)istep, sizeof(int)*(i+1));
    if(fread(&istep[i],sizeof(int),1,inputFile) != 1) READPHI_ERR("istep data read error");
    if(advanceFortranRecord(inputFile,fortranRecordByteCount))
      READPHI_ERR("Fortran record byte count mismatch"); // end record

    // calling program must free this memory if this function
    //   is called multiple times with the same arguments
    *phi_steps = (REAL **)PyMem_Realloc((void**)*phi_steps,sizeof(REAL *)*(i+1));
    (*phi_steps)[i] = (REAL *)PyMem_Malloc(space2d);
    
    fortranRecordByteCount = mzeta*(mgrid)*sizeof(REAL); 
    if(advanceFortranRecord(inputFile,fortranRecordByteCount))
      READPHI_ERR("Fortran record byte count mismatch"); // begin record
    if(fread((*phi_steps)[i],sizeof(REAL),mzeta*mgrid,inputFile) != mzeta*mgrid)
      READPHI_ERR("phi data read error");
    if(advanceFortranRecord(inputFile,fortranRecordByteCount))
      READPHI_ERR("Fortran record byte count mismatch"); // end record
    i++;
  }
  *nsteps = i;			// I don't do anything with the istep[nstep] array

  status = fclose(inputFile);

  return status;
}


//! Constructs file name for a potential file given the path and toroidal index
int constructPhiFname(char *fname,char *path,int itoroidal){
    sprintf(fname,"%s" PHI_FNAME_START "%05d",path,itoroidal);
    return 0;
  }


//! Reads "psi" grid (grid of flux surfaces) generated by gyrokinetic simulation
//  Read psi grid from NTprofile.cdf

REAL *readPsiGridFile(char* fname,int mpsi){

  int file_id;
  int psi_dim_id,psi_id;
  REAL* psi_grid;
  int error_code;
  int m;
  if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);
     
  if ((error_code = nc_inq_dimid(file_id, "mpsip1", &psi_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id, psi_dim_id, &m))) // get the dimension length
    ERR(error_code);
  if(m != mpsi)
    READPSI_ERR(fname,mpsi,m);
  psi_grid= (REAL*)PyMem_Malloc(m*sizeof(REAL));

  if ((error_code = nc_inq_varid(file_id,"psi_grid",&psi_id)))
    ERR(error_code);
  if ((error_code = nc_get_var_real(file_id,psi_id,psi_grid)))
    ERR(error_code); 
  if ((error_code = nc_close(file_id)))
    ERR(error_code);
  return psi_grid;
}



/*! Reads in the "psi" grid data from the "dsda_psi.dat" file.
 */
/*
REAL *readPsiGridFile(char *fname,int mpsi){
  int npts=0;
  FILE *inputFile = fopen(fname,"r");
#if VERBOSE > 0
  fprintf(stderr,"Reading: %s\n",fname);
#endif

  if(inputFile == NULL) READPSI_ERR(fname,mpsi,npts);
  int c;
  while ((c = fgetc(inputFile)) != EOF) {
    if(c == '=') break;
  }
  if(c == EOF) READPSI_ERR(fname,mpsi,npts);
  fscanf(inputFile,"%*[^\n]");	// skip to the end of the line
  fscanf(inputFile,"%*1[\n]");  // skip one newline
  // while ((c = fgetc(inputFile)) != EOF) {
//     if(c == '\n') break;
//   }
//   if(c == EOF) READPSI_ERR(fname,mpsi,npts);
  REAL adum,temperature,stuff,s_grid;
  REAL *psi_grid=NULL;

  while (!feof(inputFile)) {
    psi_grid = (REAL *)PyMem_Realloc((void*)psi_grid,sizeof(REAL)*(npts+1));
    if(psi_grid==NULL) READPSI_ERR(fname,mpsi,npts);
    if (fscanf(inputFile, "%lg %lg %lg %lg %lg\n",&adum,&temperature,&stuff,&s_grid,&psi_grid[npts]) != 5)
      break;
    npts++;
  }
  if(fclose(inputFile)) READPSI_ERR(fname,mpsi,npts);
  if(npts != mpsi) READPSI_ERR(fname,mpsi,npts);
  return psi_grid;
}
*/

#endif
