/*! \file profile_coord_map.h
    \brief Grid initialization, finding flux coordinates, and getting profile data

    Called by all the top-level functions.
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_vector.h>	// needed for 2-d root-finding
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_linalg.h>


#ifndef REAL_TYPEDEF
#define REAL_TYPEDEF
#ifndef single			// compiler option determines variable size
typedef double REAL;
#else
typedef float REAL;
#endif
#endif

#ifndef DEBUG			// default debug level: no output to stderr
#define DEBUG 0
#endif

#ifndef DEVELOPMENT		// default development level: use the best of what has been tested to work
#define DEVELOPMENT 0
#endif

#ifndef VERBOSE
#define VERBOSE 0
#endif

#define MAX_N_TRIES 4		// number of times to try to get the root-finder unstuck

// used by desiredrz: constants to choose the desired grid in R,Z space
#define RMIN 2.7
#define RMAX 3.4
#define ZMIN -0.2
#define ZMAX 0.2

// used by desiredxyz: constants for desired 3d grid in x,y,z
#define xMAX 2.2
#define xMIN 1.9
#define yMAX 0.5
#define yMIN -0.5
#define zMAX 0
#define zMIN -0.4


// used by getAllProfiles
#define NORM_TO_KEV 4.7894e4      // conversion constant from normalized units to keV (called Tem_00 in fortran files)  Normalization in GTS: Length 100cm, Time Deuterium Cyclotron frequency in 1T B field -- OMEGA_D^-1,  
// T normalized st. vth = sqrt(T).  Temperature has the energy dimention.

#define INVCM3_TO_INVM3 1e6	// convert cm^{-3} to m^{-3}


// used by decayNToutsideLCFS and getBoundaryPoints
#define ABOUNDARY 0.9
#define AINSIDE 0.88
#define AZERO 1.2
//#define DECAY_SCALING 0.1*(ABOUNDARY-AINSIDE) // the 0.1 makes it drop off faster

#define MINIMUM(x,y) ((x) < (y) ? (x) : (y))

extern const char* NTFILENAME;

// ---------- local function declarations
int cylToLargeAR(REAL R[],REAL Z[], REAL r[], REAL theta[],int n,REAL R0,REAL Z0);
int largeARtoCyl(REAL r[],REAL theta[], REAL R[], REAL Z[],int n,REAL R0,REAL Z0);
int desiredrz(REAL R[],REAL Z[],REAL R1d[],REAL Z1d[],int nr,int nz,REAL R0,REAL Z0);
int desiredxyz(int nx,int ny,int nz,REAL x1d[],REAL y1d[],REAL z1d[],REAL x[],REAL y[],REAL z[],REAL R0,REAL Z0);
int mesh2dGrid(REAL x2d[],REAL y2d[],REAL x1d[],REAL y1d[],int nx,int ny);
int mesh3dGrid(REAL x3d[],REAL y3d[],REAL z3d[],REAL x1d[],REAL y1d[],REAL z1d[],int nx,int ny,int nz);

int guessatheta(REAL a[],REAL theta[], REAL R[], REAL Z[],int n,REAL R0,REAL Z0);
REAL error_amt(REAL R,REAL Z,REAL a,REAL theta);
void print_state(size_t iter, gsl_multiroot_fsolver * s);
int findcorrectatheta(REAL a[],REAL theta[],REAL Ract[],REAL Zact[],REAL Bm[],REAL Rwant[], REAL Zwant[],int n,int flag[]);
int coord_f(const gsl_vector *x,void *params,gsl_vector *f);
int getBoundaryPoints(double R_bdy[],double Z_bdy[],int n_bdy);

int get_mag_axis(REAL coords[]);
int decayNToutsideLCFS(int npts,REAL a[],REAL ne[],REAL Te[],REAL Ti[],int* flag);
int getFluxCoords(int npts,REAL a[],REAL theta[],REAL Bm[],REAL Ract[],REAL Zact[],REAL Rinitial[],REAL Zinitial[],REAL Rwant[],REAL Zwant[],REAL mag_axis_coords[],int flag[]);
int getAllProfiles(int npts,REAL Bpol[],REAL Ti[],REAL Te[],REAL P[],REAL ne[],REAL qprofile[],REAL a[],REAL theta[],int InOutFlag[]);

int getProfile(REAL a,REAL* ne,REAL* Ti,REAL* Te,REAL* a_p,REAL* n_p,REAL* Ti_p,REAL* Te_p,int n_prof );

int adiabaticElectronResponse(int npts,int ntimesteps,REAL ne_tilde[],REAL ne0[],REAL phi[],REAL Te[],int flag[]);

//modified by Lei Shi adding function to find the outside points

//declaration
void inner_check(double R[],double Z[],double R_bdy[],double Z_bdy[],int flag[],int n,int n_bdy,double tol);
//R,Z contains grid point coordinates, R_bdy Z_bdy contains boundary points, flag show inside(with value 1) and outside(with value 0), n the number of grid points, n_bdy the number of boundary points.

//if (R[i],Z[i]) is inside or on the boundary of the contour specified by R_bdy and Z_bdy, flag[i] is set to 1, otherwise 0; 

//definition

int horipos_check(double Rc,double Zc, double R1,double Z1,double R2,double Z2,double tol){
  double R=R1+ (Zc-Z1)/(Z2-Z1)*(R2-R1);

  if(fabs(R-Rc)<=tol)
    return 2;
  else if(R<Rc)
    return 1;
  else
    return -1;
}

void inner_check(double R[],double Z[],double R_bdy[],double Z_bdy[],int flag[],int n,int n_bdy,double tol){
  
  int i;

  for(i=0;i<n;i++){
    double Rc=R[i];
    double Zc=Z[i];
    
    flag[i]=0;
    
    int vertpos=0;
    int horipos1=0;
    int horipos2=0;
    int j;
    for(j=0;j<n_bdy;j++){
      if(Z_bdy[j]==Zc){
	if (horipos1==0)
	  {
	    if(fabs(R_bdy[j]-Rc) < tol)
	      horipos1=2;
	    else if(R_bdy[j]<Rc)
	      horipos1=1;
	    else
	      horipos1=-1;
	    continue;
	  }
	else{
	  {
	    if(fabs(R_bdy[j]-Rc) < tol)
	      horipos2=2;
	    else if(R_bdy[j]<Rc)
	      horipos2=1;
	    else
	      horipos2=-1;
	    flag[i]= (horipos1 != horipos2)?1:0;
	    break;
	  }	  
	}
      }
      else if(Z_bdy[j]<Zc){
	if(vertpos==0){
	  vertpos = -1;
	  continue;
	}
	else if(vertpos==-1)
	  continue;
	else{
	  if(horipos1 == 0){
	    vertpos = -1;
	    horipos1 = horipos_check(Rc,Zc,R_bdy[j],Z_bdy[j],R_bdy[j-1],Z_bdy[j-1],tol);
	    continue;
	  }
	  else{
	    horipos2=horipos_check(Rc,Zc,R_bdy[j],Z_bdy[j],R_bdy[j-1],Z_bdy[j-1],tol);
	    flag[i]= (horipos1 != horipos2)?1:0;
	    break;
	  }
	  
	}
	
      }
      else{
	if(vertpos==0){
	  vertpos = 1;
	  continue;
	}
	else if(vertpos == 1)
	  continue;
	else{
	  if(horipos1 == 0){
	    vertpos = 1;
	    horipos1 = horipos_check(Rc,Zc,R_bdy[j],Z_bdy[j],R_bdy[j-1],Z_bdy[j-1],tol);
	    continue;
	  }
	  else{
	    horipos2=horipos_check(Rc,Zc,R_bdy[j],Z_bdy[j],R_bdy[j-1],Z_bdy[j-1],tol);
	    flag[i]= (horipos1 != horipos2)?1:0;
	    break;
	  }	  
	}
      }
    }
  }  
}

//end modification

// ---------- data structures
struct coordparams{ 		// structure for 2D root finding
  REAL Rwant;
  REAL Zwant;
};

// ---------- function definitions

//! Convert cylindrical coords (R,Z) to a large aspect ratio approximation of (r,theta)
/*! For n points, where (R0,Z0) is the position of the magnetic axis */
int cylToLargeAR(REAL R[],REAL Z[], REAL r[], REAL theta[],int n,REAL R0,REAL Z0){
  REAL x,z;
  int i;  

  for(i=0;i<n;i++){		// here I use: R-R0 = a*cos(theta)
    x = R[i]-R0;		//    and Z = a*sin(theta)
    z = Z[i]-Z0;
    theta[i]=-atan2(z,x);	// poloidal angle (correct quadrant)
    //    r[i] = sqrt(x*x+z*z);	// minor rad.
    r[i] = hypot(x,z);
  }

  return 0;
}

//! Convert a large aspect ratio (r,theta) to cylindrical coords (R,Z)
/*! For n points, where (R0,Z0) is the position of the magnetic axis */
int largeARtoCyl(REAL r[],REAL theta[], REAL R[], REAL Z[],int n,REAL R0,REAL Z0){
  int i; 
  for(i=0;i<n;i++){		// here I use: R-R0 = a*cos(theta)
    R[i] = R0 + r[i]*cos(theta[i]);
    Z[i] = Z0 - r[i]*sin(theta[i]);
  }

  return 0;
}

//! Generate a regular, 2D array meshes in (R,Z), and also 1D arrays for each dimension
/*! 2D arrays vary fastest along R (columns) and slowly along Z (rows).
  Also generates 1D arrays R1d, Z1d.
  nr: points in R, nz: # points in z; total # points = nr*nz
  R0,Z0 are the coordinates for the magnetic axis, which is not currently used.
*/
int desiredrz(REAL R[],REAL Z[],REAL R1d[],REAL Z1d[],int nr,int nz,REAL R0,REAL Z0){
  int i;
  REAL dr = (RMAX-RMIN)/(nr-1);
  REAL dz = (ZMAX-ZMIN)/(nz-1);
  for(i=0;i<nz;i++) Z1d[i] = (REAL)ZMIN + i*dz;
  for(i=0;i<nr;i++) R1d[i] = (REAL)RMIN + i*dr;
  mesh2dGrid(R,Z,R1d,Z1d,nr,nz);
  // printf("desiredrz: dr:%g,dz:%g\n",dr,dz);
  // for(i=0;i<nz;i++){
//     Z1d[i] = (REAL)ZMIN + i*dz;
//     for(j=0;j<nr;j++){
//       R[i*nr+j] = (REAL)RMIN + j*dr;
//       Z[i*nr+j] = (REAL)ZMIN + i*dz;
//     }
//   }
  return 0;
}

//! Generate a regular, 3D array meshes in (x,y,z), and also 1D arrays for each dimension
/*! 3D arrays vary fastest along x (columns), then along y (rows), slowest along z (pages).
  x is locally radially outward, y is vertically upward, z is locally toroidal to make a
  right-hand coordinate system.
  nx: points in x direction, ny: points in y direction, nz: points in z direction;
  total # points = nx*ny*nz.
  R0,Z0 are the coordinates for the magnetic axis, which is not currently used.
*/
int desiredxyz(int nx,int ny,int nz,REAL x1d[],REAL y1d[],REAL z1d[],REAL x[],REAL y[],REAL z[],REAL R0,REAL Z0){
  int i,j,k;
  REAL dx = (xMAX-xMIN)/(nx-1);
  REAL dy = (yMAX-yMIN)/(ny-1);
  REAL dz = (zMAX-zMIN)/(nz-1);
#if DEBUG > 1
  fprintf(stderr,"desiredrz: dx:%g,dy:%g,dz:%g\n",dx,dy,dz);
#endif
  for(i=0;i<nz;i++) z1d[i] = (REAL)zMIN + i*dz;
  for(i=0;i<ny;i++) y1d[i] = (REAL)yMIN + i*dy;
  for(i=0;i<nx;i++) x1d[i] = (REAL)xMIN + i*dx;
  mesh3dGrid(x,y,z,x1d,y1d,z1d,nx,ny,nz);
  // for(i=0;i<nz;i++){
//     z1d[i] = (REAL)zMIN + i*dz;
//     for(j=0;j<ny;j++){
//       for(k=0;k<nx;k++){
// 	x[i*ny*nx+j*nx+k] = (REAL)xMIN + k*dx;
// 	y[i*ny*nx+j*nx+k] = (REAL)yMIN + j*dy;
// 	z[i*ny*nx+j*nx+k] = (REAL)zMIN + i*dz;
//       }
//     }
//   } 
  return 0;
}

//! Generate 2d mesh grids from 1d grids in each dimension
/*! Takes 1d arrays of coordinates x1d,y1d and generates 2d arrays
    x2d,y2d of nx columns and ny rows, corresponding to the x,y
    coordinate of each point.
*/
int mesh2dGrid(REAL x2d[],REAL y2d[],REAL x1d[],REAL y1d[],int nx,int ny){
  int i,j;
  for(i=0;i<ny;i++){
    for(j=0;j<nx;j++){
      x2d[i*nx+j] = x1d[j];
      y2d[i*nx+j] = y1d[i];
    }
  }

  return 0;
}

//! Generate 3d mesh grids from 1d grids in each dimension
/*! Takes 1d arrays of coordinates x1d,y1d,z1d and generates 3d arrays
    x3d,y3d,z3d of nx columns, ny rows, and nz pages corresponding to the x,y,z
    coordinates of each point.
*/
int mesh3dGrid(REAL x3d[],REAL y3d[],REAL z3d[],REAL x1d[],REAL y1d[],REAL z1d[],int nx,int ny,int nz){
  int i,j,k;
 for(i=0;i<nz;i++){
     for(j=0;j<ny;j++){
      for(k=0;k<nx;k++){
	x3d[i*ny*nx+j*nx+k] = x1d[k];
	y3d[i*ny*nx+j*nx+k] = y1d[j];
	z3d[i*ny*nx+j*nx+k] = z1d[i];
      }
    }
  }

    return 0;
}




//! Given cylindrical coords (R,Z) calculates the initial guess for the flux coords (a,theta)
// first guess as to what the correct a,theta values are for the desired R,Z points
// this is the "large aspect ratio, circ. xsection where I take a = minor radius and
// theta is simply poloidal angle
// then I modify the angle by doing an interpolation
// n is total # of points in 1d array
int guessatheta(REAL a[],REAL theta[], REAL R[], REAL Z[],int n,REAL R0,REAL Z0){
  int i,k;
  int n1dgrid = (int)sqrt(n);
  int ngrid = n1dgrid*n1dgrid;//+10;

#if DEVELOPMENT >= 0

  REAL rwant[n],polwant[n];
  k = cylToLargeAR(R,Z,rwant,polwant,n,R0,Z0);// the minor radius and polodial angles of the R,Z we want

  // --- current method: 2D interpolation for 2 variables:
  //       (rgrid,polgrid) -> a
  //       (rgrid,polgrid) -> theta
  REAL agrid[ngrid],thetagrid[ngrid],Rgrid[ngrid],Zgrid[ngrid];
  REAL rgrid[ngrid],polgrid[ngrid],Bgrid[ngrid];
 
  // set up a grid to interpolate on
  // use to get an even grid in a,theta
  double da = 1.5/(n1dgrid-1);
  double dtheta = 2.0*M_PI/n1dgrid;
  for(i=0;i<n1dgrid;i++){
    for(k=0;k<n1dgrid;k++){
      agrid[i*n1dgrid+k] = 0.5*da + i*da;
      thetagrid[i*n1dgrid+k] = -M_PI+k*dtheta;
    }
  }

  k = esigetrzb_(Rgrid,Zgrid,Bgrid,agrid,thetagrid,&ngrid); // get Rgrid and Zgrid
  k = cylToLargeAR(Rgrid,Zgrid,rgrid,polgrid,ngrid,R0,Z0);	// get the grid minor radius and poloidal angle

  // interpolate using (rgrid,polgrid) -> (agrid,thetagrid)
  // to get (rwant,thetawant) -> (a,theta)
  k = interp2d(n,ngrid,rwant,polwant,a,theta,rgrid,polgrid,agrid,thetagrid);

#if DEBUG > 0
  fprintf(stderr,"point  agrid   thetagrid  Rgrid  Zgrid  rgrid  polgrid\n");
  for(i=0;i<ngrid;i++)
    fprintf(stderr,"%5d, %g, %g, %g, %g, %g, %g\n",i,agrid[i],thetagrid[i],
	    Rgrid[i],Zgrid[i],rgrid[i],polgrid[i]);
  fprintf(stderr,"point  a   theta  R  Z  rwant  polwant\n");
  for(i=0;i<n;i++){
    fprintf(stderr,"%5d, %g, %g, %g, %g, %g, %g\n",i,a[i],theta[i],
	    R[i],Z[i],rwant[i],polwant[i]);
    
  }

#endif

    for(i=0;i<n;i++) if(a[i]>2.5) a[i]=2.5;	// limit the maximum radial flux coord

#endif


  // // --- previous method: 1D interpolation of both variables independently
  //   //        polgrid -> (agrid=1,thetagrid)
  //   //        rgrid   -> (agrid,thetagrid=pi/6)

  //   REAL agrid[n1dgrid],thetagrid[n1dgrid],Rgrid[n1dgrid],Zgrid[n1dgrid];
  //   REAL rgrid[n1dgrid],polgrid[n1dgrid],Bgrid[n1dgrid];

  //   // first do the poloidal interpolation
  //   double dtheta = 2*M_PI/n1dgrid;
  //     for(i=0;i<n1dgrid;i++){
  //       agrid[i] = 1;
  //     thetagrid[i] = -M_PI+i*dtheta;
  //     }
  //   k = esigetrzb_(Rgrid,Zgrid,Bgrid,agrid,thetagrid,&n);
  //   k = cylToLargeAR(Rgrid,Zgrid,rgrid,polgrid,n,R0,Z0);	// get the grid minor radius and poloidal angle
  //   k = interpgsl(polwant,theta,polgrid,thetagrid,n,n1dgrid);  // interpolate in poloidal angle:

  //   // then do the radial interpolation
  //   double da = 1.5/(n1dgrid-1);
  //   for(i=0;i<n1dgrid;i++){
  //       agrid[i] = i*da;
  //     thetagrid[i] = M_PI/6;
  //     }
  //   k = esigetrzb_(Rgrid,Zgrid,Bgrid,agrid,thetagrid,&n1dgrid);
  //   k = cylToLargeAR(Rgrid,Zgrid,rgrid,polgrid,n1dgrid,R0,Z0);	// get minor rad. and pol. angle
  //   k = interpgsl(rwant,a,rgrid,agrid,n,n1dgrid); // interpolate radially

  //   // // modification that I discarded
  //   //   for(i=0;i<n;i++){
  //   //     a[i] += 1.5*a[i]*exp(-400.0*theta[i]*theta[i]/(M_PI*M_PI));//a[i]*1.5*fabs(exp(-fabs(theta[i])/M_PI)-exp(-1.0/4.0));//0.5*a[i]*(1-2*fabs(theta[i])/M_PI)*(1-2*fabs(theta[i])/M_PI);
  //   //   }
  //   // --- end method: 1D interpolation of both variables independently



  // --- solution in use previously
#if DEVELOPMENT < 0
  REAL dtheta=2*M_PI/(n-1);// divide flux-coord domain
  REAL x,alarge[n],q[n],Rmap[n],Zmap[n],Bmap[n],qpol[n];//,stheta[n-1];
  // guess a large-aspect ratio, circular cross section
  cylToLargeAR(R,Z,a,theta,n,R0,Z0);
  // for(i=0;i<n;i++){		// here I use: R-R0 = a*cos(theta)
  //     x = R[i]-R0;		//    and Z = a*sin(theta)
  //     theta[i]=-atan2(Z[i],x);	// poloidal angle (correct quadrant)
  //     a[i] = sqrt(x*x+Z[i]*Z[i]);	// minor rad.
  //   }

  // set up interpolation
  // modify large aspect ratio, circ. xsection by interpolating between angles
  // for large minor radius, map theta to actual poloidal angle:
  for(i=0;i<n;i++){//generate an array of evenly-spaced flux coords 
    q[i] = -M_PI + dtheta*i;	// to interpolate what qpol (R,Z), q (a,theta) 
    alarge[i] = 1;		// is mapped to
  }
  k = esigetrzb_(Rmap,Zmap,Bmap,alarge,q,&n);
  // check to see the poloidal angle these evenly-spaced flux coords mapped to
  for(i=0;i<n;i++){		// here I use: R-R0 = a*cos(theta), Z = a*sin(theta)
    x = Rmap[i]-R0; 
    qpol[i]=-atan2(Zmap[i],x);	// poloidal angle (correct quadrant)
    //  rmax[i]=sqrt(x*x+Z[i]*Z[i]);
    //printf("%g,",qpol[i]);
  }
  // now interpolate theta
  REAL thetatmp[n];
  k = interpgsl(theta,thetatmp,qpol,q,n,n);
  for(i=0;i<n;i++) theta[i]=thetatmp[i];      //uncomment to skip interpolation

  // for(i=0;i<n;i++){
  //   a[i] += 1.5*a[i]*exp(-400.0*theta[i]*theta[i]/(M_PI*M_PI));//a[i]*1.5*fabs(exp(-fabs(theta[i])/M_PI)-exp(-1.0/4.0));//0.5*a[i]*(1-2*fabs(theta[i])/M_PI)*(1-2*fabs(theta[i])/M_PI);
  // }  //not worth it; only removed 12 total iterations
  // --- end solution in use previously
#endif

  return 0;

}

//! the 2D function solved by the root finder
int coord_f(const gsl_vector *x,void *params,gsl_vector *f){
  REAL Rwant = ((struct coordparams *) params)->Rwant; // desired R,Z
  REAL Zwant = ((struct coordparams *) params)->Zwant;

  REAL a[1] = {gsl_vector_get (x, 0)}; // current a,theta
  REAL theta[1] = {gsl_vector_get (x, 1)};
  int n=1;
  int k;
  REAL Ract[1],Zact[1],Bm[1]; // actual R,Z at this value of a,theta

  if(!finite(a[0])) a[0] = ABOUNDARY; // in portalr4 this was necessary to eliminate faults
  if(!finite(theta[0])) theta[0] = 0.0;
 
  k = esigetrzb_(Ract,Zact,Bm,a,theta,&n); // solve for the actual R,Z values
  
  gsl_vector_set (f, 0, Rwant-Ract[0]); // the error
  gsl_vector_set (f, 1, Zwant-Zact[0]);
  return GSL_SUCCESS;
}

//! returns the distance between desired (R,Z) position and the position mapped to by flux coords (a,theta)
REAL error_amt(REAL R,REAL Z,REAL a,REAL theta){
  REAL Ract,Zact,Bact;
  int single=1;
  esigetrzb_(&Ract,&Zact,&Bact,&a,&theta,&single);

  //  return sqrt((R-Ract)*(R-Ract)+(Z-Zact)*(Z-Zact));
  return hypot(R-Ract,Z-Zact);
}

//! Print current status of the root-finder
void print_state(size_t iter, gsl_multiroot_fsolver * s){
  fprintf (stderr,"iter = %3u x = % .3f % .3f "
	   "f(x) = % .3e % .3e\n",
	   iter,
	   gsl_vector_get (s->x, 0),
	   gsl_vector_get (s->x, 1),
	   gsl_vector_get (s->f, 0),
	   gsl_vector_get (s->f, 1));
}

//! performs 2D root finding to get the (a,theta) for an array of given (R,Z) values
/*! Uses the gsl mulidimensional hybrids root solver ``gsl_multiroot_fsolver_hybrids''.  */
// Modified by Lei Shi
// use function esirz2agq to find (a,theta) for a given (R,Z) inside LCFS, and treat outside points seperately

//write CDF function show the flags

inline void ERR_CI(int e,const char* s)
{
  fprintf(stderr,"error:%s in %s\n",nc_strerror(e),s);
  exit(e);
}

void writeCDF(char* fname,int ntotal, double R[],double Z[],int n_bdy,double R_bdy[],double Z_bdy[],int flag[]){
  int R_dim_id,Z_dim_id,R_bdy_dim_id,Z_bdy_dim_id,flag_dim_id;
  int nr_id,nz_id,ntotal_id,n_bdy_id;
  int R_id,Z_id,R_bdy_id,Z_bdy_id,flag_id;
  int f_id;
  int ecode;
  if(ecode=nc_create(fname,NC_CLOBBER,&f_id))
    ERR_CI(ecode,"create file");
  if(ecode=nc_def_dim(f_id,"r_dim",ntotal,&R_dim_id))
    ERR_CI(ecode,"def r dim");
  if(ecode=nc_def_dim(f_id,"z_dim",ntotal,&Z_dim_id))
    ERR_CI(ecode,"def z dim");
  if(ecode=nc_def_dim(f_id,"r_bdy_dim",n_bdy,&R_bdy_dim_id))
    ERR_CI(ecode,"def r_bdy dim");
  if(ecode=nc_def_dim(f_id,"z__bdy_dim",n_bdy,&Z_bdy_dim_id))
    ERR_CI(ecode,"def z_bdy dim");
  if(ecode=nc_def_dim(f_id,"flag_dim",ntotal,&flag_dim_id))
    ERR_CI(ecode,"def flag dim");
  if(ecode=nc_def_var(f_id,"nr",NC_INT,0,NULL,&nr_id))
    ERR_CI(ecode,"def nr var");
  if(ecode=nc_def_var(f_id,"nz",NC_INT,0,NULL,&nz_id))
    ERR_CI(ecode,"def nz var");
  if(ecode=nc_def_var(f_id,"ntotal",NC_INT,0,NULL,&ntotal_id))
    ERR_CI(ecode,"def ntotal var");
  if(ecode=nc_def_var(f_id,"n_bdy",NC_INT,0,NULL,&n_bdy_id))
    ERR_CI(ecode,"def n_bdy var");
  if(ecode=nc_def_var(f_id,"r",NC_DOUBLE,1,&R_dim_id,&R_id))
    ERR_CI(ecode,"def r var");
  if(ecode=nc_def_var(f_id,"z",NC_DOUBLE,1,&Z_dim_id,&Z_id))
    ERR_CI(ecode,"def z var");
  if(ecode=nc_def_var(f_id,"r_bdy",NC_DOUBLE,1,&R_bdy_dim_id,&R_bdy_id))
    ERR_CI(ecode,"def r_bdy var");
  if(ecode=nc_def_var(f_id,"z_bdy",NC_DOUBLE,1,&Z_bdy_dim_id,&Z_bdy_id))
    ERR_CI(ecode,"def z_bdy var");
  if(ecode=nc_def_var(f_id,"flag",NC_INT,1,&flag_dim_id,&flag_id))
    ERR_CI(ecode,"def flag var");
  if(ecode=nc_enddef(f_id))
    ERR_CI(ecode,"enddef");


  if(ecode=nc_put_var_int(f_id,nr_id,&ntotal))
    ERR_CI(ecode,"put nr var");
  if(ecode=nc_put_var_int(f_id,nz_id,&ntotal))
    ERR_CI(ecode,"put nz var");
  if(ecode=nc_put_var_int(f_id,ntotal_id,&ntotal))
    ERR_CI(ecode,"put ntotal var");
  if(ecode=nc_put_var_int(f_id,n_bdy_id,&n_bdy))
    ERR_CI(ecode,"put n_bdy var");
  if(ecode=nc_put_var_double(f_id,R_id,R))
    ERR_CI(ecode,"put R var");
  if(ecode=nc_put_var_double(f_id,Z_id,Z))
    ERR_CI(ecode,"put Z var");
  if(ecode=nc_put_var_double(f_id,R_bdy_id,R_bdy))
    ERR_CI(ecode,"put R_bdy var");
  if(ecode=nc_put_var_double(f_id,Z_bdy_id,Z_bdy))
    ERR_CI(ecode,"put Z_bdy var");
  if(ecode=nc_put_var_int(f_id,flag_id,flag))
    ERR_CI(ecode,"put flag var");

  if(ecode=nc_close(f_id))
    ERR_CI(ecode,"close file");

  FILE* txtfile;
  txtfile=fopen("./bdy_out.txt","w");
  fprintf(txtfile,"%d\n",n_bdy);
  int i;
  for(i=0;i<n_bdy;i++)
  {
    fprintf(txtfile,"%lf\t%lf\t",R_bdy[i],Z_bdy[i]);
  }
  fclose(txtfile);
  
}

int findcorrectatheta(REAL a[],REAL theta[],REAL Ract[], REAL Zact[],REAL Bm[],REAL Rwant[], REAL Zwant[],int n,int flag[]){

  // new version starts

  extern const int NBOUNDARY;
  double R_bdy[NBOUNDARY],Z_bdy[NBOUNDARY];
  getBoundaryPoints(R_bdy,Z_bdy,NBOUNDARY);
  double b_axis;
  int single_point=1;
  double tol=1e-9;  //used in inner_check,when given point is near the bdy within tol distance, it's considered on the bdy.
  double theta_guess;
  double theta_bdy;
  double a_guess;
  double a_bdy=ABOUNDARY;
  double zero=0;

  double Z_rel;//the relative Z respect to mag_axis
  double R_rel;//the relative R respect to mag_axis
	
  double mag_axis_coords[2];
  int loc;
  int sign_ini;
  int sign_cur;

  char CDFfname[20]="./InOutFlags.cdf";
  char logfile[20]="./sl_log.txt";

  FILE* errlog;
  
  errlog=fopen(logfile,"w");

  esigetrzb_(&mag_axis_coords[0],&mag_axis_coords[1],&b_axis,&zero,&zero,&single_point);


  inner_check(Rwant,Zwant,R_bdy,Z_bdy,flag,n,NBOUNDARY,tol);
 
  writeCDF(CDFfname,n,Rwant,Zwant,NBOUNDARY,R_bdy,Z_bdy,flag);
  int i;

  /*  for(i=0;i<NBOUNDARY;i++){
    fprintf(errlog,"%lf\t%lf\n",R_bdy[i],Z_bdy[i]);
  }
  fprintf(errlog,"Boundary ends.\n");
  */
  int ierror;
  double b_tmp;
  for(i=0;i<n;i++){
    if(flag[i]==1){
      //      fprintf(stderr,"point %d mapping start,",i);
      int k=ESIrz2agq(&a[i],&theta[i],&Rwant[i],&Zwant[i],&ierror,single_point);
      if(k!=0){
	fprintf(errlog,"Point %d (%lf,%lf) has encountered and error: %d in rz2agq.\n",i,Rwant[i],Zwant[i],ierror);
      }
      esigetrzb_(&Ract[i],&Zact[i],&Bm[i],&a[i],&theta[i],&single_point);
      double err=hypot((Rwant[i]-Ract[i]),(Zwant[i]-Zact[i]));
      double dis=hypot((Rwant[i]-mag_axis_coords[0]),(Zwant[i]-mag_axis_coords[1]));
      if( err > tol){
	fprintf(errlog,"Point %d (%lf,%lf) %lf away from axis has been mapped %lf away\n",i,Rwant[i],Zwant[i],dis,err);
	if(fabs(Zwant[i])<0.1)
	{
	  a[i]=0;
	  theta[i]=0;
	}
      }
    }
    else if(flag[i]==0){
      //      fprintf(stderr,"outter point.\n");
      Bm[i]=b_axis*mag_axis_coords[0]/Rwant[i]; // let B inversely proportional to R, based on B on axis
      Ract[i]=Rwant[i];
      Zact[i]=Zwant[i];
      Z_rel=Zwant[i]-mag_axis_coords[1];
      R_rel=Rwant[i]-mag_axis_coords[0];
      int quarter=0;
      if(R_rel>=0){
	if(Z_rel>=0) quarter=1;
	else quarter =4;
      }
      else{
	if(Z_rel>=0) quarter=2;
	else quarter=3;
      }

      theta_guess=atan((Z_rel/R_rel));
      if(quarter == 2 || quarter == 3)
	theta_guess += M_PI;
      if(quarter == 4)
	theta_guess += 2*M_PI;
      
      Z_rel=Z_bdy[0]-mag_axis_coords[1];
      R_rel=R_bdy[0]-mag_axis_coords[0];
      theta_bdy=atan((Z_rel/R_rel));
      
      if(R_rel>=0){
	if(Z_rel>0) quarter=1;
	else quarter =4;
      }
      else{
	if(Z_rel>0) quarter=2;
	else quarter=3;
      }
      if(quarter == 2 || quarter == 3)
	theta_bdy+=M_PI;
      if(quarter == 4)
	theta_bdy += 2*M_PI;
      

      sign_ini=(theta_guess>=theta_bdy)?1:-1;

      int j;
      for(j=0;j<NBOUNDARY;j++)
	{
	  Z_rel=Z_bdy[j]-mag_axis_coords[1];
	  R_rel=R_bdy[j]-mag_axis_coords[0];
	  theta_bdy=atan((Z_rel/R_rel));
	  
	  if(R_rel>=0){
	    if(Z_rel>0) quarter=1;
	    else quarter =4;
	  }
	  else{
	    if(Z_rel>0) quarter=2;
	    else quarter=3;
	  }
	  if(quarter == 2 || quarter == 3)
	    theta_bdy+=M_PI;
	  if(quarter == 4)
	    theta_bdy += 2*M_PI;//set the theta into (0,2Pi]

	  sign_cur=theta_guess>=theta_bdy?1:-1;
	  if(sign_ini*sign_cur<0){ //if passes the theta we want and is not on the opposite side, then we find the right boundary point
	    loc=j;
	    break;
	  }
	  
	}
      if(j==NBOUNDARY)
	loc=0;

      Z_rel=Zwant[i]-mag_axis_coords[1];
      R_rel=Rwant[i]-mag_axis_coords[0];
      //      fprintf(stderr,"start guess a\n");
      a_guess=sqrt((Z_rel*Z_rel+R_rel*R_rel)/((R_bdy[loc]-mag_axis_coords[0])*(R_bdy[loc]-mag_axis_coords[0])+(Z_bdy[loc]-mag_axis_coords[1])*(Z_bdy[loc]-mag_axis_coords[1])))*a_bdy;
      a[i]=a_guess;
      theta[i]=2*M_PI/NBOUNDARY*loc;
 
    }
    else{
      fprintf(stderr,"flag error: point %d at (%lf,%lf) has flag %d.\n",i,Rwant[i],Zwant[i],flag[i]);
      exit(5);
    }   
  }
  return 0;

  /*old version by Erik
  // declarations for root solver
  const gsl_multiroot_fsolver_type *T;
  gsl_multiroot_fsolver *s;
  int status;			// status of root-finder (notifies if stuck, etc)

  size_t iter = 0;
  int i;
  int ntries;			// num of times for this point that we get stuck
  const size_t ndims = 2;	// 2D coordinate
  struct coordparams p;		// parameters used by ea. instance of solver: Rwant,Zwant
  gsl_multiroot_function f = {&coord_f, ndims, &p}; // root solver setup 
  gsl_vector *x = gsl_vector_alloc (ndims); // R,Z values used by root solver

  T = gsl_multiroot_fsolver_hybrids; // solver type
  s = gsl_multiroot_fsolver_alloc (T, 2);


  // loop over all points
  for(i=0;i<n;i++){
    status = GSL_CONTINUE;	// reset root finder status

    // check to see if the nearest point provides a better initial estimate
    if(i>0){
      if(error_amt(Rwant[i],Zwant[i],a[i],theta[i]) > error_amt(Rwant[i],Zwant[i],a[i-1],theta[i-1])){ // if the nearest point is a better guess than the default initial guess, use the previous point
	a[i] = a[i-1];
	theta[i] = theta[i-1];
      }
    }

    // set up this instance of the root finder
    p.Rwant = Rwant[i];		// pass the desired coordinates as parameters
    p.Zwant = Zwant[i];
    gsl_vector_set (x, 0, a[i]);
    gsl_vector_set (x, 1, theta[i]);

    // try a few times with different initial conditions if the solver gets stuck
    for(ntries=0;ntries<MAX_N_TRIES;ntries++){
      if(status == GSL_SUCCESS) break; // in this case, the root-find was successful

      // choose a new initial condition if the solver is stuck
      if(ntries>0){
	#if DEVELOPMENT <= 0
	gsl_vector_set(x,0,0.2);
	gsl_vector_set(x,1,0.25*M_PI+ 0.8*gsl_vector_get(x,1));
	#else
	//	gsl_vector_set(x,0,0.2 + fmod(0.3+gsl_vector_get(x,0),1.2));
	gsl_vector_set(x,0,0.2 + fmod(0.4+gsl_vector_get(x,0),2.0));
	//	gsl_vector_set(x,1,-M_PI + fmod(1.5*M_PI+gsl_vector_get(x,1),2.0*M_PI));
	gsl_vector_set(x,1,-M_PI + fmod(1.25*M_PI+gsl_vector_get(x,1),2.0*M_PI));
	#endif
      }
      gsl_multiroot_fsolver_set (s, &f, x); // initialize the root finder
      iter = 0;			// reset iteration counter
      do{
	iter++;
	status = gsl_multiroot_fsolver_iterate (s);    
#if DEBUG > 0
	fprintf(stderr,"point:%d, ntries:%d, status:%d, ",i,ntries,status);
	print_state (iter, s);	// uncomment to give info at each iteration
#endif
	if (status){   // check if solver is stuck
	  break;
	}
	status = gsl_multiroot_test_residual (s->f, 1e-7);
      }while (status == GSL_CONTINUE && iter < 200);
            
    }

    a[i] = gsl_vector_get(s->x,0);
    theta[i] = gsl_vector_get(s->x,1);  
  }

  gsl_multiroot_fsolver_free (s);
  gsl_vector_free (x);

  // make sure all radial flux coords are positive (a>0)
  for(i=0;i<n;i++) a[i]=copysign(a[i],1.0);

  // get the actual R,Z coordinates we've mapped
  status = esigetrzb_(Ract,Zact,Bm,a,theta,&n);

#if VERBOSE > 0
  REAL err_dist;

  //modified by Lei start
  const int NBOUNDARY=4000;
  double R_bdy[NBOUNDARY],Z_bdy[NBOUNDARY];
  getBoundaryPoints(R_bdy,Z_bdy,NBOUNDARY);

  //  for(i=0;i<NBOUNDARY;i++){
  //    fprintf(stdout,"(%lf,%lf),",R_bdy[i],Z_bdy[i]);
  //  }

  for(i=0;i<n;i++){
    err_dist = hypot((Rwant[i]-Ract[i]),(Zwant[i]-Zact[i]));
  

    if(err_dist > 1e-7)//if the error is unnormally large, try to find a reasonable a and theta for that point, and adjust the B value 
      {
	//	fprintf(stderr,"Warning, point %d is mapped %.4g [m] away from requested position.\n",i,err_dist);

	int single_point=1;
	double zero=0;
	double b;         // B field on the axis
	double theta_guess;
	double theta_bdy;
	double a_guess;
	double a_bdy=1;

	double Z_rel;//the relative Z respect to mag_axis
	double R_rel;//the relative R respect to mag_axis
	
	double mag_axis_coords[2];
	int loc;
	int sign_ini;
	int sign_cur;

	esigetrzb_(&mag_axis_coords[0],&mag_axis_coords[1],&b,&zero,&zero,&single_point);

	Bm[i]=b*mag_axis_coords[0]/Rwant[i]; // let B inversely proportional to R, based on B on axis

	Z_rel=Zwant[i]-mag_axis_coords[1];
	R_rel=Rwant[i]-mag_axis_coords[0];

	theta_guess=atan((Z_rel/R_rel));
	if(theta_guess<=0)
	  theta_guess += 2*M_PI;
	theta_bdy=atan((Z_bdy[0]-mag_axis_coords[1])/(R_bdy[0]-mag_axis_coords[0]));
	if(theta_bdy<=0)
	  theta_bdy+=2*M_PI;
	sign_ini=(theta_guess>=theta_bdy)?1:-1;
	//	fprintf(stderr,"pt[%d]:\n sign_ini = %d\n",i,sign_ini);
	int j;
	for(j=0;j<NBOUNDARY;j++)
	  {
	    theta_bdy=atan((Z_bdy[j]-mag_axis_coords[1])/(R_bdy[j]-mag_axis_coords[0]));
	    if(theta_bdy<=0)
	      theta_bdy += 2*M_PI;//set the theta into [0,2Pi)
	    sign_cur=theta_guess>=theta_bdy?1:-1;
	    if(sign_ini*sign_cur<0 && Zwant[i]*Z_bdy[j]>=0){ //if passes the theta we want and is not on the opposite side, then we find the right boundary point
	      
	      loc=j;
	      break;
	    }
	      
	  }

	a_guess=sqrt((Z_rel*Z_rel+R_rel*R_rel)/((R_bdy[loc]-mag_axis_coords[0])*(R_bdy[loc]-mag_axis_coords[0])+(Z_bdy[loc]-mag_axis_coords[1])*(Z_bdy[loc]-mag_axis_coords[1])))*a_bdy;
	a[i]=a_guess;
	theta[i]=2*M_PI/NBOUNDARY*loc;
	//	if(theta_bdy>M_PI)
	//  fprintf(stderr,"sign_cur=%d\n loc= %d,Rwant=%lf, Zwant=%lf, theta=%lf, Rbd=%lf, Zbd=%lf, thetabd=%lf, a=%lf",sign_cur,loc,Rwant[i],Zwant[i],theta_guess,R_bdy[loc],Z_bdy[loc],theta_bdy,a[i]);
      }
  }

  //Modified by Lei End
#endif
  

old version ends*/
}

//! Calculates the (R,Z) coordinates of the boundary of the last-closed-flux-surface
/*! Assumes the LCFS is given by the macro ABOUNDARY (set to 1.0 by default) */
int getBoundaryPoints(double R_bdy[],double Z_bdy[],int n_bdy){
  double dtheta = 2*M_PI/(n_bdy-1);
  double a[n_bdy],theta[n_bdy],B_bdy[n_bdy];
  int i,k;
  for(i=0;i<n_bdy;i++){
    theta[i] = dtheta*i;
    a[i] = ABOUNDARY;
  }

  k=esigetrzb_(R_bdy,Z_bdy,B_bdy,a,theta,&n_bdy);

}



//! Find the (R,Z) coordinates of the magnetic axis
/*! Uses the esi functions.
 */
int get_mag_axis(REAL coords[]){
  REAL bm;
  REAL mag_axis = 0;
  int single_point = 1;

  // get the position of the magnetic axis
  int k = esigetrzb_(&coords[0],&coords[1],&bm,&mag_axis,&mag_axis,&single_point);

  return 0;
}


// modified by Lei Shi to get desired shape outside the LCFS
// The shape function is set to be 

// shape1:exponential tan shape
// n(a)=n(1)*exp(-n'(1)/n(1)*tan(PI/2*(a-1)/(a0-1)));

// shape2:cubic shape
// n(a)=c3*a^3+c2*a^2+c1^a+c0

// so that n(1) and n'(1) are unchanged, and n(a0) and n'(a0) are 0; 
inline double Shape1(double x, double xb,double x0,double yb,double ypb){
  return yb*exp(ypb/yb*tan(M_PI/2*(x-xb)/(x0-xb)));      
}

double Shape2(double x,double xb,double x0,double yb,double ypb){
  double xb2=xb*xb;
  double xb3=xb*xb2;
  double x02=x0*x0;
  double x03=x0*x02;

  double a_data[]=
    {
      1,xb,xb2, xb3,
      1,x0,x02, x03,
      0,1, 2*xb,3*xb2,
      0,1, 2*x0,3*x02
    };

  double b_data[]={yb,0,ypb,0};
  gsl_matrix_view m= gsl_matrix_view_array(a_data,4,4);
  gsl_vector_view b= gsl_vector_view_array(b_data,4);
  gsl_vector *c = gsl_vector_alloc(4);
  int s;
  gsl_permutation *p=gsl_permutation_alloc(4);
  gsl_linalg_LU_decomp(&m.matrix,p,&s);
  gsl_linalg_LU_solve(&m.matrix,p,&b.vector,c);
  
  double C[4];
  int i;
  for(i=0;i<4;i++)
    C[i]=gsl_vector_get(c,i);

  return C[0]+C[1]*x+C[2]*x*x+C[3]*x*x*x;
}

inline double Decay(int shape,double a,double a_bdy,double a0,double y_bdy,double yp_bdy){
  switch(shape){
  case 1:
    return Shape1(a,a_bdy,a0,y_bdy,yp_bdy);
  case 2:
    return Shape2(a,a_bdy,a0,y_bdy,yp_bdy);
  default:
    fprintf(stderr,"Wrong decay shape number, check the notes in profile_coord_map.h before function decayNToutsideLCFS for more info.\n");
  }
}

//! Cause density and temperature to exponentially decay outside the LCFS
/*! Use the value of density, Ti, and Te at the Last-Closed-Flux-Surface
  and force these quantities to exponentially decay outside this surface.
  This is done so there is no abrupt increase in these quantities that might
  frustrate the paraxial solver in the reflectometry code.
*/



int decayNToutsideLCFS(int npts,REAL a[],REAL ne[],REAL Te[],REAL Ti[],int* flag){
  REAL abound = ABOUNDARY; // (a,theta) on the boundary that we use
  REAL ain = AINSIDE;	// (a,theta) inside the LCFS we use
  // (* to find the exponential drop off *) in our case, to find n'(1) 

  REAL a0 = AZERO; // the flux coord where we set n & T to be zero
  
  //load the equilibrium profile
  int** n_prof=(int**)xmalloc(sizeof(int*));
  REAL **a_p=(REAL**)xmalloc(sizeof(REAL*));
  REAL **n_p=(REAL**)xmalloc(sizeof(REAL*));
  REAL **Ti_p=(REAL**)xmalloc(sizeof(REAL*));
  REAL **Te_p=(REAL**)xmalloc(sizeof(REAL*));
  int k;
  k=loadNTProfiles(NTFILENAME,n_prof,a_p,n_p,Ti_p,Te_p);
  if(k!=0){
    fprintf(stderr,"Loading NTProfile error %d.\n",k);
    exit(1);
  }

  // values of ne,te,ti at boundary and just inside LCFS
  REAL nebound,tibound,tebound,nein,tin,tein;
  getProfile(abound,&nebound,&tibound,&tebound,*a_p,*n_p,*Ti_p,*Te_p,**n_prof);
  getProfile(ain,&nein,&tin,&tein,*a_p,*n_p,*Ti_p,*Te_p,**n_prof); 

  nebound *=INVCM3_TO_INVM3;
//  tibound *=1;
//  tebound *=1;
  nein *=INVCM3_TO_INVM3;
//  tin *=1;
//  tein *=1;

  // the 1/e distance in radial flux coord
  // the scale factor is to make it drop off faster
  //  REAL lambda_ne = -fabs(DECAY_SCALING*nebound/(nein-nebound));
  //  REAL lambda_ti = -fabs(DECAY_SCALING*tibound/(tin-tibound));
  //  REAL lambda_te = -fabs(DECAY_SCALING*tebound/(tein-tebound));

  // the direvitives are obtained based on linear estimation

  REAL ne_prime = (nebound-nein)/(abound-ain);
  REAL ti_prime = (tibound-tin)/(abound-ain);
  REAL te_prime = (tebound-tein)/(abound-ain);

  fprintf(stderr,"n'=%lf,t'=%lf",ne_prime,te_prime);

  // update the values in the ne,te,ti arrays
//  int i;
//  for(i=0;i<npts;i++){
//    if(a[i]>1.0){
//    ne[i] = nebound*exp((a[i]-abound)/lambda_ne);
//    Te[i] = tebound*exp((a[i]-abound)/lambda_te);
//    Ti[i] = tibound*exp((a[i]-abound)/lambda_ti);
//    }
  int i;
  for(i=0;i<npts;i++){
    if(flag[i] == 0 ){
      if(a[i] <= a0){
	ne[i]=Decay(2,a[i],abound,a0,nebound,ne_prime);      
	Te[i]=Decay(2,a[i],abound,a0,tebound,te_prime);      
	Ti[i]=Decay(2,a[i],abound,a0,tibound,ti_prime);      
	//	fprintf(stderr,"inside a[%d]=%lf, ne=%lf, Te=%lf , Ti=%lf\n",i,a[i],ne[i],Te[i],Ti[i]);
      }
      else{
	ne[i]=Te[i]=Ti[i]=0;  // if the point is outside a0, set all to be 0
	//	fprintf(stderr,"outside a[%d]=%lf, ne=%lf, Te=%lf , Ti=%lf\n",i,a[i],ne[i],Te[i],Ti[i]);
      }
    }
  }
  return 0;
}

//! Runs all the functions to map from cylindrical (R,Z) coords to flux coords
/*! Given the number of points npts, the desired cylindrical coords (Rwant,Zwant)
    and the (R,Z) coordinate array of the magnetic axis position "mag_axis_coords",
    gives the initial guess (Rinitial,Zinitial), the flux coords (a,theta), and
    the actual (Ract,Zact) these flux coords map to.
 */
int getFluxCoords(int npts,REAL a[],REAL theta[],REAL Bm[],REAL Ract[],REAL Zact[],REAL Rinitial[],REAL Zinitial[],REAL Rwant[],REAL Zwant[],REAL mag_axis_coords[],int flag[]){
  // field-line coords: a: flux (radial), theta: angle (poloidal), Bm: |B|
  // (Rwant,Zwant): the R,Z coordinatew we want to map to
  // (Rinitial,Zinitial): the R,Z coords of our initial guess
  // (Ract,Zact): actual R,Z coordinates we end up with
  int k;
  //  k = guessatheta(a,theta,Rwant,Zwant,npts,mag_axis_coords[0],mag_axis_coords[1]); // initial guess for the flux coords
  //  k = esigetrzb_(Rinitial,Zinitial,Bm,a,theta,&npts); // map to cylindrical coords
  k = findcorrectatheta(a,theta,Ract,Zact,Bm,Rwant,Zwant,npts,flag); // root-find to get flux coords

  return 0;
}


//! Gets all the profile data at the specified flux coordinates
/*! Given the number of points npts, the flux coords (a,theta), gets the
    values of: Bpol, Ti, Te, P, ne, and the q profile at these points.
 */
extern const char* NTFILENAME;

int getProfile(REAL a,REAL* ne,REAL* Ti,REAL* Te,REAL* a_p,REAL* n_p,REAL* Ti_p,REAL* Te_p,int n_prof){ //get ne,Ti,Te for a specified flux surface psi=a, linear interpolation based on the loaded ne,Ti,Te profile

  interpxy(&a,ne,a_p,n_p,1,n_prof);
  interpxy(&a,Ti,a_p,Ti_p,1,n_prof);
  interpxy(&a,Te,a_p,Te_p,1,n_prof);
  return 0;
}


int getAllProfiles(int npts,REAL Bpol[],REAL Ti[],REAL Te[],REAL P[],REAL ne[],REAL qprofile[],REAL a[],REAL theta[],int InOutFlag[]){
  // used by esilink2c and esiget2dfunctions
  REAL *F = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *Fa = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *gFa = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *gFaa = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *gYa = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *gYaa = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *Ta = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *Pa = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *r = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *ra = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *rq = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *z = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *za = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *zq = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *B = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *Ba = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *Bq = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *gh = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *gha = (REAL *)xmalloc(npts*sizeof(REAL));
  REAL *ghq = (REAL *)xmalloc(npts*sizeof(REAL));

  int k = esilink2c_(F,Fa,gFa,gFaa,gYa,gYaa,Ti,Ta,P,Pa,r,ra,rq,z,za,zq,B,Ba,Bq,gh,gha,ghq);
  k = esiget2dfunctions_(a,theta,&npts);

  // get q profile
  int i;
  for(i=0;i<npts;i++){// reverse of method documented in escZ_march03.c
    if(a[i]==0.0){		// this method is used in setup_v2.F90
      qprofile[i] = -gFaa[i]/gYaa[i]; // the magnetic axis is a special case
    }else qprofile[i]=-gFa[i]/gYa[i];
  }
  
  int** n_prof=(int**)xmalloc(sizeof(int*));
  REAL **a_p=(REAL**)xmalloc(sizeof(REAL*));
  REAL **n_p=(REAL**)xmalloc(sizeof(REAL*));
  REAL **Ti_p=(REAL**)xmalloc(sizeof(REAL*));
  REAL **Te_p=(REAL**)xmalloc(sizeof(REAL*));

  k=loadNTProfiles(NTFILENAME,n_prof,a_p,n_p,Ti_p,Te_p);
  if(k!=0){
    fprintf(stderr,"Loading NTProfile error %d.\n",k);
    exit(1);
  }

  // get density and temperature profiles 
  for(i=0;i<npts;i++){

    if(InOutFlag[i]==1){
      getProfile(a[i],&ne[i],&Ti[i],&Te[i],*a_p,*n_p,*Ti_p,*Te_p,**n_prof);
      Ti[i] *= 100;
      Te[i] *= 100;
      ne[i] *= INVCM3_TO_INVM3;
    }
    else{
      ne[i]=0;
      Ti[i]=0;
      Te[i]=0;
    }

    /* Old version of getting profiles
// need to call Fortran functions
    this_a = MINIMUM(a[i],ABOUNDARY); // in order to keep from erroring out on rh4
    //this_a = a[i]; // in order to keep from erroring out on rh4
    Ti[i] = temperature_(&this_a,&theta[i])*NORM_TO_KEV;
    ne[i] = density_(&this_a,&theta[i])*INVCM3_TO_INVM3;
    Te[i] = temperature_e_(&this_a,&theta[i])*NORM_TO_KEV;	// use this instead for Te, since esiget2dfunctions doesn't give reasonable values 
    */
  }

  // get poloidal magnetic field
  REAL grad_a;
  for(i=0;i<npts;i++){
    if(InOutFlag[i]==1){
      grad_a = hypot(rq[i],zq[i])/(za[i]*rq[i]-zq[i]*ra[i]);
      Bpol[i] = -grad_a/r[i]*gYa[i];
    }
    else{
      Bpol[i] = 0;
    }
  }

  free(*a_p);
  free(*n_p);
  free(*Te_p);
  free(*Ti_p);
  free(n_prof);
  free(a_p);
  free(n_p);
  free(Te_p);
  free(Ti_p);
  // free arrays that are no longer needed:
  free(F);
  free(Fa);
  free(gFa);
  free(gFaa);
  free(gYa);
  free(gYaa);
  free(Ta);
  free(Pa);
  free(r);
  free(ra);
  free(rq);
  free(z);
  free(za);
  free(zq);
  free(B);
  free(Ba);
  free(Bq);
  free(gh);
  free(gha);
  free(ghq);

  return 0;
}

//! Determine the density fluctuation assuming adiabatic electron response to potential fluctuations.
/*! Given the number of points npts, the background density (ne0), electron temperature (Te),
    and potential fluctuation (phi), computes the density fluctuation (ne_tilde) and the total
    density (ne), using: ne_tilde = ne0*phi/Te, and ne = ne0 + ne_tilde
 */
int adiabaticElectronResponse(int npts,int ntimesteps,REAL ne_tilde[],REAL ne0[],REAL phi[],REAL Te[],int flag[]){
    int i,j;
    for(i=0;i<ntimesteps;i++){
      for(j=0;j<npts;j++){
	if(flag[j]==1){
	  double* phi_this = &phi[i*npts+j];
	  *phi_this *= NORM_TO_KEV;
	  ne_tilde[i*npts+j] = ne0[j]* (*phi_this)/Te[j];
	}
	else
	  ne_tilde[i*npts+j] = 0;
      }
    }
    return 0;
  }
