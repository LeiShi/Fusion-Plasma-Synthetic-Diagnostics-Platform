/*! \file interp.h
    \brief Functions for interpolation to initialize the root solver
 */

// used by the gsl interpolation functions
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sort_double.h>

#ifndef REAL_TYPEDEF
#define REAL_TYPEDEF
#ifndef single			// compiler option determines variable size
typedef double REAL;
#else
typedef float REAL;
#endif
#endif

#define N_INTERP_POINTS 3
#define NEAREST_TOL 1.0e-3

#define MIN_IDX(indices,values) ((values[0] < values[1]) ? indices[0]:indices[1])
#define DISTANCE(x1,y1,x2,y2) hypot(x1-x2,y1-y2)
#define MAXIMUM(x,y) ((x) > (y) ? (x) : (y))



// interpolation functions
REAL interpBetw2ptsDist(REAL distx_x0,REAL f0,REAL f1,REAL distx1_x0);
int cartesianToCylindrical(int npts,REAL R[],REAL Z[],REAL zeta[],REAL x[],REAL y[],REAL z[]);
int interp2d(int nx,int nxi,REAL x[],REAL y[],REAL z[],REAL w[],REAL xi[],REAL yi[],REAL zi[],REAL wi[]);
int find_nearestn(int nclosest,int nx,int nxi,size_t idx_closest[nx][nclosest],REAL x[],REAL y[],REAL xi[],REAL yi[]);
int calc_slopes(double mx[],double my[],size_t idxs[],double x[],double y[],double z[]);

// legacy functions used by previous versions of the root finder
int interp(REAL y[],REAL xi[],REAL si[],int ny,int nxi);
int interpxy(REAL x[],REAL y[],REAL xi[],REAL yi[],int nx,int nxi);
int interpgsl(REAL x[],REAL y[],REAL xi[],REAL yi[],int nx,int nxi);
int find_nearest(int nx,int xirows,int xicols,int xi_idx_below[],int yi_idx_below[],REAL x[],REAL y[],REAL xi[][xicols],REAL yi[][xicols]);
int interpReg2d(int nx,int xirows,int xicols,REAL x[],REAL y[],REAL z[],REAL w[],REAL xi[][xicols],REAL yi[][xicols],REAL zi[][xicols],REAL wi[][xicols]);


// ---  local function definitions

//! Linear interpolation using two points to a third point
/*! Uses the distance between the two known points ``distx1_x0'' = (x1-x0),
    the distance between the unknown point to a known point ``distx_x0'' = (x-x0),
    and the values of the function at x1 and x0 (f1 = f(x1),f2 = f(x2)),
    to find the value at x: f= f(x).
 */
REAL interpBetw2ptsDist(REAL distx_x0,REAL f0,REAL f1,REAL distx1_x0){
  REAL f = (f1-f0)/distx1_x0*distx_x0 + f0;
  return f;
}

//! Converts cartesian coordinates (x,y,z) to cylindrical coordinates (R,Z,zeta)
int cartesianToCylindrical(int npts,REAL R[],REAL Z[],REAL zeta[],REAL x[],REAL y[],REAL z[]){
  int i;
  for(i=0;i<npts;i++){
    R[i]=hypot(x[i],z[i]);
    Z[i]=y[i];
    zeta[i]=-atan2(z[i],x[i]);	// negative because in forming right-hand coord system
  }                               //  x: radially out, y: vertically up, z: toroidal
                                  //  esi routines give theta going clockwise,
                                  //  a: out in minor radius, gives z and theta antiparallel

  return 0;
}



//! Finds the nearest n points on a grid of (xi,yi) for each point (x,y)
int find_nearestn(int nclosest,int nx,int nxi,size_t idx_closest[nx][nclosest],REAL x[],REAL y[],REAL xi[],REAL yi[]){
  double dist[nxi],d01,d02,d12,hyp;
  int point,i,n_temp;
  size_t *closest_pts;
  size_t *idx_closest_pt;

#if DEBUG
  fprintf(stderr,"point,   d01,    d02,   d12,    hyp,    collinear\n");
#endif

  for(point=0;point<nx;point++){		// loop for each x,y

    for(i=0;i<nxi;i++){	// find the distance between (x,y) and all the (xi,yi)
      //      dist[i]=sqrt((x[point]-xi[i])*(x[point]-xi[i])+(y[point]-yi[i])*(y[point]-yi[i]));
      dist[i]=DISTANCE(x[point],y[point],xi[i],yi[i]);
    }
    idx_closest_pt=idx_closest[point];
    gsl_sort_smallest_index(idx_closest_pt,nclosest,dist,1,nxi);// find the nearest xi

    // make sure they aren't collinear
    d01=DISTANCE(xi[idx_closest_pt[0]],yi[idx_closest_pt[0]],xi[idx_closest_pt[1]],yi[idx_closest_pt[1]]);
    d02=DISTANCE(xi[idx_closest_pt[0]],yi[idx_closest_pt[0]],xi[idx_closest_pt[2]],yi[idx_closest_pt[2]]);
    d12=DISTANCE(xi[idx_closest_pt[1]],yi[idx_closest_pt[1]],xi[idx_closest_pt[2]],yi[idx_closest_pt[2]]);
    //    hyp = fmax(fmax(d01,d02),d12);
    hyp = MAXIMUM(MAXIMUM(d01,d02),d12);
    n_temp = nclosest;
    while((d01+d02+d12-2.0*hyp < NEAREST_TOL) && (nclosest < nxi)){ // loop until the points aren't collinear
      n_temp++;
      closest_pts = (size_t *)malloc(sizeof(size_t)*n_temp);
      gsl_sort_smallest_index(closest_pts,n_temp,dist,1,nxi);// find the nearest xi
      idx_closest_pt[2]=closest_pts[n_temp-1];
      d02=DISTANCE(xi[idx_closest_pt[0]],yi[idx_closest_pt[0]],xi[idx_closest_pt[2]],yi[idx_closest_pt[2]]);
      d12=DISTANCE(xi[idx_closest_pt[1]],yi[idx_closest_pt[1]],xi[idx_closest_pt[2]],yi[idx_closest_pt[2]]);
      //hyp = fmax(fmax(d01,d02),d12);
      hyp = MAXIMUM(MAXIMUM(d01,d02),d12);

      free(closest_pts);
    }
#if DEBUG
    fprintf(stderr,"%d, %g, %g, %g, %g, %g\n",point,d01,d02,d12,hyp,2.0*hyp-d01-d02-d12);
#endif
	 
  }
    //   for(i=0;i<xirows*xicols;i++){
    //     fprintf(stderr,"xi       yi\n");
    //   fprintf(stderr,"%g, %g\n",xi[0][i],yi[0][i]);
    //   }
    //   for(i=0;i<nx;i++){
    //     fprintf(stderr,"x,      xi_below,      y,    ,yi_below\n");
    //     fprintf(stderr,"%g, %g, %g, %g\n",x[i],xi[xi_idx_below[i]][0],y[i],yi[0][yi_idx_below[i]]);
    //   }


    return 0;
 }


  //! Calculates the slopes (mx,my) for three points (x,y) which each have function value z
  /*! these three points are have the indices idxs */
  int calc_slopes(double mx[],double my[],size_t idxs[],double x[],double y[],double z[]){
    size_t i,ip1,ip2,j,jp1,jp2;
    double dx[N_INTERP_POINTS],dy[N_INTERP_POINTS],dz[N_INTERP_POINTS];

    for(j=0;j<N_INTERP_POINTS;j++){
      ip1 = idxs[(size_t)fmod(j+1,N_INTERP_POINTS)];
      ip2 = idxs[(size_t)fmod(j+2,N_INTERP_POINTS)];
      dx[j] = x[ip1]-x[ip2];
      dy[j] = y[ip1]-y[ip2];
      dz[j] = z[ip1]-z[ip2];
    }
    for(j=0;j<N_INTERP_POINTS;j++){
      jp1 = (size_t)fmod(j+1,N_INTERP_POINTS);
      jp2 = (size_t)fmod(j+2,N_INTERP_POINTS);
      i = idxs[j];
      ip1 = idxs[jp1];
      ip2 = idxs[jp2];
      mx[j] = -1.0*(y[i]*dz[j]+y[ip1]*dz[jp1]+y[ip2]*dz[jp2]);
      mx[j]/=(x[i]*dy[j]+x[ip1]*dy[jp1]+x[ip2]*dy[jp2]);
      my[j] = -1.0*(x[i]*dz[j]+x[ip1]*dz[jp1]+x[ip2]*dz[jp2]);
      my[j]/=(y[i]*dx[j]+y[ip1]*dx[jp1]+y[ip2]*dx[jp2]);
    }



    return 0;
  }

//! High-level 2d interpolation function
/*! Given values zi = f(xi,yi) and wi = g(xi,yi) on an arbitrary, irregular grid (xi,yi)
    finds the values at (x,y).
    The method is to find the nearest 3 grid points for each (x,y), then linearly interpolate (triangulation).
 */
  int interp2d(int nx,int nxi,REAL x[],REAL y[],REAL z[],REAL w[],REAL xi[],REAL yi[],REAL zi[],REAL wi[]){
    int point;
    size_t *idxs;
    double mxi[N_INTERP_POINTS],myi[N_INTERP_POINTS],a,b,c;

    // find the indices of the 3 (xi,yi) closest to x,y
    size_t idx_closest[nx][N_INTERP_POINTS];
    find_nearestn(N_INTERP_POINTS,nx,nxi,idx_closest,x,y,xi,yi);

    // loop around points
    for(point=0;point<nx;point++){
      idxs = idx_closest[point];
      // first get z
      // calculate the slopes (mx,my) for the nearest 3 points
      calc_slopes(mxi,myi,idxs,xi,yi,zi);
      // linear interp at each nearest point
      a = zi[idxs[0]] + mxi[0]*(x[point]-xi[idxs[0]]) + myi[0]*(y[point]-yi[idxs[0]]);
      b = zi[idxs[1]] + mxi[1]*(x[point]-xi[idxs[1]]) + myi[1]*(y[point]-yi[idxs[1]]);
      c = zi[idxs[2]] + mxi[2]*(x[point]-xi[idxs[2]]) + myi[2]*(y[point]-yi[idxs[2]]);
      // average to get the best guess
      z[point] = (a+b+c)/3.0;

      // next get w
      // calculate the slopes (mx,my) for the nearest 3 points
      calc_slopes(mxi,myi,idxs,xi,yi,wi);
      // linear interp at each nearest point
      a = wi[idxs[0]] + mxi[0]*(x[point]-xi[idxs[0]]) + myi[0]*(y[point]-yi[idxs[0]]);
      b = wi[idxs[1]] + mxi[1]*(x[point]-xi[idxs[1]]) + myi[1]*(y[point]-yi[idxs[1]]);
      c = wi[idxs[2]] + mxi[2]*(x[point]-xi[idxs[2]]) + myi[2]*(y[point]-yi[idxs[2]]);
      // average to get the best guess
      w[point] = (a+b+c)/3.0;

    }

    return 0;
  }

// --- legacy functions

int interp(REAL y[],REAL xi[],REAL si[],int ny,int nxi){
  int i,j;
  //  printf("interp:\n%10s,%10s,%10s,%10s,%10s\n","q_orig","q_final","q_i","q_i+1","s_i");
  for(i=0;i<ny;i++){
    for(j=0;j<nxi-1;j++) if(xi[j+1] > y[i]) break;
    //printf("%10g,",y[i]);
    y[i] = xi[j] + si[j]*(y[i]-xi[j]);
    //printf("%10g,%10g,%10g,%10g\n",y[i],xi[j],xi[j+1],si[j]);
  }
  return 0;
}

int interpxy(REAL x[],REAL y[],REAL xi[],REAL yi[],int nx,int nxi){
  int i,j;
  REAL si[nxi-1];
  // use linear interpolation 
  //printf("\nstheta:");
  for(i=0;i<nxi-1;i++){
    si[i] = (yi[i+1]-yi[i])/(xi[i+1]-xi[i]);
    //printf("%g,",si[i]);
  }
  si[nxi-2] = si[0];
  //printf("\n");

  //printf("interp:\n%10s,%10s,%10s,%10s,%10s\n","q_orig","q_final","q_i","q_i+1","s_i");
  for(i=0;i<nx;i++){
    for(j=0;j<nxi-1;j++) if(xi[j+1] > x[i]) break;
    //printf("%10g,",x[i]);
    y[i] = yi[j] + si[j]*(x[i]-xi[j]);
    //printf("%10g,%10g,%10g,%10g\n",y[i],xi[j],xi[j+1],si[j]);
  }

  return 0;
}

// interpolate the array x[] of size nx
// on the grid xi[],y[i] of size nxi
int interpgsl(REAL x[],REAL y[],REAL xi[],REAL yi[],int nx,int nxi){
  // make sure periodic
  //  yi[nxi-1]=yi[0];
  // xi[nxi-1]=-xi[0];
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  const gsl_interp_type *t = gsl_interp_cspline; 
  gsl_spline *spline = gsl_spline_alloc (t, nxi);
     
  int i;
  gsl_spline_init (spline, xi, yi, nxi);
  //printf("\ninterpgsl:\n");
  for (i = 0; i < nx; i++)
    {
      y[i] = gsl_spline_eval (spline, x[i], acc);
      //printf ("%g, %g\n", x[i], y[i]);
    }
       
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);

  return 0;
}

int find_nearest1d(int nx,int nxi,int nnearest,int stride,size_t xi_idx[nx][nnearest],REAL x[],REAL xi[]){
  int i,j;
  REAL xerr[nxi];
  for(i=0;i<nx;i++){
    for(j=0;j<nxi;j++){
      xerr[j]=fabs(x[i]-xi[j*stride]);
    }
    gsl_sort_smallest_index(xi_idx[i],nnearest,xerr,stride,nxi);// find the n nearest xi
  }

  return 0;
}


// xi varies slowly (each row), yi varies quickly (each col)
int find_nearest(int nx,int xirows,int xicols,int xi_idx_below[],int yi_idx_below[],REAL x[],REAL y[],REAL xi[][xicols],REAL yi[][xicols]){
  size_t nearest2_idxs[2];
  double nearest2_vals[2],xerr[xirows],yerr[xirows];
  int i,j;

  for(i=0;i<nx;i++){		// loop for each x,y

    for(j=0;j<xirows;j++){	// find the error between x and all the xi
      xerr[j]=fabs(x[i]-xi[j][0]);
    }
    gsl_sort_smallest_index(nearest2_idxs,2,xerr,1,xirows);// find the two nearest xi
    nearest2_vals[0] = xi[nearest2_idxs[0]][0];
    nearest2_vals[1] = xi[nearest2_idxs[1]][0];
    xi_idx_below[i] = MIN_IDX(nearest2_idxs,nearest2_vals); // this is the nearest xi less than x

    for(j=0;j<xicols;j++){	// find the error between y and all the yi
      yerr[j]=fabs(y[i]-yi[0][j]);
    }
    gsl_sort_smallest_index(nearest2_idxs,2,yerr,1,xicols);// find the two nearest yi
    nearest2_vals[0] = yi[0][nearest2_idxs[0]];
    nearest2_vals[1] = yi[0][nearest2_idxs[1]];
    yi_idx_below[i] = MIN_IDX(nearest2_idxs,nearest2_vals); // this is the nearest yi less than y
  }

  return 0;
}

// limitation: assumes regularly-spaced grids xi and yi
int interpReg2d(int nx,int xirows,int xicols,REAL x[],REAL y[],REAL z[],REAL w[],REAL xi[][xicols],REAL yi[][xicols],REAL zi[][xicols],REAL wi[][xicols]){
  int i,j;

  // form the matrix of of slopes for y interpolation
  double slopeziyi[xirows][xicols-1],slopewiyi[xirows][xicols-1];
  double invdi;
  for(i=0;i<xirows;i++){
    for(j=0;j<xicols-1;j++){
      invdi = 1.0/(yi[i][j+1]-yi[i][j]);
      slopeziyi[i][j] = (zi[i][j+1]-zi[i][j])*invdi;
      slopewiyi[i][j] = (wi[i][j+1]-wi[i][j])*invdi;
    }
  }

  // find the indices of the nearest points (xi[i],yi[j]); note always exists (xi[i+1],yi[j+1])
  int xi_idx_below[nx],yi_idx_below[nx];
  find_nearest(nx,xirows,xicols,xi_idx_below,yi_idx_below,x,y,xi,yi);

  // for fixed x, interpolate in y
  double a,b,dist_y_yi,x_frac;
  for(i=0;i<nx;i++){
    dist_y_yi=y[i]-yi[0][yi_idx_below[i]];
    x_frac = (x[i]-xi[xi_idx_below[i]][0])/(xi[xi_idx_below[i]+1][0]-xi[xi_idx_below[i]][0]);

    // compute z[i]
    // for fixed xi = i, interpolate in y between yi[j] and yi[j+1]
    a = slopeziyi[xi_idx_below[i]][yi_idx_below[i]]*dist_y_yi + zi[xi_idx_below[i]][yi_idx_below[i]];
    // for fixed xi = i+1, interpolate in y between yi[j] and yi[j+1]
    b = slopeziyi[xi_idx_below[i]+1][yi_idx_below[i]]*dist_y_yi + zi[xi_idx_below[i]+1][yi_idx_below[i]];  
    // now interpolate in x between xi[i] and xi[i+1]
    z[i] = (b-a)*x_frac + a;

    // compute w[i]
    // for fixed xi = i, interpolate in y between yi[j] and yi[j+1]
    a = slopewiyi[xi_idx_below[i]][yi_idx_below[i]]*dist_y_yi + wi[xi_idx_below[i]][yi_idx_below[i]];
    // for fixed xi = i+1, interpolate in y between yi[j] and yi[j+1]
    b = slopewiyi[xi_idx_below[i]+1][yi_idx_below[i]]*dist_y_yi + wi[xi_idx_below[i]+1][yi_idx_below[i]];  
    // now interpolate in x between xi[i] and xi[i+1]
    w[i] = (b-a)*x_frac + a;

  }

  return 0;
}



