//The module functions to be called in python loaded modules

#include<Python.h>
#include<numpy/arrayobject.h>
#include<stdio.h>

#include "interp.h"
#include "fileio.h"
#include "profile_coord_map.h"
#include "fluc.h"

extern double findMin(int n,REAL data[]);


double Xmin=2.0,Xmax=2.6,Ymin=-0.6,Ymax=0.6,Zmin=0,Zmax=0;
int NX=101,NY=201,NZ=1, NBOUNDARY = 1001;
int TStart=100, TStep=10, NT=10;
double Fluc_Amplification = 50;
char* FlucFilePath="./Fluctuations/";
char* EqFileName="./ESI_EQFILE";
char* NTFileName="./NTProfiles.cdf";
char* PHI_FNAME_START="PHI.";
char* DEN_FNAME_START="DEN.";
char* GTS_DATA_DIR="./GTS_OUTPUT_FILES/";


static PyObject*
set_parameters_(PyObject* self, PyObject* args, PyObject* kws){
  int sts=0;
  static char* kwlist[]={"Xmin","Xmax","NX","Ymin","Ymax","NY","Zmin","Zmax","NZ",
			 "TStart","TStep","NT","NBOUNDARY","Fluc_Amplification",
			 "FlucFilePath","EqFileName","NTFileName","PHIFileNameStart","DENFileNameStart","GTSDataDir",NULL};
  if(!PyArg_ParseTupleAndKeywords(args,kws,"|ddiddiddiiiiidssssss",kwlist,
				  &Xmin,&Xmax,&NX,&Ymin,&Ymax,&NY,&Zmin,&Zmax,&NZ,
				  &TStart,&TStep,&NT,&NBOUNDARY,&Fluc_Amplification,&FlucFilePath,&EqFileName,&NTFileName,&PHI_FNAME_START,&DEN_FNAME_START,&GTS_DATA_DIR))
    return NULL;
  return Py_BuildValue("i",sts);
}

static PyObject*
show_parameters_(PyObject* self, PyObject* args){
  int sts = 0;
  printf("Parameters set as following:\n");
  printf("X: (Xmin=%lf,Xmax=%lf,NX=%d)\n",Xmin,Xmax,NX);
  printf("Y: (Ymin=%lf,Ymax=%lf,NY=%d)\n",Ymin,Ymax,NY);
  printf("Z: (Zmin=%lf,Zmax=%lf,NZ=%d)\n",Zmin,Zmax,NZ);
  printf("NBOUNDARY: %d\n",NBOUNDARY);
  printf("T: (T0=%d,dT=%d,NT=%d)\n",TStart,TStep,NT);
  printf("Fluc_Amplification: %lf\n",Fluc_Amplification);
  printf("FlucFilePath: %s \n",FlucFilePath);
  printf("EqFileName: %s \n",EqFileName);
  printf("NTFileName: %s \n",NTFileName);
  printf("PHIFileNameStart: %s \n",PHI_FNAME_START);
  printf("DENFileNameStart: %s \n",DEN_FNAME_START);
  printf("GTSDataDir: %s \n",GTS_DATA_DIR);
  return Py_BuildValue("i",sts);
} 
/*
static PyObject*
esi_init_(PyObject* self, PyObject* args){
  int sts = 0;
  printf("MPI initializing.\n");
  if 
}
*/
static PyObject*
get_GTS_profiles_(PyObject* self, PyObject* args){
  int sts=0;
  printf("C code entered.\n");
  //parse the arguments, get ne,Te,B arrays, ne has time series.
  int toroidal_startnum;
  PyObject *input1,*input2,*input3,*input4,*input5,*input6,*input7,*input8,*input9,*input10,*input11;
  PyArrayObject *x3d,*y3d,*z3d,*ne0_arr,*Te0_arr,*Bt_arr,*Bp_arr,*dne_ad_arr,*nane_arr,*nate_arr, *mismatch_arr;
  if(!PyArg_ParseTuple(args,"OOOOOOOOOOOi",&input1,&input2,&input3,&input4,&input5,&input6,&input7,&input8,&input9,&input10,&input11,&toroidal_startnum))
    return NULL;
  printf("arguments parsed.\n");
  x3d =(PyArrayObject*) PyArray_ContiguousFromObject(input1,PyArray_DOUBLE,3,3);
  y3d =(PyArrayObject*) PyArray_ContiguousFromObject(input2,PyArray_DOUBLE,3,3);
  z3d =(PyArrayObject*) PyArray_ContiguousFromObject(input3,PyArray_DOUBLE,3,3);
  printf("x,y,z arrays got.\n");
  ne0_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input4,PyArray_DOUBLE,3,3);
  Te0_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input5,PyArray_DOUBLE,3,3);
  Bt_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input6,PyArray_DOUBLE,3,3);
  Bp_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input7,PyArray_DOUBLE,3,3);
  dne_ad_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input8,PyArray_DOUBLE,4,4);
  nane_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input9,PyArray_DOUBLE,4,4);
  nate_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input10,PyArray_DOUBLE,4,4);
  mismatch_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input11,PyArray_INT,3,3);
  printf("arrays loaded.\n"); 
  
  //start dealing with GTS data
  int n3d = NX*NY*NZ;
  double *xwant,*ywant,*zwant;
  xwant = (double*) x3d->data;
  ywant = (double*) y3d->data;
  zwant = (double*) z3d->data;
  //get cylindrical coordinates on mesh
  double Rwant[n3d],Zwant[n3d],zeta[n3d];
  cartesianToCylindrical(n3d,Rwant,Zwant,zeta,xwant,ywant,zwant);
  //convert zeta from [zeta_min,zeta_max] to [0,zeta_max-zeta_min] range. We can do this because the perturbations are statistically invariant along toroidal rotation.
  double zeta_min = findMin(n3d,zeta);
  int i;
  for(i=0;i<n3d;i++)
    zeta[i] -= zeta_min;
  printf("Finish cartesian to Cylindrical.\n");

  //initialize esi package
  double B_0,R_0;
  esiread_(&B_0,&R_0,EqFileName);

  //get corresponding flux coords
  double mag_axis_coords[2];
  get_mag_axis(mag_axis_coords);

  printf("Finish getting mag_axis.\n");

  double a[n3d],theta[n3d];//field-line coords: flux(radial), angle(poloidal), |B|
  double *Btol = (double*) Bt_arr->data;
  double Rinitial[n3d],Zinitial[n3d];//R,Z value of our initial guesses
  double Ract[n3d],Zact[n3d];//actual R,Z coordinates we have in the end
  int *InOutFlag = (int*) PyMem_Malloc(n3d*sizeof(int));//flags for points in or out LCFS
  int *mismatch = (int*) mismatch_arr->data;

  printf("Finish allocate PYthon mem.\n");
  getFluxCoords(n3d,a,theta,Btol,Ract,Zact,Rinitial,Zinitial,Rwant,Zwant,mag_axis_coords,InOutFlag,mismatch); 
  
  printf("Finish get FluxCoords.\n");

  //get the profiles
  double *Te0 = (double*) Te0_arr->data;
  double *ne0 = (double*) ne0_arr->data;
  double *Bpol = (double*) Bp_arr->data;
  double Ti0[n3d],P[n3d],qprofile[n3d];
  
  getAllProfiles(n3d,Bpol,Ti0,Te0,P,ne0,qprofile,a,theta,InOutFlag);

  printf("Finish get All profiles.\n");

  //get boundary coords (not used)
  //  double R_bdy[n_bdy], Z_bdy[n_bdy];
  //  getBoundaryPoints(R_bdy,Z_bdy,n_bdy);

  //decay equilibrium quantities outside LCFS
  decayNToutsideLCFS(n3d,a,ne0,Te0,Ti0,InOutFlag);
  
  printf("Finish decay outside LCFS.\n");
  //get the potential fluctuations
  double phi[n3d*NT];
  double *nane = (double*) nane_arr->data;
  double *nate = (double*) nate_arr->data;
  int timesteps[NT];
  for(i=0;i<NT;i++)
    timesteps[i]=TStart +TStep*i;

  int* FlucInOutFlag = (int*) PyMem_Malloc(n3d*sizeof(int));
  int ntoroidal;
  get_fluctuations(n3d, NT, &ntoroidal, phi,nane,nate, a, theta, zeta, timesteps, FlucInOutFlag, toroidal_startnum);

  printf("Finish get_fluctuations.\n");
  //electrons respond adiabatically to the potential
  double *ne_tilde = (double*) dne_ad_arr->data;
  adiabaticElectronResponse(n3d,NT,ne_tilde,ne0,phi,Te0,FlucInOutFlag);
  
  printf("Finish adiabatic response.\n"); 


  PyMem_Free(InOutFlag);
  PyMem_Free(FlucInOutFlag);
  Py_DECREF(x3d);
  Py_DECREF(y3d);
  Py_DECREF(z3d);
  
  printf("Finish decreasing instances.\n");
  return Py_BuildValue("i",ntoroidal);
  
}




static PyMethodDef Map_Mod_Methods[]={
  {"set_para_",set_parameters_,METH_VARARGS|METH_KEYWORDS, "Set the parameters used in Mapper functions. Default values can be found in source file Map_Mod_C.c."},
  {"show_para_",show_parameters_,0,"Print out current parameters."},
  /* {"esi_init_",esi_init_,0,"initialize the esi equilibrium solver."},*/
  {"get_GTS_profiles_",get_GTS_profiles_,METH_VARARGS, "Read the GTS output data.Pass in arrays: x,y,z,ne,Te,B. Where ne is in form (NT,NZ,NY,NX),others are all in (NZ,NY,NX). x,y and z need to be set properly according to the global parameters. See set_para for parameter details."},
  {NULL,NULL,0,NULL} //sentinal
};

PyMODINIT_FUNC
initMap_Mod_C(void){
  (void) Py_InitModule("Map_Mod_C",Map_Mod_Methods);
  import_array();
}
