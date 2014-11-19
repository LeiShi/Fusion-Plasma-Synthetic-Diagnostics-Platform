//The module functions to be called in python loaded modules

#include<Python.h>
#include<numpy/arrayobject.h>
#include<stdio.h>

#include "interp.h"
#include "fileio.h"
#include "profile_coord_map.h"
#include "fluc.h"


double Xmin=2.0,Xmax=2.6,Ymin=-0.6,Ymax=0.6,Zmin=0,Zmax=0;
int NX=101,NY=201,NZ=1, NBOUNDARY = 1001;
int TStart=100, TStep=10, NT=10;
double Fluc_Amplification = 50;
char* FlucFilePath="./Fluctuations/";
char* EqFileName="./ESI_EQFILE";
char* NTFileName="./NTProfiles.cdf";
char* PHI_FNAME_START="PHI.";
char* PHI_DATA_DIR="./PHI_FILES/";

static PyObject*
set_parameters_(PyObject* self, PyObject* args, PyObject* kws){
  int sts=0;
  static char* kwlist[]={"Xmin","Xmax","NX","Ymin","Ymax","NY","Zmin","Zmax","NZ",
			 "TStart","TStep","NT","NBOUNDARY","Fluc_Amplification",
			 "FlucFilePath","EqFileName","NTFileName","PHIFileNameStart","PHIDataDir",NULL};
  if(!PyArg_ParseTupleAndKeywords(args,kws,"|ddiddiddiiiiidsssss",kwlist,
				  &Xmin,&Xmax,&NX,&Ymin,&Ymax,&NY,&Zmin,&Zmax,&NZ,
				  &TStart,&TStep,&NT,&NBOUNDARY,&Fluc_Amplification,&FlucFilePath,&EqFileName,&NTFileName,&PHI_FNAME_START,&PHI_DATA_DIR))
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
  printf("PHIDataDir: %s \n",PHI_DATA_DIR);
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
  PyObject *input1,*input2,*input3,*input4,*input5,*input6;
  PyArrayObject *x3d,*y3d,*z3d,*ne_arr,*Te_arr,*B_arr;
  if(!PyArg_ParseTuple(args,"OOOOOO",&input1,&input2,&input3,&input4,&input5,&input6))
    return NULL;
  printf("arguments parsed.\n");
  x3d =(PyArrayObject*) PyArray_ContiguousFromObject(input1,PyArray_DOUBLE,3,3);
  y3d =(PyArrayObject*) PyArray_ContiguousFromObject(input2,PyArray_DOUBLE,3,3);
  z3d =(PyArrayObject*) PyArray_ContiguousFromObject(input3,PyArray_DOUBLE,3,3);
  printf("x,y,z arrays got.\n");
  ne_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input4,PyArray_DOUBLE,4,4);
  Te_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input5,PyArray_DOUBLE,3,3);
  B_arr = (PyArrayObject*)PyArray_ContiguousFromObject(input6,PyArray_DOUBLE,3,3);

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
  
  printf("after cartesian to Cylindrical.\n");

  //initialize esi package
  double B_0,R_0;
  esiread_(&B_0,&R_0,EqFileName);

  //get corresponding flux coords
  double mag_axis_coords[2];
  get_mag_axis(mag_axis_coords);

  printf("after getting mag_axis.\n");

  double a[n3d],theta[n3d];//field-line coords: flux(radial), angle(poloidal), |B|
  double *Bm = (double*) B_arr->data;
  double Rinitial[n3d],Zinitial[n3d];//R,Z value of our initial guesses
  double Ract[n3d],Zact[n3d];//actual R,Z coordinates we have in the end
  int *InOutFlag = (int*) PyMem_Malloc(n3d*sizeof(int));//flags for points in or out LCFS

  printf("after allocate PYthon mem.\n");
  getFluxCoords(n3d,a,theta,Bm,Ract,Zact,Rinitial,Zinitial,Rwant,Zwant,mag_axis_coords,InOutFlag); 
  
  printf("after get FluxCoords.\n");

  //get the profiles
  double *Te = (double*) Te_arr->data;
  double Bpol[n3d],Ti[n3d],P[n3d],ne0[n3d],qprofile[n3d];
  getAllProfiles(n3d,Bpol,Ti,Te,P,ne0,qprofile,a,theta,InOutFlag);

  printf("after get All profiles.\n");

  //get boundary coords (not used)
  //  double R_bdy[n_bdy], Z_bdy[n_bdy];
  //  getBoundaryPoints(R_bdy,Z_bdy,n_bdy);

  //decay equilibrium quantities outside LCFS
  decayNToutsideLCFS(n3d,a,ne0,Te,Ti,InOutFlag);
  
  printf("after decay outside LCFS.\n");
  //get the potential fluctuations
  double phi[n3d*NT];
  int i;
  int timesteps[NT];
  for(i=0;i<NT;i++)
    timesteps[i]=TStart +TStep*i;

  int* FlucInOutFlag = (int*) PyMem_Malloc(n3d*sizeof(int));
  get_fluctuations(n3d, NT, phi, a, theta, zeta, timesteps, FlucInOutFlag);

  printf("after get_fluctuations.\n");
  //electrons respond adiabatically to the potential
  double *ne_tilde = (double*) ne_arr->data;
  adiabaticElectronResponse(n3d,NT,ne_tilde,ne0,phi,Te,FlucInOutFlag);
  
  printf("after adiabatic response.\n"); 

  //add ne0 onto ne_tilde to get the total ne
  for(i=0;i<NT;i++){
    int j;
    for(j=0;j<n3d;j++){
      *(ne_tilde + i*n3d +j) *= Fluc_Amplification;
      *(ne_tilde + i*n3d +j) += *(ne0 + j);
    }
  }

  PyMem_Free(InOutFlag);
  PyMem_Free(FlucInOutFlag);
  Py_DECREF(x3d);
  Py_DECREF(y3d);
  Py_DECREF(z3d);
  
  printf("after decreasing instances.\n");
  return Py_BuildValue("i",0);
  
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
