/*
File contains functions that calculates useful quantities related to the GTS equilibrium

Using functions from esiZ120813.c

written by Lei Shi, 02/26/2015

*/
#ifndef ESIEQLOAD
#define ESIEQLOAD
#include "memalloc.h"

extern int esiget2dthese_(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int*);

struct esi_out{
  double  *Fj,*Faj,*gFaj,*gYaj,*rj,*raj,*rqj,*zj,*zaj,*zqj,*Bj,*aaa,*the;
  int n;
} esi_output;

void initialize_esi_out(double* a, double* theta, int n){
  esi_output.n = n;
  esi_output.Fj =  (double*)xmalloc(sizeof(double)*n);
  esi_output.Faj =  (double*)xmalloc(sizeof(double)*n);  
  esi_output.gFaj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.gYaj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.rj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.raj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.rqj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.zj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.zaj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.zqj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.Bj =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.aaa =  (double*)xmalloc(sizeof(double)*n); 
  esi_output.the =  (double*)xmalloc(sizeof(double)*n); 
  int i=0;
  for(i=0;i<n;i++){
    esi_output.aaa[i] = a[i];
    esi_output.the[i] = theta[i];
  }  
}

void free_esi_output(){
  xfree(esi_output.Fj);
  xfree(esi_output.Faj);
  xfree(esi_output.gFaj);
  xfree(esi_output.gYaj);
  xfree(esi_output.rj);
  xfree(esi_output.raj);
  xfree(esi_output.rqj);
  xfree(esi_output.zj);
  xfree(esi_output.zaj);
  xfree(esi_output.zqj);
  xfree(esi_output.Bj);
  xfree(esi_output.aaa);
  xfree(esi_output.the);
}

int load_esi_quantities(){
  int k;
  k = esiget2dthese_(esi_output.Fj,esi_output.Faj,esi_output.gFaj,esi_output.gYaj,esi_output.rj,esi_output.raj,esi_output.rqj,esi_output.zj,esi_output.zaj,esi_output.zqj,esi_output.Bj,esi_output.aaa,esi_output.the,&esi_output.n);
  return k;
}


double** get_metric_elements(){
  // returns a array of shape [6][n] which contains the metric tensor elements from coordinate (R,Z,Phi) to coordinate (a,theta,phi). The tensor elements are stored in the order as follows: gaa,gtt,gpp,gat,gap,gtp. where t means theta, and p means phi. Each element is a [n] array which corresponds to the n (a,theta) locations
  int i=0;
  double** metric = (double**) xmalloc(sizeof(double*) * 6);
  for(i=0;i<6;i++){
    metric[i] = (double*) xmalloc(sizeof(double)*esi_output.n);
  }
  for (i=0;i<esi_output.n;i++){
    double D = esi_output.zaj[i]*esi_output.rqj[i] - esi_output.zqj[i]*esi_output.raj[i];
    double D2 = D*D;
    metric[0][i] = (esi_output.rqj[i]*esi_output.rqj[i] + esi_output.zqj[i]*esi_output.zqj[i])/D2; // (grad a)^2
    metric[1][i] = (esi_output.zaj[i]*esi_output.zaj[i] + esi_output.raj[i]*esi_output.raj[i])/D2; // (grad theta)^2
    metric[2][i] = 1./(esi_output.rj[i]*esi_output.rj[i]); //(grad phi)^2
    metric[3][i] = (-esi_output.raj[i]*esi_output.rqj[i]-esi_output.zaj[i]*esi_output.zqj[i])/D2; //grad a * grad theta
    metric[4][i] = 0; //grad a * grad phi 
    metric[5][i] = 0; // grad theta * grad phi
  }

  return metric;
  
}

void free_metric_elements(double** metric){
  int i=0;
  for(i=0;i<6;i++){
    xfree(metric[i]);
  }
  xfree(metric);
}

double* get_Jacobian(){
  // returns the Jacobian = [grad(a) x grad(theta) . grad(phi)]^(-1) at each (a,theta) location. 

  double* J = (double*) xmalloc(sizeof(double)*esi_output.n);
  int i=0;
  for(i=0;i<esi_output.n;i++){
    J[i] = esi_output.rj[i]*(esi_output.zaj[i]*esi_output.rqj[i] - esi_output.zqj[i]*esi_output.raj[i]);
  }
  return J;
}

void free_Jacobian(double* J){
  xfree(J);
}

void free_all(double** metric, double* J){
  free_esi_output();
  free_metric_elements(metric);
  free_Jacobian(J);
}

#endif
