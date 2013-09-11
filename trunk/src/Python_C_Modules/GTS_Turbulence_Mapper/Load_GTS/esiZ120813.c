/*********  Autonomous Routines for Real Space Equilibrium ********/
#ifdef Add_

#define esigetb esigetb_
#define esigetrzb esigetrzb_
#define esiget2dfunctions esiget2dfunctions_
#define esiget2dthese  esiget2dthese_
#define esiread esiread_
#define esilink2c esilink2c_
#define gcmotion gcmotion_

#endif


#ifdef Compilation

cc -c -o escZ.o escZ.c	- for double precision arrays

or

cc -c -Dsingle -o escZ.o escZ.c	- for single precision arrays

#endif

#ifdef Example_Of_FORTRAN_use

      program esitest
      implicit none
      integer i,j,k
      real*8 da,dgq

      integer na1,np1,n
      parameter (na1=21,np1=65,n=na1*np1)
      integer sw(n)
      real*8 q(na1)
      real*8 r(n),ra(n),rq(n),z(n),za(n),zq(n),B(n),Ba(n),Bq(n),gh(n)
     &     ,gha(n),ghq(n)
      real*8 a(n),gq(n),F(n),Fa(n),gFa(n),gFaa(n),gYa(n),gYaa(n),T(n)
     &     ,Ta(n),P(n),Pa(n)

      integer esiread,esigetprofiles,esiget2dfunctions
     &     ,esigetrzb,gcmotion
      external esiread,esilink2c,esigetprofiles,esiget2dfunctions
     &     ,esigetrzb,gcmotion

c1.     Making link with C-ESI. Provides addresses of arrays to be used by 
ccc     esiget2dfunctions() to direct its output. Should be called only 
ccc     once (if there is no intentional redirection of the output).
      call esilink2c(F,Fa
     &     ,gFa,gFaa
     &     ,gYa,gYaa
     &     ,T,Ta
     &     ,P,Pa
     &     ,r,ra,rq
     &     ,z,za,zq
     &     ,B,Ba,Bq
     &     ,gh,gha,ghq
     &     ,sw) 	!integer label of the particle

c2.    Reading data file (s) File names: <A-Z,a-z>,<0-9>,<_>,<.>
      k=esiread('esiA.00 ') 	! one read per equiulibrium

      if(k.ne.0) then
         write(*,*)'bad file for reading'
         stop
      endif

ccc   Example of getting plasma profiles as a function of minor radius

      write(*,'(a2,7a10)')'i','a','F','gFa','gYa','q','T','P'
      da=1./(na1-1)
      do i =1,na1
         a(i)=da*(i-1)
         k=esigetprofiles(F(i),Fa(i), gFa(i),gFaa(i), gYa(i),gYaa(i)
     &        ,T(i),Ta(i), P(i),Pa(i), a(i))
         if(i.eq.1) then
            q(i)=-gYaa(i)/gFaa(i)
         else
            q(i)=-gYa(i)/gFa(i)
         endif
         write(*,'(i2,1p7e10.3)')i,a(i),F(i),gFa(i),gYa(i),q(i),T(i),P(i
     &        )
      enddo
ccc

ccc   Example of setting coordinates of particles

      da=0.8/(na1-1)
      dgq=8.*datan(1.d0)/(np1-1)
      k=0
      do i=1,na1
         do j=1,np1
            k=k+1
            a(k)=da*(i-1)+0.1
            gq(k)=dgq*(j-1)+0.2
         enddo
      enddo
ccc

c3.   Example of getting 2-D functions. Output goes to the arrays given by
ccc   esilink2c(). Designed for multiple calls inside the time step loops.

      k=esiget2dfunctions(a,gq,n)

      if(k.ne.0) then
         write(*,*)'wrong a or gq for particle #',k
         call esifree()
         stop
      endif

      write(*,'(1p6a10)')'r','z','ra','za','rq','zq'
      do i=4,6
         j=np1*i
         write(*,'(a,i2)')"i=",i
         write(*,'(1p6e10.3)')(r(k),z(k),ra(k),za(k),rq(k),zq(k)
     &        ,k=j-np1+1,j)
         j=j+np1
      enddo
c end of 3.

c4.   Example of getting r,z,B in a number of points

      k=esigetrzb(r,z,B,a,gq,n)
      if(k.ne.0) then
         write(*,*)'wrong a or gq for particle #',k
         call esifree()
         stop
      endif
c end of 4.

c5.   Example of getting time derivatives of the guiding center 
c     coordinates gr_parallel,a,theta,phi of n particles with magnetic 
c     momentum mu

      k=gcmotion(dgr,da,dgq,dgf,gr,a,gq,gm,n)
c end of 5

c6    Example of setting output for density value at the particle position
c     next routine sends adresses of arrays ne,dne for density and it 
c     derivative to ESI
c     Should be called just once

      call esilink2c4ne(ne,dne)

c     Specifying tempterature at the point (nT is known only by ESI)

      Te=3.
      Ti=3.
      call settemperatures(Te,Ti)
c     After this density will be calculated during calls of 
c     esiget2dfunctions()

      call esifree()
c end of 2.   Freeing ESI    
      end
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

#include <Python.h>

#ifndef single
typedef double REAL;
#define MPI_RSIZE MPI_DOUBLE
#else
typedef float REAL;
#define MPI_RSIZE MPI_FLOAT 
#endif

static double cr2,cr3,cr4,cr6,cr12,cgp,c2gp,cgp_4,cr2gp,cgm0,crgm0;
static double c2rqgp;
static int kD=sizeof(double),kI=sizeof(int);

static int Na,Na1=0,Np,Np1=0,MemFl=0,FlMeshRZ=1;
static REAL *sa,*gq,*aF,*daF,*dgF,*d2gF,*dgY,*d2gY,*aT,*daT,*aP,*daP;
static REAL *sne,*dsne;
static REAL *sr,*sra,*srq,*sraq;
static REAL *sz,*sza,*szq,*szaq;
static REAL *aB,*aBa,*aBq,*aBaq;
static REAL *gH,*gHa,*gHq,*gHaq,*gHqq,*gHaqq;

static double Zbnd0,sb;
static double *Rbnd,*Rbnd1,*Zbnd,*Zbnd1;

static int Nr=32,Nr1=33,Nz=32,Nz1=33;
static double *aLab,*qLab,rLab1,rLab2,zLab1,zLab2,dzLab,rdzLab;
static double rL1[0x21],rL2[0x21],zLab[0x21];

static int i0=0,i1=1,j00,j10,j01,j11; 
static REAL A,ha,rha,hq,rhq,cgq0,crgq0;/* period and inverse period in gq */
static REAL xF,XF,xD,XD,dX,dxD,dXD,yF,YF,yD,YD,dY,dyD,dYD;
static REAL y,Y,yy,YY;

static int nF=128;
static double F0[128],F1[128],F2[128],F3[128],tF1=5.,hF,rhF;

int qHermit(double *f, double *df,double x,double *g,double *dg)
{
  double t;
  double p1,p2,p3;

  p1	=hq*dg[0];
  p3	=hq*dg[1];
  t	=g[1]-g[0];
  p2	=3.*t-2.*p1-p3;
  p3	+=p1-2.*t;

  t	=x*rhq;

  *f	=g[0]+(p1+(p2+p3*t)*t)*t;
  if(df != NULL) *df	=rhq*(p1+(2.*p2+3.*p3*t)*t);
  return(0);
}


int InitLocalConstants()
{
  cr2	=0.5;
  cr3	=1./3.;
  cr4	=0.25;
  cr6	=1./6.;
  cr12	=1./12.;
  cgp_4	=atan((double)1.);
  cgp	=4.*cgp_4;
  c2gp	=8.*cgp_4;
  cr2gp	=1./c2gp;
  cgm0	=0.4*cgp;
  crgm0	=1./cgm0;

  c2rqgp=2./sqrt(cgp);
  return(0);
}

int esifree()
{
  if(aLab != NULL){
    PyMem_Free(aLab);
    aLab	=NULL;	
  }
  PyMem_Free(gH);
  PyMem_Free(aB);
  PyMem_Free(sz);
  PyMem_Free(sr);
  PyMem_Free(gq);
  MemFl=0;
}

int SetFunctionErf()
{
  int i;
  double x,t,dt,s,r6h,r2h,F,dF;
  double S,dY0,dY1,d2Y0,d2Y1,f;
  
  hF	=tF1/nF;
  rhF	=1./hF;
  r2h	=0.5*hF;
  r6h	=hF/6.;

  S	=0.;
  dY1	=0.;
  d2Y1	=0.;

  f	=2.*cr3*c2rqgp;
  F	=0.;
  dF	=0.;
  for(i=0; i < nF; i++){
    F0[i]	=F;
    F1[i]	=dF;
    t	=hF*(i+1);
    dY0	=dY1;
    d2Y0=d2Y1;
    x	=t*t;
    s	=exp(-x);
    dY1	=x*x*x*s;
    d2Y1=t*x*x*s*(6.-2.*x);
    S	+=(dY0+dY1+r6h*(d2Y0-d2Y1))*r2h;
    F	=f*(s*(1.+0.4*x)+0.8*S/(t*x));
    dF	=F-t*f*(1.2*t*s+2.4*S/(x*x));
    F	*=t;
    F2[i]=3.*(F-F0[i])-hF*(2.*F1[i]+dF);
    F3[i]=-2.*(F-F0[i])+hF*(F1[i]+dF);
  }
  return(0);
}

int ESIMemAlloc()
{
  int i,j;

  InitLocalConstants();
  SetFunctionErf();
  gq	=(REAL*)PyMem_Malloc((Np1+13*Na1)*sizeof(REAL));
  if(gq != NULL){
    MemFl	+=0x01;
  }
  i	=Na1*Np1;
  j	=4*i;
  sr	=(REAL *)PyMem_Malloc(j*sizeof(REAL));
  if(sr != NULL){
    MemFl	+=0x02;
  }
  sz	=(REAL *)PyMem_Malloc(j*sizeof(REAL));
  if(sz != NULL){
    MemFl	+=0x04;
  }
  aB	=(REAL *)PyMem_Malloc(j*sizeof(REAL));
  if(aB != NULL){
    MemFl	+=0x08;
  }
  gH	=(REAL *)PyMem_Malloc((j+2*i)*sizeof(REAL));
  if(gH != NULL){
    MemFl	+=0x10;
  }
  if(MemFl != 0x1F){
    printf("No memory for BiCubic ESI MemFl=0x%x%c\n",7,MemFl);
    if((MemFl&0x10)){
      PyMem_Free(gH);
    }
    if((MemFl&0x08)){
      PyMem_Free(aB);
    }
    if((MemFl&0x04)){
      PyMem_Free(sz);
    }
    if((MemFl&0x02)){
      PyMem_Free(sr);
    }
    if((MemFl&0x01)){
      PyMem_Free(sa);
    }
    return(1);
  }
  sa	=gq	+Np1;
  aF	=sa	+Na1;
  daF	=aF	+Na1;
  dgF	=daF	+Na1;
  d2gF=dgF	+Na1;
  dgY	=d2gF	+Na1;
  d2gY=dgY	+Na1;
  aT	=d2gY	+Na1;
  daT	=aT	+Na1;
  aP	=daT	+Na1;
  daP	=aP	+Na1;
  sne	=daP	+Na1;
  dsne	=sne	+Na1;
  
  sra	=sr	+i;
  srq	=sra	+i;
  sraq=srq	+i;
  sza	=sz	+i;
  szq	=sza	+i;
  szaq=szq	+i;
  aBa	=aB	+i;
  aBq	=aBa	+i;
  aBaq=aBq	+i;
  gHa	=gH	+i;
  gHq	=gHa	+i;
  gHaq=gHq	+i;
  gHqq=gHaq	+i;
  gHaqq=gHqq	+i;

  Rbnd	=sra-Np1;
  Rbnd1	=sraq-Np1;
  Zbnd	=sza-Np1;
  Zbnd1	=szaq-Np1;
  aLab	=(REAL*)PyMem_Malloc(2*Nr1*Nz1*kD);
  if(aLab == NULL){
    printf("Failure: No memory for ESI\n");
    return(1);
  }
  qLab	=aLab+Nr1*Nz1;
  return(0);
}

int ESICopy(double *esr,double *esz,double *eaB,double *egH,int na1,int np1)
{
  int i,n;
  double *esra,*esrt,*esrat,*esza,*eszt,*eszat,*eaBa,*eaBt,*eaBat
    ,*egHa,*egHt,*egHat;

  n	=np1*na1;
  esra	=esr	+n;
  esrt	=esra	+n;
  esrat	=esrt	+n;
  esza	=esz	+n;
  eszt	=esza	+n;
  eszat	=eszt	+n;
  eaBa	=eaB	+n;
  eaBt	=eaBa	+n;
  eaBat	=eaBt	+n;
  egHa	=egH	+n;
  egHt	=egHa	+n;
  egHat	=egHt	+n;
  for(i=0; i < n; i++){
    esr[i]	=sr[i];
    esra[i]	=sra[i];
    esrt[i]	=-srq[i];
    esrat[i]	=-sraq[i];
    esz[i]	=sz[i];
    esza[i]	=sza[i];
    eszt[i]	=-szq[i];
    eszat[i]	=-szaq[i];
    eaB[i]	=aB[i];
    eaBa[i]	=aBa[i];
    eaBt[i]	=-aBq[i];
    eaBat[i]	=-aBaq[i];
    egH[i]	=-gH[i];
    egHa[i]	=-gHa[i];
    egHt[i]	=gHq[i];
    egHat[i]	=gHaq[i];

    egH[i]	=gHq[i];
    egHa[i]	=gHaq[i];
    egHt[i]	=-gHqq[i];
    egHat[i]	=-gHaqq[i];
  }
  return(0);
}

/* Binary Double */
int ESIReadBD(char* FNm) 
{
  int i,j,k;
  FILE *lf;
  REAL s,ss;

  lf	=fopen(FNm,"r");
  if(lf == NULL){
    printf("%s !!! cannot be open%c\n",FNm,7);
    return(1);
  }
 
  fread(&j,sizeof(int),1,lf);
  fread(&i,sizeof(int),1,lf);
  if(MemFl == 0 || Np1 != j || Na1 != i){
    if(MemFl){
      esifree();
    }
    Np1	=j;
    Na1	=i;
    if(ESIMemAlloc()){
      return(1);
    }
  } 
  Np	=Np1-1;
  Na	=Na1-1;
  fread(gq,sizeof(REAL),Np1,lf);
  fread(sa,sizeof(REAL),Na1,lf);
  fread(aF,sizeof(REAL),Na1,lf);
  fread(daF,sizeof(REAL),Na1,lf);
  fread(dgF,sizeof(REAL),Na1,lf);
  fread(d2gF,sizeof(REAL),Na1,lf);
  fread(dgY,sizeof(REAL),Na1,lf);
  fread(d2gY,sizeof(REAL),Na1,lf);
  fread(aT,sizeof(REAL),Na1,lf);
  fread(daT,sizeof(REAL),Na1,lf);
  fread(aP,sizeof(REAL),Na1,lf);
  fread(daP,sizeof(REAL),Na1,lf);
  j	=0;
  for(i=0; i < Na1; i++){
    fread(sr+j,sizeof(REAL),Np1,lf);
    fread(sra+j,sizeof(REAL),Np1,lf);
    fread(srq+j,sizeof(REAL),Np1,lf);
    fread(sraq+j,sizeof(REAL),Np1,lf);
    fread(sz+j,sizeof(REAL),Np1,lf);
    fread(sza+j,sizeof(REAL),Np1,lf);
    fread(szq+j,sizeof(REAL),Np1,lf);
    fread(szaq+j,sizeof(REAL),Np1,lf);
    fread(aB+j,sizeof(REAL),Np1,lf);
    fread(aBa+j,sizeof(REAL),Np1,lf);
    fread(aBq+j,sizeof(REAL),Np1,lf);
    fread(aBaq+j,sizeof(REAL),Np1,lf);
    fread(gHq+j,sizeof(REAL),Np1,lf);
    fread(gHaq+j,sizeof(REAL),Np1,lf);
    fread(gHqq+j,sizeof(REAL),Np1,lf);
    if(fread(gHaqq+j,sizeof(REAL),Np1,lf) != Np1){
      fclose(lf);
      printf("%s Bad binary file\n",FNm);
      return(1);
    }
    j	+=Np1;
  }
  fclose(lf);
  return(0);
}

int ESIReadAD(char* FNm) 
{
  int i,j,k;
  double s;
  FILE *lf;
  char ln[256],*lc;

    fprintf(stderr,"In ESIReadAD: reading file <%s>\n",FNm,7);

  lf	=fopen(FNm,"r");
  if(lf == NULL){
    fprintf(stderr,"File=<%s> !!! cannot be open%c\n",FNm,7);
    return(1);
  }
  lc	=ln;
  while(feof(lf) == 0 && (*lc=fgetc(lf)) != '\n'){
    if(lc-ln < 25){
      lc++;
    }
  }
  *lc	='\0';
  i	=lc-ln;
  lc	="!!! Do not edit this file";
  if(i < strlen(lc) || strncmp(ln,lc,strlen(lc))){
#ifdef H
    printf("%s first line is not !!! Do not edit this file%c\n",FNm,7);
#endif
    fclose(lf);
    return(2);
  }
  if(fscanf(lf,"%d x%d",&j,&i) != 2){
    fprintf(stderr,"%s - wrong data for Np1, Na1%c\n",FNm,7);
    return(1);
  }
  lc	=ln;
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  if(MemFl == 0 || Np1 != j || Na1 != i){
    if(MemFl){
      esifree();
    }
    Np1	=j;
    Na1	=i;
    if(ESIMemAlloc()){
      return(1);
    }
  } 
  Np	=Np1-1;
  Na	=Na1-1;
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(j=0; j < Np1; j++){
    if(fscanf(lf,"%lg",gq+j) != 1){
      fprintf(stderr,"%s - wrong data for gq[%d]%c\n",FNm,j,7);
      return(1);
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(i=0; i < Na1; i++){
    if(fscanf(lf,"%lg%lg%lg",sa+i,aF+i,daF+i) != 3){
      fprintf(stderr,"%s - wrong data for a,F,F'[%d]%c\n",FNm,i,7);
      return(1);
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(i=0; i < Na1; i++){
    if(fscanf(lf,"%lg%lg%lg%lg",dgF+i,d2gF+i,dgY+i,d2gY+i) != 4){
      fprintf(stderr,"%s - wrong data for gF',gF'',gY',gY'',[%d]%c\n",FNm,i,7);
      return(1);
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  for(i=0; i < Na1; i++){
    if(fscanf(lf,"%lg%lg%lg%lg",aT+i,daT+i,aP+i,daP+i) != 4){
      fprintf(stderr,"%s - wrong data for T,T',P,P'[%d]%c\n",FNm,i,7);
      return(1);
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",sr+k,sra+k,srq+k,sraq+k) != 4){
	fprintf(stderr,"%s - wrong data [j=%d,i=%d] for r,r'_a,r'_gq,r''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",sz+k,sza+k,szq+k,szaq+k) != 4){
	fprintf(stderr,"%s - wrong data [j=%d,i=%d] for z,z'_a,z'_gq,z''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",aB+k,aBa+k,aBq+k,aBaq+k) != 4){
	fprintf(stderr,"%s - wrong data [j=%d,i=%d] for B,B'_a,B'_gq,B''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",gHq+k,gHaq+k,gHqq+k,gHaqq+k) != 4){
	fprintf(stderr,"%s -wrong data [j=%d,i=%d] for gh,gh'_a,gh'_gq,gh''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(i=0; i < Na1; i++){
    if(fscanf(lf,"%lg%lg%lg",&s,sne+i,dsne+i) != 3){
      if(i == 0){
	break;
      }
      fprintf(stderr,"%s -wrong data [i=%d] for ne,dne%c\n",FNm,i,7);
      return(1);
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  fclose(lf);
  return(0);
}

#ifdef single
/* Binary Float */
int ESIReadBF(char* FNm) 
{
  int i,j,k;
  FILE *lf;
  REAL s,ss,*ld;
  double *d;

  lf	=fopen(FNm,"r");
  if(lf == NULL){
    fprintf(stderr,"%s !!! cannot be open%c\n",FNm,7);
    return(1);
  }
  fread(&j,sizeof(int),1,lf);
  fread(&i,sizeof(int),1,lf);
  if(MemFl == 0 || Np1 != j || Na1 != i){
    if(MemFl){
      esifree();
    }
    Np1	=j;
    Na1	=i;
    if(ESIMemAlloc()){
      return(1);
    }
  } 
  Np	=Np1-1;
  Na	=Na1-1;
  d	=(double*)PyMem_Malloc(Np1*sizeof(double));
  fread(d,sizeof(double),Np1,lf);
  ld	=gq;
  for(k=0; k < Np1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=sa;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=aF;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=daF;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=dgF;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=d2gF;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=dgY;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=d2gY;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=aT;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=daT;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=aP;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  fread(d,sizeof(double),Na1,lf);
  ld	=daP;
  for(k=0; k < Na1; k++){
    *ld++	=d[k];
  }
  j	=0;
  for(i=0; i < Na1; i++){
    fread(d,sizeof(double),Np1,lf);
    ld	=sr+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=sra+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=srq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=sraq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=sz+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=sza+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=szq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=szaq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=aB+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=aBa+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=aBq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=aBaq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=gHq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=gHaq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=gHqq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    fread(d,sizeof(double),Np1,lf);
    ld	=gHaqq+j;
    for(k=0; k < Na1; k++){
      *ld++	=d[k];
    }
    j	+=Np1;
  }
  PyMem_Free(d);
  fclose(lf);
  return(0);
}

int ESIReadAF(char* FNm) 
{
  int i,j,k;
  FILE *lf;
  char ln[256],*lc;
  double d[4];

  lf	=fopen(FNm,"r");
  if(lf == NULL){
    printf("%s !!! cannot be open%c\n",FNm,7);
    return(1);
  }
  lc	=ln;
  while(feof(lf) == 0 && (*lc=fgetc(lf)) != '\n'){
    if(lc-ln < 25){
      lc++;
    }
  }
  *lc	='\0';
  i	=lc-ln;
  lc	="!!! Do not edit this file";
  if(i < strlen(lc) || strncmp(ln,lc,strlen(lc))){
    printf("%s first line is not !!! Do not edit this file%c\n",FNm,7);
    fclose(lf);
    return(2);
  }
  if(fscanf(lf,"%d x%d",&j,&i) != 2){
    printf("%s - wrong data on Np, Na%c\n",FNm,7);
    return(1);
  }
  *lc	=ln;
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  if(MemFl == 0 || Np1 != j || Na1 != i){
    if(MemFl){
      esifree();
    }
    Np1	=j;
    Na1	=i;
    if(ESIMemAlloc()){
      return(1);
    }
  } 
  Np	=Np1-1;
  Na	=Na1-1;
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(j=0; j < Np1; j++){
    if(fscanf(lf,"%lg",d) != 1){
      printf("%s - wrong data for gq[%d]%c\n",FNm,j,7);
      return(1);
    }
    gq[j]	=d[0];
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(i=0; i < Na1; i++){
    if(fscanf(lf,"%lg%lg%lg",d,d+1,d+2) != 3){
      printf("%s - wrong data for a,F,F'[%d]%c\n",FNm,i,7);
      return(1);
    }
    sa[i]	=d[0];
    aF[i]	=d[1];
    daF[i]	=d[2];
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(i=0; i < Na1; i++){
    if(fscanf(lf,"%lg%lg%lg%lg",d,d+1,d+2,d+3) != 4){
      printf("%s - wrong data for gF',gF'',gY',gY''[%d]%c\n",FNm,i,7);
      return(1);
    }
    dgF[i]	=d[0];
    d2gF[i]	=d[1];
    dgY[i]	=d[2];
    d2gY[i]	=d[3];
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  for(i=0; i < Na1; i++){
    if(fscanf(lf,"%lg%lg%lg%lg",d,d+1,d+2,d+3) != 4){
      printf("%s - wrong data for T,T',P,P'[%d]%c\n",FNm,i,7);
      return(1);
    }
    aT[i]	=d[0];
    daT[i]	=d[1];
    aP[i]	=d[2];
    daP[i]	=d[3];
  }	
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",d,d+1,d+2,d+3) != 4){
	printf("%s - wrong data [j=%d,i=%d] for r,r'_a,r'_gq,r''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      sr[k]	=d[0];
      sra[k]	=d[1];
      srq[k]	=d[2];
      sraq[k]	=d[3];
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",d,d+1,d+2,d+3) != 4){
	printf("%s - wrong data [j=%d,i=%d] for z,z'_a,z'_gq,z''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      sz[k]	=d[0];
      sza[k]	=d[1];
      szq[k]	=d[2];
      szaq[k]	=d[3];
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",d,d+1,d+2,d+3) != 4){
	printf("%s - wrong data [j=%d,i=%d] for B,B'_a,B'_gq,B''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      aB[k]	=d[0];
      aBa[k]	=d[1];
      aBq[k]	=d[2];
      aBaq[k]	=d[3];
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }

  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  k	=0;
  for(i=0; i < Na1; i++){
    for(j=0; j < Np1; j++){
      if(fscanf(lf,"%lg%lg%lg%lg",d,d+1,d+2,d+3) != 4){
	printf("%s -wrong data [j=%d,i=%d] for gh,gh'_a,gh'_gq,gh''_{a,gq}%c\n"
	       ,FNm,j,i,7);
	return(	1);
      }
      gHq[k]	=d[0];
      gHaq[k]	=d[1];
      gHqq[k]	=d[2];
      gHaqq[k]	=d[3];
      k++;
    }
  }
  while(feof(lf) == 0 && (*lc=fgetc(lf)) !='\n'){
    ;
  }
  fclose(lf);
  return(0);
}
#endif

int ESIMakeMesh()
{
  int i,i0,ji,iter;
  int j,j0,j1,k;
  double x,X,xx,XX,err=1e-9;
  double y,Y,yy,YY;
  double f0,f1,fq0,fq1;
  double a,q,a0,q0,R,Z,r,z,ra,za,rq,zq;
  double D,da,dq,dr,dz,dR;
  double qL1[0x100],qL2[0x100];

//  printf("!! Attention!! : MakeMesh Removed\n");
//  return(0);


  FlMeshRZ=1;
  zLab2	=Zbnd[3*Np/4];
  zLab1	=Zbnd[Np/4];
  dzLab=(zLab2-zLab1)/Nz;
  rdzLab=1./dzLab;
  
  zLab[0]	=zLab1;
  rL1[0]	=Rbnd[Np/4];
  rL2[0]	=rL1[0];
  qL1[0]	=0.5*cgp;
  qL2[0]	=qL1[0];
  for(i=1; i < Nz; i++){
    zLab[i]	=zLab1+dzLab*i;
    q	=-asin((zLab[i]-Zbnd0)/sb);
    qL1[i]	=cgp-q;
    j	=qL1[i]*rhq;
    qHermit(rL1+i,NULL,qL1[i]-gq[j],Rbnd+j,Rbnd1+j);
    if(q < 0.) q+=c2gp;
    qL2[i]	=q;
    j	=q*rhq;
    qHermit(&z,NULL,q-gq[j],Zbnd+j,Zbnd1+j);
    qHermit(rL2+i,NULL,q-gq[j],Rbnd+j,Rbnd1+j);
  }
  zLab[Nz]	=zLab2;
  rL1[Nz]	=Rbnd[3*Np/4];
  rL2[Nz]	=rL1[Nz];
  qL1[Nz]	=1.5*cgp;
  qL2[Nz]	=qL1[Nz];
  for(i=0; i < Nz1; i++){
    a	=1.;
    q	=qL1[i];
    j	=q*rhq;
    qHermit(&z,NULL,q-gq[j],Zbnd+j,Zbnd1+j);
    q	=qL2[i];
    j	=q*rhq;
    qHermit(&z,NULL,q-gq[j],Zbnd+j,Zbnd1+j);
  }

  r	=0.;
  rLab1	=1e+6;
  for(j=0; j < Np1; j++){
    if(r < Rbnd[j]){
      r=Rbnd[j];
      j1=j;
    }
    if(rLab1 > Rbnd[j]){
      rLab1=Rbnd[j];
      j0=j;
    }
  }
  if(Rbnd1[j1] < 0.) j1--;
  f0	=Rbnd[j1];
  f1	=hq*Rbnd1[j1];
  j	=j1+1;
  fq0	=hq*Rbnd1[j];
  rq	=Rbnd[j]-Rbnd[j1];
  fq1	=f1+fq0-2.*rq;
  fq0	=3.*rq-2.*f1-fq0;
  q	=f1/(sqrt(fq0*fq0-3.*f1*fq1)-fq0);
  rLab2	=f0+q*(f1+q*(fq0+q*fq1));

  if(Rbnd1[j0] < 0.) j0--;
  f0	=Rbnd[j0];
  f1	=hq*Rbnd1[j0];
  j	=j0+1;
  fq0	=hq*Rbnd1[j];
  rq	=Rbnd[j]-Rbnd[j0];
  fq1	=f1+fq0-2.*rq;
  fq0	=3.*rq-2.*f1-fq0;
  q	=-f1/(sqrt(fq0*fq0-3.*fq1*f1)+fq0);
  rLab1	=f0+q*(f1+q*(fq0+q*fq1));

  ji	=0.;
  i	=0;
  for(j=0; j < Nr1; j++){
    aLab[ji]	=1.;
    qLab[ji]	=0.5*cgp;
    ji++;
  }
  for(i=1; i < Nz; i++){
    Z	=zLab[i];
    dR	=(rL2[i]-rL1[i])/Nr;
    a	=1.;
    q	=qL1[i];
    aLab[ji]	=a;
    qLab[ji]	=q;
    j	=0;
    ji++;
    for(j=1; j < Nr; j++){
      R	=rL1[i]+dR*j;
      iter=0;
      do{
	j0	=q*rhq;
	j1	=j0+1;
	y	=(q-gq[j0])*rhq;
	Y	=1.-y;
   
	YY	=Y*Y;
	yy	=y*y;
	YF	=YY*(3.-2.*Y);
	YD	=YY*y*hq;
	yF	=yy*(3.-2.*y);
	yD	=-yy*Y*hq;
    
	dY	=3.*y*Y;
	dYD	=Y-dY;
	dyD	=y-dY;
	dY	*=2.*rhq;
	if(a < 0.) a=-a;
	i0	=(a-sa[0])*rha;
	if(i0 >= Na) i0--;
	if(a != 0.){
	  x	=(a-sa[i0])*rha;
	  X	=1.-x;

	  XX	=X*X;
	  xx	=x*x;
	  XF	=XX*(3.-2.*X);
	  XD	=XX*x*ha;
	  xF	=xx*(3.-2.*x);
	  xD	=-xx*X*ha;
      
	  dX	=3.*x*X;
	  dXD	=X-dX;
	  dxD	=x-dX;
	  dX	*=2.*rha;
      
	  j00	=Np1*i0+j0;
	  j01	=j00+1;
	  j10	=j00+Np1;
	  j11	=j10+1;

	  f0	=XF*sr [j00]+xF*sr [j10]+XD*sra [j00]+xD*sra [j10];
	  f1	=XF*sr [j01]+xF*sr [j11]+XD*sra [j01]+xD*sra [j11];
	  fq0	=XF*srq[j00]+xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
	  fq1	=XF*srq[j01]+xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
	  r		=YF*f0+yF*f1+YD*fq0+yD*fq1;
	  rq	=dY*(f1-f0)+dYD*fq0+dyD*fq1;
	  f0	=dX*(sr [j10]-sr [j00])+dXD*sra [j00]+dxD*sra [j10];
	  f1	=dX*(sr [j11]-sr [j01])+dXD*sra [j01]+dxD*sra [j11];
	  fq0	=dX*(srq[j10]-srq[j00])+dXD*sraq[j00]+dxD*sraq[j10];
	  fq1	=dX*(srq[j11]-srq[j01])+dXD*sraq[j01]+dxD*sraq[j11];
	  ra	=YF*f0+yF*f1+YD*fq0+yD*fq1;

	  f0	=XF*sz [j00]+xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
	  f1	=XF*sz [j01]+xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
	  fq0	=XF*szq[j00]+xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
	  fq1	=XF*szq[j01]+xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
	  z		=YF*f0+yF*f1+YD*fq0+yD*fq1;
	  zq	=dY*(f1-f0)+dYD*fq0+dyD*fq1;
	  f0	=dX*(sz [j10]-sz [j00])+dXD*sza [j00]+dxD*sza [j10];
	  f1	=dX*(sz [j11]-sz [j01])+dXD*sza [j01]+dxD*sza [j11];
	  fq0	=dX*(szq[j10]-szq[j00])+dXD*szaq[j00]+dxD*szaq[j10];
	  fq1	=dX*(szq[j11]-szq[j01])+dXD*szaq[j01]+dxD*szaq[j1];
	  za	=YF*f0+yF*f1+YD*fq0+yD*fq1;
	}
	else{	/* a == 0. */
	  r	=sr[0];
	  ra	=YF*sra[j0]+yF*sra[j1]+YD*sraq[j0]+yD*sraq[j1];
	  rq	=dY*(sra[j1]-sra[j0])+dYD*sraq[j0]+dyD*sraq[j1];
	  z	=sz[0];
	  za	=YF*sza[j0]+yF*sza[j1]+YD*szaq[j0]+yD*szaq[j1];
	  zq	=dY*(sza[j1]-sza[j0])+dYD*szaq[j0]+dyD*szaq[j1];
	}
	D	=1./(za*rq-zq*ra);
	dr	=R-r;
	dz	=Z-z;
	da	=(rq*dz-zq*dr)*D;
	dq	=(za*dr-ra*dz)*D;
	a	+=da;
	q	+=dq;
	while(q < 0.) q+=c2gp;
	while(q  > c2gp) q-=c2gp;
	if(a< 0.){
	  a=-a;
	  q=q > cgp ? q-cgp : q+cgp;
	}
	iter++;
      }while(iter < 10 && fabs(da)+fabs(dq) > err);
      if(iter > 9){
	printf("Error: %2d %2d %10.3e %10.3e %10.3e %10.3e %10.3e\n"
	       ,i,j,R,Z,r,z,fabs(da)+fabs(dq));
	return(1);
      }
      aLab[ji]	=a;
      qLab[ji]	=q;
      ji++;
    }
    aLab[ji]	=1.;
    qLab[ji]	=qL2[i];
    ji++;
  }
  for(j=0; j < Nr1; j++){
    aLab[ji]	=1.;
    qLab[ji]	=1.5*cgp;
    ji++;
  }
  FlMeshRZ=0;
  return(0);
}

int ESIInit()
{
  int i,j,ji,ji1;
  REAL s,ss;

  cgq0	=gq[Np];
  crgq0	=1./cgq0;
  hq	=gq[1];
  rhq	=1./hq;
  ha	=sa[1]-sa[0];
  rha	=1./ha;

  ji	=0;
  for(i=0; i < Na1; i++){
    s	=0;
    ss	=0;
    gH[ji]	=0.;
    gHa[ji]	=0.;
    for(j=1; j < Np1; j++){
      ji1	=ji;
      ji++;
      gH[ji]	=gH[ji1]+((gHq[ji1]+gHq[ji])+cr6*hq*(gHqq[ji1]-gHqq[ji]))
	*hq*0.5;
      gHa[ji]	=gHa[ji1]+((gHaq[ji1]+gHaq[ji])
			   +cr6*hq*(gHaqq[ji1]-gHaqq[ji]))*hq*0.5;
      s	+=gH[ji];
      ss+=gHa[ji];
    }
    ji++;
#ifdef H
    s	/=Np;
    ss	/=Np;
    ji	-=Np;
    ji++;
    for(j=1; j < Np1; j++){
      gH[ji]	-=s;
      gHa[ji]	-=ss;
      ji++;
    }
#endif
  }

  j	=Np1*Na+Np/4;
  ji	=j+Np/2;
  Zbnd0	=0.5*(sz[ji]+sz[j]);
  sb	=0.5*(sz[ji]-sz[j]);
  ESIMakeMesh();
  return(0);
}

int esiread(REAL *b_axis, REAL *rmajor, char* FName)
{
  int i;
  char FNm[128],*lc,*ls;
  int *ll0,*ll1,*ll2;

  int mype,icount,int_buff[2];
  MPI_Comm_rank(MPI_COMM_WORLD,&mype);

#if ( PRINTlevel>=2 )
  fprintf(stderr,"%d:In esiread: reading file <%s>\n",mype,FName,7);
#endif

  i=0;

  ls	=FName;
  while(isspace(*ls)){
    ls++;
  }
  lc	=FNm;
  while(*ls != '\0' && lc-FNm < 127 && !isspace(*ls)){
    *lc++	=*ls++;
  }
#ifdef H
  while(!isspace(*ls) && lc-FNm < 127){
    *lc++	=*ls++;
  }
#endif
  *lc	='\0';
#ifndef single
/* Only the master process (mype=0) reads the file */
  if(mype == 0){
    if((i=ESIReadAD(FNm)) == 2){ 
      i	=ESIReadBD(FNm);
    }
  }
#else
/* Only the master process (mype=0) reads the file */
  if(mype == 0){
    if((i=ESIReadAF(FNm)) == 2){ 
      i	=ESIReadBF(FNm);
    }
  }
#endif
  if(i){
    return(i);
  }
/* The reading operation was successful so now we need to broadcast the newly
   read quantities to the other processors. We first send "Np1" and "Na1"
   since they are required to allocate memory for the other quantities */
  int_buff[0] = Np1;
  int_buff[1] = Na1;
  MPI_Bcast(int_buff,2,MPI_INT,0,MPI_COMM_WORLD);

/* Np and Na are set only for the root processor at this point. They are global
   variables that need to also be set for all the other processors. */
  if(mype != 0){
    Np1 = int_buff[0];
    Na1 = int_buff[1];
    Np = Np1-1;
    Na = Na1-1;
  /* Allocate memory for the other quantities */
    if(ESIMemAlloc()){
      return(1);
    }
  }
/* Now the master can broadcast all of the other quantities that were read. */
  icount = Np1 + 13*Na1;  /* Must match memory allocation in ESIMemAlloc() */
  MPI_Bcast(gq,icount,MPI_RSIZE,0,MPI_COMM_WORLD);

  icount = 4*Na1*Np1;
  MPI_Bcast(sr,icount,MPI_RSIZE,0,MPI_COMM_WORLD);
  MPI_Bcast(sz,icount,MPI_RSIZE,0,MPI_COMM_WORLD);
  MPI_Bcast(aB,icount,MPI_RSIZE,0,MPI_COMM_WORLD);

  icount = 6*Na1*Np1;
  MPI_Bcast(gH,icount,MPI_RSIZE,0,MPI_COMM_WORLD);

  ESIInit();

  *b_axis= aB[0];
  *rmajor= sr[0];

#if ( PRINTlevel>=2 )
     fprintf(stderr,"mype=%d :magnetic field at axis B_0=%f  major radius R_0=%f  Na1=%d  Np1=%d\n",mype,*b_axis,*rmajor,Na1,Np1);
     fprintf(stderr,"%d: gq[2]=%f  sr[2]=%f  sz[2]=%f  aB[2]=%f  gH[2]=%f\n",mype,gq[2],sr[2],sz[2],aB[2],gH[2]);
     fprintf(stderr,"%d: dsne[2]=%f  sraq[2]=%f  szaq[2]=%f  aBaq[2]=%f  gHaqq[2]=%f\n",mype,dsne[2],sraq[2],szaq[2],aBaq[2],gHaqq[2]);
#endif

  return(0);
}

/***************** FORTRAN-C Link Routine ****************************/

/* static int *sw; */
static REAL *r=NULL,*ra=NULL,*rq=NULL,*z=NULL,*za=NULL,*zq=NULL
,*B=NULL,*Ba=NULL,*Bq=NULL,*gh=NULL,*gha=NULL,*ghq=NULL,*ap=NULL,*qp=NULL;
static REAL *F=NULL,*Fa=NULL,*gFa=NULL,*gFaa=NULL,*gYa=NULL,*gYaa=NULL
,*T=NULL,*Ta=NULL,*P=NULL,*Pa=NULL;

static REAL *Ne=NULL,*dNe=NULL;
static REAL aTe=15.,aTi=15.,raT;

/* the arrays used by the esi c functions are initialized to be the same as those */
/* used by the calling function */
void esilink2c(REAL *XaF,REAL *XaFa	 
		,REAL *XgFa,REAL *XgFaa
		,REAL *XgYa,REAL *XgYaa
		,REAL *XT,REAL *XTa
		,REAL *XP,REAL *XPa
		,REAL *Xr,REAL *Xra,REAL *Xrq
		,REAL *Xz,REAL *Xza,REAL *Xzq
		,REAL *XB,REAL *XBa,REAL *XBq
		,REAL *Xgh,REAL *Xgha,REAL *Xghq
                )
	/*	,int *Xsw) */
{
  F	=XaF;
  Fa	=XaFa;
  gFa	=XgFa;
  gFaa	=XgFaa;
  gYa	=XgYa;
  gYaa	=XgYaa;
  T	=XT;
  Ta	=XTa;
  P	=XP;
  Pa	=XPa;
  r	=Xr;
  ra	=Xra;
  rq	=Xrq;
  z	=Xz;
  za	=Xza;
  zq	=Xzq;
  B	=XB;
  Ba	=XBa;
  Bq	=XBq;
  gh	=Xgh;
  gha	=Xgha;
  ghq	=Xghq;
/*  sw	=Xsw; */

  return;
}

void esilink2c4ne(double *Xne,double *Xdne)
{
  Ne	=Xne;
  dNe	=Xdne;
  return;
}

void settemperatures(double *Te,double *Ti)
{
  aTe	=*Te;
  aTi	=*Ti;
  raT	=(aTe+aTi) != 0. ? 30./(aTe+aTi) : 1.;
  return;
}

/***************** Main Reconstruction Routines **********************/
/* sF,sFa,sgFa,sgFaa,sgYa,sgYaa,sT,sTa,sP,sPa are the addresses to return point values */
/* a is a real, passed argument */
int esigetprofiles(REAL *sF,REAL *sFa,	 
		       /* \baF,\R{d\baF}{da}*/
		       REAL *sgFa,REAL *sgFaa,
		       /* \R{d\bgF}{ada},(\R{d\bgF}{ada})'_a */
		       REAL *sgYa,REAL *sgYaa,
		       /* \R{d\bgY}{ada},(\R{d\bgY}{ada})'_a */
		       REAL *sT,REAL *sTa,
		       /* T=\baF\R{d\baF}{d\bgY},\R{dT}{da} */
		       REAL *sP,REAL *sPa
		       /* P=\R{d\bsp}{d\bgY},\R{dP}{da} */
		       ,REAL *a)
{
  REAL x,X,xx,XX;

  A	=*a;
  if(A < 0.){
    A	=-A;
  }
  i0	=(A-sa[0])*rha;
  if(i0 >= Na){
    i0	=Na-1;
  }
  i1	=i0+1;

  x	=(A-sa[i0])*rha;
  X	=1.-x;

  XX	=X*X;
  xx	=x*x;
  XF	=XX*(3.-2.*X);
  XD	=XX*x*ha;
  xF	=xx*(3.-2.*x);
  xD	=-xx*X*ha;

  dX	=3.*x*X;
  dXD	=X-dX;
  dxD	=x-dX;
  dX	*=2.*rha;

  *sF	=XF*aF[i0]+xF*aF[i1]+XD*daF[i0]+xD*daF[i1];
  *sFa	=dX*(aF[i1]-aF[i0])+dXD*daF[i0]+dxD*daF[i1];

  *sgFa	=XF*dgF[i0]+xF*dgF[i1]+XD*d2gF[i0]+xD*d2gF[i1];
  *sgFaa=dX*(dgF[i1]-dgF[i0])+dXD*d2gF[i0]+dxD*d2gF[i1];
  *sgFaa=A*(*gFaa)+(*gFa);
  *sgYa	=XF*dgY[i0]+xF*dgY[i1]+XD*d2gY[i0]+xD*d2gY[i1];
  *sgYaa=dX*(dgY[i1]-dgY[i0])+dXD*d2gY[i0]+dxD*d2gY[i1];
  *sgYaa=A*(*gYaa)+(*gYa);

  if(A != 0.){
    *sgFa	*=A;
    *sgYa	*=A;
  }
  *sT	=XF*aT[i0]+xF*aT[i1]+XD*daT[i0]+xD*daT[i1];
  *sTa	=dX*(aT[i1]-aT[i0])+dXD*daT[i0]+dxD*daT[i1];

  *sP	=XF*aP[i0]+xF*aP[i1]+XD*daP[i0]+xD*daP[i1];
  *sPa	=dX*(aP[i1]-aP[i0])+dXD*daP[i0]+dxD*daP[i1];

  if(Ne != NULL){
    *Ne	=(XF*sne[i0]+xF*sne[i1]+XD*dsne[i0]+xD*dsne[i1])*raT;
    *dNe=(dX*(sne[i1]-sne[i0])+dXD*dsne[i0]+dxD*dsne[i1])*raT;
  }
  return(0);
}

int esiget2dfunctions0(REAL a,REAL q,int k)
{
  REAL x,X,XX;
  REAL f0,f1,fq0,fq1,r0;
 
  j01	=j00+1;
  j10	=j00+Np1;
  j11	=j10+1;
  
  x	=a*rha;
  X	=1.-x;
  
  XX	=X*X;
  XF	=XX*(3.-2.*X);
  XD	=XX;
  xF	=x*(3.-2.*x)*rha;
  xD	=-x*X;
  
  dX	=3.*x*X;
  dXD	=X-dX;
  dxD	=x-dX;
  dX	*=2.*rha;
  
  r0	=sr[0];
  f0	=xF*sr[j10]+XD*sra[j00]+xD*sra[j10];
  f1	=xF*sr[j11]+XD*sra[j01]+xD*sra[j11];
  fq0	=xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
  fq1	=xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
  r[k]=(YF+yF)*XF*sr[0]+a*(YF*f0+yF*f1+YD*fq0+yD*fq1);
  rq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
  f0	=dX*(sr [j10]-r0)+dXD*sra [j00]+dxD*sra [j10];
  f1	=dX*(sr [j11]-r0)+dXD*sra [j01]+dxD*sra [j11];
  fq0	=dX*(srq[j10]-srq[j00])+dXD*sraq[j00]+dxD*sraq[j10];
  fq1	=dX*(srq[j11]-srq[j01])+dXD*sraq[j01]+dxD*sraq[j11];
  ra[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
  
  f0	=xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
  f1	=xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
  fq0	=xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
  fq1	=xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
  zq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
  z[k]=(YF+yF)*XF*sz[0]+a*(YF*f0+yF*f1+YD*fq0+yD*fq1);
  f0	=dX*(sz [j10]-sz [j00])+dXD*sza [j00]+dxD*sza [j10];
  f1	=dX*(sz [j11]-sz [j01])+dXD*sza [j01]+dxD*sza [j11];
  fq0	=dX*(szq[j10]-szq[j00])+dXD*szaq[j00]+dxD*szaq[j10];
  fq1	=dX*(szq[j11]-szq[j01])+dXD*szaq[j01]+dxD*szaq[j11];
  za[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
  
  f0	=xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
  f1	=xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
  fq0	=xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
  fq1	=xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
  B[k]=(YF+yF)*XF*aB[0]+a*(YF*f0+yF*f1+YD*fq0+yD*fq1);
  Bq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
  f0	=dX*(aB [j10]-aB [j00])+dXD*aBa [j00]+dxD*aBa [j10];
  f1	=dX*(aB [j11]-aB [j01])+dXD*aBa [j01]+dxD*aBa [j11];
  fq0	=dX*(aBq[j10]-aBq[j00])+dXD*aBaq[j00]+dxD*aBaq[j10];
  fq1	=dX*(aBq[j11]-aBq[j01])+dXD*aBaq[j01]+dxD*aBaq[j11];
  Ba[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
  
  XD	*=a;
  xF	*=a;
  xD	*=a;
  
  F[k]	=XF*aF[i0]+xF*aF[i1]+XD*daF[i0]+xD*daF[i1];
  Fa[k]	=dX*(aF[i1]-aF[i0])+dXD*daF[i0]+dxD*daF[i1];
  gFa[k]	=XF*dgF[i0]+xF*dgF[i1]+XD*d2gF[i0]+xD*d2gF[i1];
  gFaa[k]	=dX*(dgF[i1]-dgF[i0])+dXD*d2gF[i0]+dxD*d2gF[i1];
  gFaa[k]	=a*(*gFaa)+(*gFa);

  gYa[k]	=XF*dgY[i0]+xF*dgY[i1]+XD*d2gY[i0]+xD*d2gY[i1];
  gYaa[k]	=dX*(dgY[i1]-dgY[i0])+dXD*d2gY[i0]+dxD*d2gY[i1];
  gYaa[k]	=a*(*gYaa)+(*gYa);

  T[k]	=XF*aT[i0]+xF*aT[i1]+XD*daT[i0]+xD*daT[i1];
  Ta[k]	=dX*(aT[i1]-aT[i0])+dXD*daT[i0]+dxD*daT[i1];
  P[k]	=XF*aP[i0]+xF*aP[i1]+XD*daP[i0]+xD*daP[i1];
  Pa[k]	=dX*(aP[i1]-aP[i0])+dXD*daP[i0]+dxD*daP[i1];
  if(Ne != NULL){
    *Ne	=(XF*sne[i0]+xF*sne[i1]+XD*dsne[i0]+xD*dsne[i1])*raT;
    *dNe=(dX*(sne[i1]-sne[i0])+dXD*dsne[i0]+dxD*dsne[i1])*raT;
  }
  return(0);
}

int esiget2dfunctions(REAL *a0,REAL *gq0,int *n)
{
  /* Exception: at a=0 the routine returns:
     dr_q/a, dz_q/a, dB_q/a, dgF/a, dgY/a
     rather than
     dr_q=0, dz_q=0, dB_q=0, dgF=0, dgY=0 */
  int j0,j1,k;
  REAL x,X,xx,XX;
  REAL f0,f1,fq0,fq1;
  REAL a,q;

  for(k =0; k < *n; k++){
    a	=a0[k];
    q	=gq0[k];
    
    j1	=q*crgq0;
    if(q < 0.){
      j1--;
    }
    q	-=cgq0*j1;
    j0	=q*rhq;
    if(j0 >= Np){
      j0	-=Np;
      q	-=cgq0;
    }
    if(j0 < 0){
      j0	+=Np;
      q	+=cgq0;
    }
    j1	=j0+1;
    y	=(q-gq[j0])*rhq;
    Y	=1.-y;
   
    YY	=Y*Y;
    yy	=y*y;
    YF	=YY*(3.-2.*Y);
    YD	=YY*y*hq;
    yF	=yy*(3.-2.*y);
    yD	=-yy*Y*hq;
    
    dY	=3.*y*Y;
    dYD	=Y-dY;
    dyD	=y-dY;
    dY	*=2.*rhq;
    if(a < 0.){
      a	=-a;
    }
    i0	=(a-sa[0])*rha;
    if(a != 0.){
      if(i0 >= Na){
	i0	=Na-1;
      }
      i1	=i0+1;
      
      x	=(a-sa[i0])*rha;
      X	=1.-x;

      XX	=X*X;
      xx	=x*x;
      XF	=XX*(3.-2.*X);
      XD	=XX*x*ha;
      xF	=xx*(3.-2.*x);
      xD	=-xx*X*ha;
      
      dX	=3.*x*X;
      dXD	=X-dX;
      dxD	=x-dX;
      dX	*=2.*rha;
      
      j00	=Np1*i0+j0;
      j01	=j00+1;
      j10	=j00+Np1;
      j11	=j10+1;
      
      /* r,r'_a,r'_q */
      f0	=XF*sr [j00]+xF*sr [j10]+XD*sra [j00]+xD*sra [j10];
      f1	=XF*sr [j01]+xF*sr [j11]+XD*sra [j01]+xD*sra [j11];
      fq0	=XF*srq[j00]+xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
      fq1	=XF*srq[j01]+xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
      r[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      rq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sr [j10]-sr [j00])+dXD*sra [j00]+dxD*sra [j10];
      f1	=dX*(sr [j11]-sr [j01])+dXD*sra [j01]+dxD*sra [j11];
      fq0	=dX*(srq[j10]-srq[j00])+dXD*sraq[j00]+dxD*sraq[j10];
      fq1	=dX*(srq[j11]-srq[j01])+dXD*sraq[j01]+dxD*sraq[j11];
      ra[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* z,z'_a,z'_q */
      f0	=XF*sz [j00]+xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
      f1	=XF*sz [j01]+xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
      fq0	=XF*szq[j00]+xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
      fq1	=XF*szq[j01]+xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
      z[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      zq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sz [j10]-sz [j00])+dXD*sza [j00]+dxD*sza [j10];
      f1	=dX*(sz [j11]-sz [j01])+dXD*sza [j01]+dxD*sza [j11];
      fq0	=dX*(szq[j10]-szq[j00])+dXD*szaq[j00]+dxD*szaq[j10];
      fq1	=dX*(szq[j11]-szq[j01])+dXD*szaq[j01]+dxD*szaq[j11];
      za[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* B,B'_a,B'_q */
      f0	=XF*aB [j00]+xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
      f1	=XF*aB [j01]+xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
      fq0	=XF*aBq[j00]+xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
      fq1	=XF*aBq[j01]+xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
      B[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      Bq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(aB [j10]-aB [j00])+dXD*aBa [j00]+dxD*aBa [j10];
      f1	=dX*(aB [j11]-aB [j01])+dXD*aBa [j01]+dxD*aBa [j11];
      fq0	=dX*(aBq[j10]-aBq[j00])+dXD*aBaq[j00]+dxD*aBaq[j10];
      fq1	=dX*(aBq[j11]-aBq[j01])+dXD*aBaq[j01]+dxD*aBaq[j11];
      Ba[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;

      /* gh,gh'_a,gh'_q */
      f0	=XF*gHq [j00]+xF*gHq [j10]+XD*gHaq [j00]+xD*gHaq [j10];
      f1	=XF*gHq [j01]+xF*gHq [j11]+XD*gHaq [j01]+xD*gHaq [j11];
      fq0	=XF*gHqq[j00]+xF*gHqq[j10]+XD*gHaqq[j00]+xD*gHaqq[j10];
      fq1	=XF*gHqq[j01]+xF*gHqq[j11]+XD*gHaqq[j01]+xD*gHaqq[j11];
      ghq[k]	=YF*f0+yF*f1+YD*fq0+yD*fq1;
      gh[k]	=XF*gH[j00]+xF*gH[j10]+XD*gHa[j00]+xD*gHa[j10]+
	hq*0.5*((1.+YY*(YY-2.*Y))*f0+yy*(2.*y-yy)*f1
		+cr6*hq*((1.-YY*(YY+4.*Y*y))*fq0-yy*(4.*y*Y+yy)*fq1));
      f0	=dX*(gHq [j10]-gHq [j00])+dXD*gHaq [j00]+dxD*gHaq [j10];
      f1	=dX*(gHq [j11]-gHq [j01])+dXD*gHaq [j01]+dxD*gHaq [j11];
      fq0	=dX*(gHqq[j10]-gHqq[j00])+dXD*gHaqq[j00]+dxD*gHaqq[j10];
      fq1	=dX*(gHqq[j11]-gHqq[j01])+dXD*gHaqq[j01]+dxD*gHaqq[j11];
      gha[k]	=dX*(gH[j10]-gH[j00])+dXD*gHa[j00]+dxD*gHa[j10]+
	hq*0.5*((1.+YY*(YY-2.*Y))*f0+yy*(2.*y-yy)*f1
		+cr6*hq*((1.-YY*(YY+4.*Y*y))*fq0-yy*(4.*y*Y+yy)*fq1));

      F[k]	=XF*aF[i0]+xF*aF[i1]+XD*daF[i0]+xD*daF[i1];
      Fa[k]	=dX*(aF[i1]-aF[i0])+dXD*daF[i0]+dxD*daF[i1];
      gFa[k]	=XF*dgF[i0]+xF*dgF[i1]+XD*d2gF[i0]+xD*d2gF[i1];
      gFaa[k]	=dX*(dgF[i1]-dgF[i0])+dXD*d2gF[i0]+dxD*d2gF[i1];
      gFaa[k]	=a*(*gFaa)+(*gFa);
      gYa[k]	=XF*dgY[i0]+xF*dgY[i1]+XD*d2gY[i0]+xD*d2gY[i1];
      gYaa[k]	=dX*(dgY[i1]-dgY[i0])+dXD*d2gY[i0]+dxD*d2gY[i1];
      gYaa[k]	=a*(*gYaa)+(*gYa);
      T[k]	=XF*aT[i0]+xF*aT[i1]+XD*daT[i0]+xD*daT[i1];
      Ta[k]	=dX*(aT[i1]-aT[i0])+dXD*daT[i0]+dxD*daT[i1];
      P[k]	=XF*aP[i0]+xF*aP[i1]+XD*daP[i0]+xD*daP[i1];
      Pa[k]	=dX*(aP[i1]-aP[i0])+dXD*daP[i0]+dxD*daP[i1];
      if(Ne != NULL){
	Ne[k]	=(XF*sne[i0]+xF*sne[i1]+XD*dsne[i0]+xD*dsne[i1])*raT;
	dNe[k]	=(dX*(sne[i1]-sne[i0])+dXD*dsne[i0]+dxD*dsne[i1])*raT;
      }
      gFa[k]	*=a;
      gYa[k]	*=a;
    }
    else{	/* a == 0. */
      r[k]	=sr[0];
      ra[k]	=YF*sra[j0]+yF*sra[j1]+YD*sraq[j0]+yD*sraq[j1];
      rq[k]	=dY*(sra[j1]-sra[j0])+dYD*sraq[j0]+dyD*sraq[j1];
      z[k]	=sz[0];
      za[k]	=YF*sza[j0]+yF*sza[j1]+YD*szaq[j0]+yD*szaq[j1];
      zq[k]	=dY*(sza[j1]-sza[j0])+dYD*szaq[j0]+dyD*szaq[j1];
      B[k]	=aB[0];
      Ba[k]	=YF*aBa[j0]+yF*aBa[j1]+YD*aBaq[j0]+yD*aBaq[j1];
      Bq[k]	=dY*(aBa[j1]-aBa[j0])+dYD*aBaq[j0]+dyD*aBaq[j1];
      ghq[k]	=YF*gHaq[j0]+yF*gHaq[j1]+YD*gHaqq[j0]+yD*gHaqq[j1];
      gh[k]	=0.;
      gha[k]	=gHa[j0]+hq*0.5*((1.+YY*(YY-2.*Y))*gHaq[j0]
				 +yy*(2.*y-yy)*gHaq[j1]
				 +cr6*hq*((1.-YY*(YY+4.*Y*y))*gHaqq[j0]
					   -yy*(4.*y*Y+yy)*gHaqq[j1]));
      F[k]	=aF[0];
      Fa[k]	=0.;
      gFa[k]	=dgF[0];
      gFaa[k]	=dgF[0];
      gYa[k]	=dgY[0];
      gYaa[k]	=dgY[0];
      T[k]	=aT[0];
      Ta[k]	=0.;
      P[k]	=aP[0];
      Pa[k]	=0.;
      if(Ne != NULL){
	Ne[k]	=sne[0];
	dNe[k]	=0.;
      }
    }
  }
  return(0);
}
/* R,Z,Bm are pointers to store array data */
/* a0 and q0 are pointers to toroidal flux array (radial coord) and angular coord, needed */
/* n is the number of elements in a0 and q0 */
int esigetrzb(REAL *R, REAL *Z, REAL *Bm,REAL *a0,REAL *q0,int *n)
{
  int j0,j1,k;
  REAL x,X,xx,XX;
  REAL y,Y,yy,YY;
  REAL f0,f1,fq0,fq1;
  REAL a,q;

  for(k =0; k < *n; k++){
    a	=a0[k];
    q	=q0[k];
    
    if(a < 0.){
      a	=-a;
    }
    i0	=(a-sa[0])*rha;
    if(i0 >= Na){
      i0	=Na-1;
    }
    i1	=i0+1;
    
    x	=(a-sa[i0])*rha;
    X	=1.-x;
    
    XX	=X*X;
    xx	=x*x;
    XF	=XX*(3.-2.*X);
    XD	=XX*x*ha;
    xF	=xx*(3.-2.*x);
    xD	=-xx*X*ha;
    
    dX	=3.*x*X;
    dXD	=X-dX;
    dxD	=x-dX;
    dX	*=2.*rha;
    
    j1	=q*crgq0;
    if(q < 0.){
      j1--;
    }
    q	-=cgq0*j1;
    j0	=q*rhq;
    if(j0 >= Np){
      j0	-=Np;
      q	-=cgq0;
    }
    if(j0 < 0){
      j0	+=Np;
      q	+=cgq0;
    }
    j1	=j0+1;
    
    Y	=(gq[j1]-q)*rhq;
    y	=1.-Y;
    
    YY	=Y*Y;
    yy	=y*y;
    YF	=YY*(3.-2.*Y);
    YD	=YY*y*hq;
    yF	=yy*(3.-2.*y);
    yD	=-yy*Y*hq;
    
    dY	=3.*y*Y;
    dYD	=Y-dY;
    dyD	=y-dY;
    dY	*=2.*rhq;
    
    j00	=Np1*i0+j0;
    j01	=j00+1;
    j10	=j00+Np1;
    j11	=j10+1;
    
    f0	=XF*sr [j00]+xF*sr [j10]+XD*sra [j00]+xD*sra [j10];
    f1	=XF*sr [j01]+xF*sr [j11]+XD*sra [j01]+xD*sra [j11];
    fq0	=XF*srq[j00]+xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
    fq1	=XF*srq[j01]+xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
    R[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
    
    f0	=XF*sz [j00]+xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
    f1	=XF*sz [j01]+xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
    fq0	=XF*szq[j00]+xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
    fq1	=XF*szq[j01]+xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
    Z[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
    
    f0	=XF*aB [j00]+xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
    f1	=XF*aB [j01]+xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
    fq0	=XF*aBq[j00]+xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
    fq1	=XF*aBq[j01]+xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
    Bm[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
  }
  return(0);
}


int IsOut(double rU,double zU)
{
  int j;
  double q,R1,R2;

  zU =(zU-Zbnd0)/sb;
  if(zU > 1. || zU < -1.)  return(1);
  q	=-asin(zU);
  if(zU < 0.){
    R1	=Rbnd[Np/4];
    if(rU < R1) q=cgp-q;
  }else{
    R1	=Rbnd[3*Np/4]; 
    q =rU > R1 ? c2gp+q : cgp-q;
  }
  j	=q*rhq;
  qHermit(&R2,NULL,q-gq[j],Rbnd+j,Rbnd1+j);
  if(rU > R1 && rU > R2 || rU < R1 && rU < R2) return(1);
  return(0);
}

int ESIrz2agq(double *aU, double *qU,double *rU,double *zU,int *ierr,int n)
{
  int i,i0,ji,iter,nerr;
  int j0,j1,k;
  double x,X,xx,XX,err=1e-9;
  double y,Y,yy,YY;
  double f0,f1,fq0,fq1;
  double a,q,r,z,ra,za,rq,zq;
  double D,da,dq,dr,dz,R,Z;

  if(FlMeshRZ){
    printf("Error: r-z mesh was not created\n");
    return(1);
  }
  nerr=0;
  for(i=0; i < n; i++){
    if( 1 /*(ierr[i]=IsOut(rU[i],zU[i])) == 0*/){
      R	=rU[i];
      Z	=zU[i];
      if(Z == zLab1){
	if(fabs(R-rL1[0]) > err){
	  ierr[i]=1;
	  nerr++;
	}
	continue;
      } 
      i0=(Z-zLab1)*rdzLab;
      if(i0 == Nz){
	if(fabs(R-rL1[i0]) > err){
	  ierr[i]=1;
	  nerr++;
	}
	continue;
      } 
      a	=(R-rL1[i0])/(rL2[i0]-rL1[i0]);
      j0	=i0 ? a*Nr : 0;

      if(j0 < 0) j0=0;
      if(j0 > Nr) j0=Nr;
 
      ji	=i0*Nr1+j0;
      a	=aLab[ji];
      q	=qLab[ji];
      iter=0;
      do{
	j0	=q*rhq;
	j1	=j0+1;
	y	=(q-gq[j0])*rhq;
	Y	=1.-y;
   
	YY	=Y*Y;
	yy	=y*y;
	YF	=YY*(3.-2.*Y);
	YD	=YY*y*hq;
	yF	=yy*(3.-2.*y);
	yD	=-yy*Y*hq;
    
	dY	=3.*y*Y;
	dYD	=Y-dY;
	dyD	=y-dY;
	dY	*=2.*rhq;
	if(a < 0.) a=-a;
	i0	=(a-sa[0])*rha;
	if(i0 >= Na) i0--;
	if(a != 0.){
	  x	=(a-sa[i0])*rha;
	  X	=1.-x;

	  XX	=X*X;
	  xx	=x*x;
	  XF	=XX*(3.-2.*X);
	  XD	=XX*x*ha;
	  xF	=xx*(3.-2.*x);
	  xD	=-xx*X*ha;
      
	  dX	=3.*x*X;
	  dXD	=X-dX;
	  dxD	=x-dX;
	  dX	*=2.*rha;
      
	  j00	=Np1*i0+j0;
	  j01	=j00+1;
	  j10	=j00+Np1;
	  j11	=j10+1;

	  f0	=XF*sr [j00]+xF*sr [j10]+XD*sra [j00]+xD*sra [j10];
	  f1	=XF*sr [j01]+xF*sr [j11]+XD*sra [j01]+xD*sra [j11];
	  fq0	=XF*srq[j00]+xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
	  fq1	=XF*srq[j01]+xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
	  r		=YF*f0+yF*f1+YD*fq0+yD*fq1;
	  rq	=dY*(f1-f0)+dYD*fq0+dyD*fq1;
	  f0	=dX*(sr [j10]-sr [j00])+dXD*sra [j00]+dxD*sra [j10];
	  f1	=dX*(sr [j11]-sr [j01])+dXD*sra [j01]+dxD*sra [j11];
	  fq0	=dX*(srq[j10]-srq[j00])+dXD*sraq[j00]+dxD*sraq[j10];
	  fq1	=dX*(srq[j11]-srq[j01])+dXD*sraq[j01]+dxD*sraq[j11];
	  ra	=YF*f0+yF*f1+YD*fq0+yD*fq1;

	  f0	=XF*sz [j00]+xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
	  f1	=XF*sz [j01]+xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
	  fq0	=XF*szq[j00]+xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
	  fq1	=XF*szq[j01]+xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
	  z		=YF*f0+yF*f1+YD*fq0+yD*fq1;
	  zq	=dY*(f1-f0)+dYD*fq0+dyD*fq1;
	  f0	=dX*(sz [j10]-sz [j00])+dXD*sza [j00]+dxD*sza [j10];
	  f1	=dX*(sz [j11]-sz [j01])+dXD*sza [j01]+dxD*sza [j11];
	  fq0	=dX*(szq[j10]-szq[j00])+dXD*szaq[j00]+dxD*szaq[j10];
	  fq1	=dX*(szq[j11]-szq[j01])+dXD*szaq[j01]+dxD*szaq[j1];
	  za	=YF*f0+yF*f1+YD*fq0+yD*fq1;
	}
	else{	/* a == 0. */
	  r	=sr[0];
	  ra	=YF*sra[j0]+yF*sra[j1]+YD*sraq[j0]+yD*sraq[j1];
	  rq	=dY*(sra[j1]-sra[j0])+dYD*sraq[j0]+dyD*sraq[j1];
	  z	=sz[0];
	  za	=YF*sza[j0]+yF*sza[j1]+YD*szaq[j0]+yD*szaq[j1];
	  zq	=dY*(sza[j1]-sza[j0])+dYD*szaq[j0]+dyD*szaq[j1];
	}
	D	=1./(za*rq-zq*ra);
	dr	=R-r;
	dz	=Z-z;
	da	=(rq*dz-zq*dr)*D;
	dq	=(za*dr-ra*dz)*D;
	a	+=da;
	q	+=dq;
	while(q < 0.) q+=c2gp; 
	while(q > c2gp) q-=c2gp; 
	if(a < 0.){
	  a=-a;
	  q=q < cgp ? cgp+q : q-cgp;
	}
	if(a > 1.) a=1.;
	iter++;
      }while(iter < 10 && fabs(da)+fabs(dq) > err);
      if(iter > 9){
	ierr[i]=2;
	nerr++;
      }
      aU[i]	=a;
      qU[i]	=q;
    }
  }
  return(nerr);
}

int esirz2agq_(double *aU, double *qU,double *rU,double *zU,int *ierr,int *n)
{
  return(ESIrz2agq(aU,qU,rU,zU,ierr,*n));
}

/* #include "omp.h" */

int gcmotion(REAL *dXgr,
             REAL *dXa,
             REAL *dXgq,
             REAL *dXgf,
             REAL *particle_array,
             REAL *wpi_array,
             REAL *Er_particle,
             REAL *nonlinear,
             int *n)
/*
  Xgr = particle_array[k][3] = zion(4,k+1) (Fortran side)
  Xa  = particle_array[k][0] = zion(1,k+1) (Fortran side)
  Xgq = particle_array[k][1] = zion(2,k+1) (Fortran side)
  Xgm = particle_array[k][5]^2 = zion(6,k+1)**6 (Fortran side)

  zion is dimensioned zion(7,mimax) on the Fortran side. Since we view this
  array as a 1-dimensional vector on the C side through the pointer
  "particle_array", we need to use pointer arithmetic:
      particle_array[k][3] = *(particle_array + k*7 + 3)

  Same trick with the wpi array which is dimensioned wpi(3,mi) on Fortran side
   wpi1 = wpi_array[k][0] = *(wpi_array+3*k)   = wpi(1,k+1)
   wpi2 = wpi_array[k][1] = *(wpi_array+3*k+1) = wpi(2,k+1)
   wpi2 = wpi_array[k][2] = *(wpi_array+3*k+2) = wpi(3,k+1)
*/
{
  int j0,j1,k;
  REAL r,ra,rq,za,zq,BB,Ba,Bq,Bf,F,gYa,gFa,T,P;
  REAL a,q,a11,a12,a21,a22;

  REAL x,X,xx,XX;
  REAL f0,f1,fq0,fq1;
  REAL y,Y,yy,YY;
  REAL D,Dhh;
  REAL Xgr,Xa,Xgq,Xgm,wpi1,wpi2,wpi3;

  int myid;
  double tt0,tt1;
  double MPI_Wtime(void);

/*  tt0 = MPI_Wtime(); */
/* ADDITIONAL OUTPUT: wpi3=dphi/dzeta;  */
/*                    B[k]=B;                       */
/*                    ghq[k]=da/dt|_EXB             */
/*                    gha[k]=1/|grad a|             */
#pragma omp parallel for private(k,a,q,j1,j0,y,Y,YY,yy,x,X,xx,XX,D,Dhh,r,ra,rq,za,zq,BB,Ba,Bq,Bf,F,gYa,gFa,T,P, a11,a12,a21,a22,f0,f1,fq0,fq1,YF,YD,yF,yD,dY,dYD,dyD,i0,i1,XF,XD,xF,xD,dX,dXD,dxD,j00,j01,j10,j11,Xa,Xgq,Xgr,Xgm,wpi1,wpi2,wpi3)
  for(k=0; k < *n; k++){
    Xa  = *(particle_array+k*7);    /* particle_array[k][0] */
    Xgq = *(particle_array+k*7+1);  /* particle_array[k][1] */
    Xgr = *(particle_array+k*7+3);  /* particle_array[k][3] */
    Xgm = *(particle_array+k*7+5);  /* particle_array[k][5] */
    Xgm = Xgm*Xgm;      /* mu = zion(6,k+1)**2 on Fortran side */
    wpi1 = *(wpi_array+3*k);      /* wpi[k][0] */
    wpi2 = *(wpi_array+3*k+1);    /* wpi[k][1] */
    wpi3 = *(wpi_array+3*k+2);    /* wpi[k][2] */
    a	=Xa;
    if(a < 1e-100){
      a	=0.;
    }
    q	=Xgq;

    j1	=q*crgq0;
    if(q < 0.){
      j1--;
    }
    q	-=cgq0*j1;
    j0	=q*rhq;
    if(j0 >= Np){
      j0	-=Np;
      q	-=cgq0;
    }
    if(j0 < 0){
      j0	+=Np;
      q	+=cgq0;
    }
    j1	=j0+1;
    y	=(q-gq[j0])*rhq;

    Y	=1.-y;
    YY	=Y*Y;
    yy	=y*y;
    YF	=YY*(3.-2.*Y);
    YD	=YY*y*hq;
    yF	=yy*(3.-2.*y);
    yD	=-yy*Y*hq;
    dY	=3.*y*Y;
    dYD	=Y-dY;
    dyD	=y-dY;
    dY	*=2.*rhq;

    if(a < 0.){
      a	=-a;
    }
    i0	=(a-sa[0])*rha;
    if(a != 0.){
      if(i0 >= Na){
	i0	=Na-1;
      }
      i1	=i0+1;
      x		=(a-sa[i0])*rha;
      X		=1.-x;
      XX	=X*X;
      xx	=x*x;
      XF	=XX*(3.-2.*X);
      XD	=XX*x*ha;
      xF	=xx*(3.-2.*x);
      xD	=-xx*X*ha;
      
      dX	=3.*x*X;
      dXD	=X-dX;
      dxD	=x-dX;
      dX	*=2.*rha;
      
      j00	=Np1*i0+j0;
      j01	=j00+1;
      j10	=j00+Np1;
      j11	=j10+1;
      
      /* r,r'_a,r'_q */
      f0	=XF*sr [j00]+xF*sr [j10]+XD*sra [j00]+xD*sra [j10];
      f1	=XF*sr [j01]+xF*sr [j11]+XD*sra [j01]+xD*sra [j11];
      fq0	=XF*srq[j00]+xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
      fq1	=XF*srq[j01]+xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
      r		=YF*f0+yF*f1+YD*fq0+yD*fq1;
      rq	=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sr [j10]-sr [j00])+dXD*sra [j00]+dxD*sra [j10];
      f1	=dX*(sr [j11]-sr [j01])+dXD*sra [j01]+dxD*sra [j11];
      fq0	=dX*(srq[j10]-srq[j00])+dXD*sraq[j00]+dxD*sraq[j10];
      fq1	=dX*(srq[j11]-srq[j01])+dXD*sraq[j01]+dxD*sraq[j11];
      ra	=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* z,z'_a,z'_q */
      f0	=XF*sz [j00]+xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
      f1	=XF*sz [j01]+xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
      fq0	=XF*szq[j00]+xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
      fq1	=XF*szq[j01]+xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
      zq	=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sz [j10]-sz [j00])+dXD*sza [j00]+dxD*sza [j10];
      f1	=dX*(sz [j11]-sz [j01])+dXD*sza [j01]+dxD*sza [j11];
      fq0	=dX*(szq[j10]-szq[j00])+dXD*szaq[j00]+dxD*szaq[j10];
      fq1	=dX*(szq[j11]-szq[j01])+dXD*szaq[j01]+dxD*szaq[j11];
      za	=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* B,B'_a,B'_q */
      f0	=XF*aB [j00]+xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
      f1	=XF*aB [j01]+xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
      fq0	=XF*aBq[j00]+xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
      fq1	=XF*aBq[j01]+xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
      BB	=YF*f0+yF*f1+YD*fq0+yD*fq1;
      Bq	=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(aB [j10]-aB [j00])+dXD*aBa [j00]+dxD*aBa [j10];
      f1	=dX*(aB [j11]-aB [j01])+dXD*aBa [j01]+dxD*aBa [j11];
      fq0	=dX*(aBq[j10]-aBq[j00])+dXD*aBaq[j00]+dxD*aBaq[j10];
      fq1	=dX*(aBq[j11]-aBq[j01])+dXD*aBaq[j01]+dxD*aBaq[j11];
      Ba	=YF*f0+yF*f1+YD*fq0+yD*fq1;

      F		=XF*aF[i0]+xF*aF[i1]+XD*daF[i0]+xD*daF[i1];
      gFa	=XF*dgF[i0]+xF*dgF[i1]+XD*d2gF[i0]+xD*d2gF[i1];
      gYa	=XF*dgY[i0]+xF*dgY[i1]+XD*d2gY[i0]+xD*d2gY[i1];
      T		=XF*aT[i0]+xF*aT[i1]+XD*daT[i0]+xD*daT[i1];
      P		=XF*aP[i0]+xF*aP[i1]+XD*daP[i0]+xD*daP[i1];
      gFa	*=a;
      gYa	*=a;
    }
    else{	/* a == 0. */
      r		=sr[0];
      ra	=YF*sra[j0]+yF*sra[j1]+YD*sraq[j0]+yD*sraq[j1];
      rq	=dY*(sra[j1]-sra[j0])+dYD*sraq[j0]+dyD*sraq[j1];
      za	=YF*sza[j0]+yF*sza[j1]+YD*szaq[j0]+yD*szaq[j1];
      zq	=dY*(sza[j1]-sza[j0])+dYD*szaq[j0]+dyD*szaq[j1];
      BB	=aB[0];
      Ba	=YF*aBa[j0]+yF*aBa[j1]+YD*aBaq[j0]+yD*aBaq[j1];
      Bq	=dY*(aBa[j1]-aBa[j0])+dYD*aBaq[j0]+dyD*aBaq[j1];
      F		=aF[0];
      gFa	=dgF[0];
      gYa	=dgY[0];
      T		=aT[0];
      P		=aP[0];
      
      T		*=Xgr;
      D		=za*rq-zq*ra;
      a21	=1./(D*((F+T)/r+Xgr*r*P));
#ifdef H
      r  =cgq0*Xgr*BB;
      BB*=r;
      r	=r*Xgr+cgq0*Xgm;
#endif
      r  =Xgr*BB;
      BB	*=r;
      r	=r*Xgr+Xgm;

      Ba	*=r;
      Bq	*=-r;
      dXgr[k]	=0.;
      Bq	*=a21;
      Ba	*=a21;
      dXgf[k]	=BB/F;
      T	=cos(Xgq);
      P	=sin(Xgq);
      dXa[k]	=Bq*T-Ba*P;
      dXgq[k]	=Bq*P+Ba*T;
    }
    B[k]=BB;    /* output of B field */
    a21	=F;
    T	*=Xgr;
    D	=za*rq-zq*ra;
    Dhh =rq*rq+zq*zq;
    gha[k]=sqrt(D*D/Dhh);    /*  1/|grad_a|  */ 
    a12	=D*((F+T)/r+Xgr*r*P);
    D	=-gYa/(r*D);
/*    a11	=(rq*rq+zq*zq)*D;   */
    a11	=Dhh*D;
    D	*=ra*rq+za*zq;
    a22	=gYa*(1.+T/F);
    r	=1./(a11*a22-a12*a21);
    a11	*=r;
    a12	*=r;
    a21	*=r;
    a22	*=r;
    D	*=r;
/*    qq	=-gFa/gYa;   */
    wpi3=wpi3-wpi2*(-gYa/gFa);      /* =dphi/dzeta */
#ifdef H
    r  =Xgr*BB*cgq0;
    BB	*=r;
    r	=r*Xgr+Xgm*cgq0;
#endif
    r  =Xgr*BB;
    BB	*=r;         /* =dH/drho//  */
    r	=r*Xgr+Xgm;
    Ba	*=r;
/*  add EXB drift with only radial electric field; Er_particle=d\phi/da  */
/*    Ba  =Ba+Er_particle[k];  */
    Bq	*=-r;
/*  add EXB drift  */
    Ba	 = *nonlinear*wpi1+Ba;         /* =dH/da      */
    Ba   =Ba+Er_particle[k];           /* =dH/da, adding E_r0XB drift */   
    Bq	 =-*nonlinear*wpi2+Bq;         /* =-dH/dtheta */
    Bf	 =-*nonlinear*wpi3;            /* =-dH/dphi   */
    dXgr[k]	=a22*Bq-a12*Bf;
    dXgf[k]	=-a12*BB+a11*Ba+D*Bq;
    dXa[k]	=-a21*Bq+a11*Bf;
    dXgq[k]	=a22*BB-a21*Ba-D*Bf;
    ghq[k]	=-a21*(-wpi2)+a11*(-wpi3);     /* da/dt|_EXB */
    *(wpi_array+3*k+2) = wpi3;   /* write back the new value of wpi3 */
/* Not used for now   S.Ethier 5/7/08
    if(sw != NULL && sw[k]){
      Bq	=a*dXgq[k];
      Ba	=dXa[k];
      T	=cos(Xgq);
      P	=sin(Xgq);
      dXa[k]	=Ba*T-Bq*P;
      dXgq[k]	=Ba*P+Bq*T;
    }
*/
  }
/*
  tt1 = MPI_Wtime();
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid == 2) printf("in gcmotion: wall = %f\n",(tt1 - tt0));
*/
  
  return(0);
}


void esigetpressure(double *p,double *dp,double *a, int *n)
{
  int k,i;
  double p0[Na1],dp0,dp1,d2p0,d2p1,x,X,p3x,p3X,XF,XD,xF,xD;

  p0[0]	=0.;
  dp1	=0.;
  d2p1	=aP[0]*dgY[0];
  for(i=1,k=0; i < Na1; i++,k++){
    dp0	=dp1;
    d2p0=d2p1;
    dp1	=aP[i]*dgY[i]*sa[i];
    d2p1=aP[i]*dgY[i]+(daP[i]*dgY[i]+aP[i]*d2gY[i])*sa[i];
    p0[i]=p0[k]+0.5*ha*((dp0+dp1)+ha*(d2p0-d2p1)/6.);
  }
  for(i=0; i < Na1; i++){
    p0[i]	-=p0[Na];
  }
  for(k=0; k < *n; k++){
    i	=(a[k]-sa[0])*rha;
    if(i >= Na){
      i	=Na-1;
    }
    x	=(a[k]-sa[i])/ha;
    X	=1.-x;
    XF	=X*X*(3.-2.*X);
    XD	=X*X*x*ha;
    xF	=x*x*(3.-2.*x);
    xD	=-x*x*X*ha;
    p3x	=x*x*x;
    p3X	=X*X*X;
    p[k]=p0[i]+0.5*ha*((1.+p3X*(X-2.))*aP[i]+p3x*(2.-x)*aP[i+1]
		       +ha*((1.-p3X*(X+4.*x))*daP[i]
			    -p3x*(4.*X+x)*daP[i+1])/6.);
    dp[k]	=(XF*aP[i]+xF*aP[i+1]+XD*daP[i]+xD*daP[i+1])*
      (XF*dgY[i]+xF*dgY[i+1]+XD*d2gY[i]+xD*d2gY[i+1])*a[k];
  }
  return;
}

static double cme=9.1094e-28; /* [g] - electron mass */
static double cmi=1.6726e-24; /* [g] - proton mass */
static double cc=2.9979e+10; /* [cm/sec] - speed of light */
static double ce=4.8032e-10; /* [CGS] - proton electric charge */
static double cEeV=1.6022e-12; /* erg energy of 1 eV */
static double cEkeV=1.6022e-9; /* erg energy of 1 keV */

static char tga,tgb;
static double sega,smga,gmga,Zga,pZei,segb,smgb,gmgb,Zgb;

void esigetdensity(double *ne,double *dne,double *Te,double *Ti,double *a)
{
  int k,i;
  double s,x,X,XF,XD,xF,xD;

  i	=(*a-sa[0])*rha;
  if(i >= Na){
    i	=Na-1;
  }
  x	=(*a-sa[i])/ha;
  X	=1.-x;
  XF	=X*X*(3.-2.*X);
  XD	=X*X*x*ha;
  xF	=x*x*(3.-2.*x);
  xD	=-x*x*X*ha;
  s	=(*Te)+(*Ti) != 0. ? 30./((*Te)+(*Ti)) : 1.;
  *ne	=(XF*sne[i]+xF*sne[i+1]+XD*dsne[i]+xD*dsne[i+1])*s;
  if(dne != NULL){
    double dX,dXD,dxD;
    dX	=3.*x*X;
    dXD	=X-dX;
    dxD	=x-dX;
    dX	*=2.*rha;
    *dne	=(dX*(sne[i+1]-sne[i])+dXD*dsne[i]+dxD*dsne[i+1])*s;
  }
  return;
}

void GetEECollisions(double *gn,double *E,double *Te,double *ne)
{
  int i;
  double gl,t;
  double pva,pvb,v,x,gy;

  pva	=2.*cEkeV*(*E)/smga;
  pvb	=2.*cEkeV*(*Te)/smgb;

  x	=pva/pvb;
  v	=sqrt(x);
  if(v >= tF1){
    gy	=1.-cr2/x;
  }
  else{
    t	=v*rhF;
    i	=t;
    t	-=(double)i;
    gy	=(x-cr2)*(F0[i]+t*(F1[i]+t*(F2[i]+t*F3[i])));
  }
  gy	+=c2rqgp*v*exp(-x);

  gl	=*Te > 0.01 ? 24.-log(sqrt((*ne)*1e+8)/(*Te)) :
    23.-0.5*log((*ne)*1e+5/((*Te)*(*Te)*(*Te)));
  t	=sega*segb/smga;
  *gn	=8.*cgp*t*t*gl*(*ne)*1e+14*gy/(pva*sqrt(pva));
  return;
}

void GetIECollisions(double *gn,double *E,double *Te,double *ne)
{
  int i;
  double gl,t;
  double pva,pvb,v,x,gy;

  pva	=2.*cEkeV*(*E)/smga;
  pvb	=2.*cEkeV*(*Te)/smgb;

  x	=pva/pvb;
  v	=sqrt(x);
  if(v >= tF1){
    gy	=1.-cr2/x;
  }
  else{
    t	=v*rhF;
    i	=t;
    t	-=(double)i;
    gy	=(x-cr2)*(F0[i]+t*(F1[i]+t*(F2[i]+t*F3[i])));
  }
  gy	+=c2rqgp*v*exp(-x);

  if(*Te < 0.01*pZei){
    gl	=23.-0.5*log(pZei*(*ne)*1e+5/((*Te)*(*Te)*(*Te)));
  }
  else{
    gl	=24.-log(sqrt((*ne)*1e+8)/(*Te));
  }
  t	=sega*segb/smga;
  *gn	=8.*cgp*t*t*gl*(*ne)*1e+14*gy/(pva*sqrt(pva));
  return;
}

void GetIICollisions(double *gn,double *E,double *Ti,double *ne)
{
  int i;
  double gl,t,Tj;
  double pva,pvb,v,x,gy;
  
  Tj	=2.*(*E)/3.;
  pva	=2.*cEkeV*(*E)/smga;
  pvb	=2.*cEkeV*(*Ti)/smgb;

  x	=pva/pvb;
  v	=sqrt(x);
  if(v >= tF1){
    gy	=1.-cr2/x;
  }
  else{
    t	=v*rhF;
    i	=t;
    t	-=(double)i;
    gy	=(x-cr2)*(F0[i]+t*(F1[i]+t*(F2[i]+t*F3[i])));
  }
  gy	+=c2rqgp*v*exp(-x);
  gl	=23.-log(Zga*Zgb*Zgb*(gmga+gmgb)/(gmga*Tj+gmgb*(*Ti))
		 *sqrt((*ne)*1e+5/(*Ti)));
  t	=sega*segb/smga;
  *gn	=8.*cgp*t*t*gl*(*ne)*1e+14*gy/(pva*sqrt(pva));
  return;
}

void getcollisions(
		       double *gn	/* Collision grequency */
		       ,double *E	/* Test particle Energy [keV] */
		       ,double *Tp	/* background temperature [keV] */
		       ,double *n	/* background density [10^{20}/m^3] */
		       )
{
  switch(tga){
  case 'e':	/* electrons */
    switch(tgb){
    case 'e':			/* electrons */
      GetEECollisions(gn,E,Tp,n);
      break;
    case 'P':			/* Proton */
    case 'D':			/* Deuteron */
    case 'T':			/* Triton */
    case 'A':			/* Alpha */
      pZei		=Zgb*Zgb;
      GetIECollisions(gn,E,Tp,n);
      break;
    }
    break;
  case 'P':	/* Proton */
  case 'D':	/* Deuteron */
  case 'T':	/* Triton */
  case 'A':			/* Alpha */
    switch(tgb){
    case 'e':			/* electrons */
      pZei		=Zga*Zga;
      GetIECollisions(gn,E,Tp,n);
      break;
    case 'P':			/* Proton */
    case 'D':			/* Deuteron */
    case 'T':			/* Triton */
    case 'A':			/* Alpha */
      GetIICollisions(gn,E,Tp,n);
      break;
    }
    break;
  }
  return;
}


int setparticletype(
		     char *ega /* Type of test particles:
				  e   -electrons;
				  H,P -protons;
				  D   -deutrons;
				  T   -tritons;
				  A   -alphas.
				  */
		     ,char *egb /* Type of test particles:
				   e   -electrons;
				   H,P -protons;
				   D   -deutrons;
				   T   -tritons;
				   A   -alphas.
				   */
		     )
{
  tga	=*ega;
  tgb	=*egb;
  switch(tga){
  case 'E':	/* electrons */
    tga	='e';
    break;
  case 'H':	/* Proton */
  case 'p':	/* Proton */
  case 'h':	/* Proton */
    tga	='P';
    break;
  case 'd':	/* Deuteron */
    tga	='D';
    break;
  case 't':	/* Triton */
    tga	='T';
    break;
  case 'a':	/* Alpha */
    tga	='A';
    break;
  }
  switch(tgb){
  case 'E':	/* electrons */
    tgb	='e';
    break;
  case 'H':	/* Proton */
  case 'p':	/* Proton */
  case 'h':	/* Proton */
    tgb	='P';
    break;
  case 'd':	/* Deuteron */
    tgb	='D';
    break;
  case 't':	/* Triton */
    tgb	='T';
    break;
  case 'a':	/* Alpha */
    tgb	='A';
    break;
  }
  switch(tga){
  case 'e':	/* electrons */
    gmga	=cme/cmi;
    Zga		=-1.;
    break;
  case 'P':	/* Proton */
    gmga	=1.;
    Zga		=1.;
    break;
  case 'D':	/* Deuteron */
    gmga	=2.;
    Zga		=1.;
    break;
  case 'T':	/* Triton */
    gmga	=3.;
    Zga		=1.;
    break;
  case 'A':	/* Alpha */
    gmga	=4.;
    Zga		=2.;
    break;
  default:
    printf("'%c' -wrong type. Use from the set:\n'e','H','P','D','T','A' \n",
	   tga);
    return(1);
  }
  sega	=Zga*ce;
  smga	=gmga*cmi;

  switch(tgb){
  case 'e':	/* electrons */
    gmgb	=cme/cmi;
    Zgb		=-1.;
    break;
  case 'P':	/* Proton */
    gmgb	=1.;
    Zgb		=1.;
    break;
  case 'D':	/* Deuteron */
    gmgb	=2.;
    Zgb		=1.;
    break;
  case 'T':	/* Triton */
    gmgb	=3.;
    Zgb		=1.;
    break;
  case 'A':	/* Alpha */
    gmgb	=4.;
    Zgb		=2.;
    break;
  default:
    printf("'%c' -wrong type. Use from the set:\n'e','H','P','D','T','A' \n",
	   tgb);
    return(1);
  }
  segb	=Zgb*ce;
  smgb	=gmgb*cmi;

#ifdef H
  switch(tga){
  case 'e':	/* electrons */
    switch(tgb){
    case 'e':			/* electrons */
      getcollisions_	=GetEECollisions;
      break;
    case 'P':			/* Proton */
    case 'D':			/* Deuteron */
    case 'T':			/* Triton */
    case 'A':			/* Alpha */
      pZei		=Zgb*Zgb;
      getcollisions_	=GetIECollisions;
      break;
    }
    break;
  case 'P':	/* Proton */
  case 'D':	/* Deuteron */
  case 'T':	/* Triton */
  case 'A':			/* Alpha */
    switch(tgb){
    case 'e':			/* electrons */
      pZei		=Zga*Zga;
      getcollisions_	=GetIECollisions;
      break;
    case 'P':			/* Proton */
    case 'D':			/* Deuteron */
    case 'T':			/* Triton */
    case 'A':			/* Alpha */
      getcollisions_	=GetIICollisions;
      break;
    }
    break;
  }
#endif
  return(0);
}


/* the following subroutine are added for convenience 2003 */
/* a0 and gq0 are passed arrays with needed data, n is their size */
int esiget2dthese(REAL *Fj,  REAL *Faj, REAL *gFaj, REAL *gYaj, REAL *rj,
             REAL *raj, REAL *rqj, REAL *zj,   REAL *zaj,  REAL *zqj, REAL *Bj,
             REAL *a0,  REAL *gq0, int  *n)
{
  /* Exception: at a=0 the routine returns:
     dr_q/a, dz_q/a, dB_q/a, dgF/a, dgY/a
     rather than
     dr_q=0, dz_q=0, dB_q=0, dgF=0, dgY=0 */
  int j0,j1,k;
  REAL x,X,xx,XX;
  REAL f0,f1,fq0,fq1;
  REAL a,q;

  for(k =0; k < *n; k++){
    a	=a0[k];
    q	=gq0[k];
    
    j1	=q*crgq0;
    if(q < 0.){
      j1--;
    }
    q	-=cgq0*j1;
    j0	=q*rhq;
    if(j0 >= Np){
      j0	-=Np;
      q	-=cgq0;
    }
    if(j0 < 0){
      j0	+=Np;
      q	+=cgq0;
    }
    j1	=j0+1;
    y	=(q-gq[j0])*rhq;
    Y	=1.-y;
   
    YY	=Y*Y;
    yy	=y*y;
    YF	=YY*(3.-2.*Y);
    YD	=YY*y*hq;
    yF	=yy*(3.-2.*y);
    yD	=-yy*Y*hq;
    
    dY	=3.*y*Y;
    dYD	=Y-dY;
    dyD	=y-dY;
    dY	*=2.*rhq;
    if(a < 0.){
      a	=-a;
    }
    i0	=(a-sa[0])*rha;
    if(a != 0.){
      if(i0 >= Na){
	i0	=Na-1;
      }
      i1	=i0+1;
      
      x	=(a-sa[i0])*rha;
      X	=1.-x;

      XX	=X*X;
      xx	=x*x;
      XF	=XX*(3.-2.*X);
      XD	=XX*x*ha;
      xF	=xx*(3.-2.*x);
      xD	=-xx*X*ha;
      
      dX	=3.*x*X;
      dXD	=X-dX;
      dxD	=x-dX;
      dX	*=2.*rha;
      
      j00	=Np1*i0+j0;
      j01	=j00+1;
      j10	=j00+Np1;
      j11	=j10+1;
      
      /* r,r'_a,r'_q */
      f0	=XF*sr [j00]+xF*sr [j10]+XD*sra [j00]+xD*sra [j10];
      f1	=XF*sr [j01]+xF*sr [j11]+XD*sra [j01]+xD*sra [j11];
      fq0	=XF*srq[j00]+xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
      fq1	=XF*srq[j01]+xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
      rj[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      rqj[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sr [j10]-sr [j00])+dXD*sra [j00]+dxD*sra [j10];
      f1	=dX*(sr [j11]-sr [j01])+dXD*sra [j01]+dxD*sra [j11];
      fq0	=dX*(srq[j10]-srq[j00])+dXD*sraq[j00]+dxD*sraq[j10];
      fq1	=dX*(srq[j11]-srq[j01])+dXD*sraq[j01]+dxD*sraq[j11];
      raj[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* z,z'_a,z'_q */
      f0	=XF*sz [j00]+xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
      f1	=XF*sz [j01]+xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
      fq0	=XF*szq[j00]+xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
      fq1	=XF*szq[j01]+xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
      zj[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      zqj[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sz [j10]-sz [j00])+dXD*sza [j00]+dxD*sza [j10];
      f1	=dX*(sz [j11]-sz [j01])+dXD*sza [j01]+dxD*sza [j11];
      fq0	=dX*(szq[j10]-szq[j00])+dXD*szaq[j00]+dxD*szaq[j10];
      fq1	=dX*(szq[j11]-szq[j01])+dXD*szaq[j01]+dxD*szaq[j11];
      zaj[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* B */
      f0	=XF*aB [j00]+xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
      f1	=XF*aB [j01]+xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
      fq0	=XF*aBq[j00]+xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
      fq1	=XF*aBq[j01]+xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
      Bj[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;

      Fj[k]	=XF*aF[i0]+xF*aF[i1]+XD*daF[i0]+xD*daF[i1];
      Faj[k]	=dX*(aF[i1]-aF[i0])+dXD*daF[i0]+dxD*daF[i1];
      gFaj[k]	=XF*dgF[i0]+xF*dgF[i1]+XD*d2gF[i0]+xD*d2gF[i1];
      gYaj[k]	=XF*dgY[i0]+xF*dgY[i1]+XD*d2gY[i0]+xD*d2gY[i1];
      gFaj[k]	*=a;
      gYaj[k]	*=a;
    }
    else{	/* a == 0. */
      rj[k]	=sr[0];
      raj[k]	=YF*sra[j0]+yF*sra[j1]+YD*sraq[j0]+yD*sraq[j1];
      rqj[k]	=dY*(sra[j1]-sra[j0])+dYD*sraq[j0]+dyD*sraq[j1];
      zj[k]	=sz[0];
      zaj[k]	=YF*sza[j0]+yF*sza[j1]+YD*szaq[j0]+yD*szaq[j1];
      zqj[k]	=dY*(sza[j1]-sza[j0])+dYD*szaq[j0]+dyD*szaq[j1];
      Bj[k]	=aB[0];
      Fj[k]	=aF[0];
      Faj[k]	=0.;
      gFaj[k]	=dgF[0];
      gYaj[k]	=dgY[0];
    }
  }
  return(0);
}


int esigetb(REAL *particle_array,int *n)
{
  /* Exception: at a=0 the routine returns:
     dr_q/a, dz_q/a, dB_q/a, dgF/a, dgY/a
     rather than
     dr_q=0, dz_q=0, dB_q=0, dgF=0, dgY=0 */

/*
  a  = particle_array[k][0] = zion(1,k+1) (Fortran side)
  q  = particle_array[k][1] = zion(2,k+1) (Fortran side)
*/

  int j0,j1,k;
  REAL x,X,xx,XX;
  REAL f0,f1,fq0,fq1;
  REAL a,q;

#pragma omp parallel for private(k,a,q,j1,j0,y,Y,YY,yy,x,X,xx,XX,f0,f1,fq0,fq1,YF,YD,yF,yD,dY,dYD,dyD,i0,i1,XF,XD,xF,xD,dX ,dXD,dxD,j00,j01,j10,j11)
  for(k =0; k < *n; k++){
    a	= *(particle_array+k*7);    /* particle_array[k][0] */
    q	= *(particle_array+k*7+1);  /* particle_array[k][1] */
    
    j1	=q*crgq0;
    if(q < 0.){
      j1--;
    }
    q	-=cgq0*j1;
    j0	=q*rhq;
    if(j0 >= Np){
      j0	-=Np;
      q	-=cgq0;
    }
    if(j0 < 0){
      j0	+=Np;
      q	+=cgq0;
    }
    j1	=j0+1;
    y	=(q-gq[j0])*rhq;
    Y	=1.-y;
   
    YY	=Y*Y;
    yy	=y*y;
    YF	=YY*(3.-2.*Y);
    YD	=YY*y*hq;
    yF	=yy*(3.-2.*y);
    yD	=-yy*Y*hq;
    
    dY	=3.*y*Y;
    dYD	=Y-dY;
    dyD	=y-dY;
    dY	*=2.*rhq;
    if(a < 0.){
      a	=-a;
    }
    i0	=(a-sa[0])*rha;
    if(a != 0.){
      if(i0 >= Na){
	i0	=Na-1;
      }
      i1	=i0+1;
      
      x	=(a-sa[i0])*rha;
      X	=1.-x;

      XX	=X*X;
      xx	=x*x;
      XF	=XX*(3.-2.*X);
      XD	=XX*x*ha;
      xF	=xx*(3.-2.*x);
      xD	=-xx*X*ha;
      
      dX	=3.*x*X;
      dXD	=X-dX;
      dxD	=x-dX;
      dX	*=2.*rha;
      
      j00	=Np1*i0+j0;
      j01	=j00+1;
      j10	=j00+Np1;
      j11	=j10+1;
      
      /* B */
      f0	=XF*aB [j00]+xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
      f1	=XF*aB [j01]+xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
      fq0	=XF*aBq[j00]+xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
      fq1	=XF*aBq[j01]+xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
      B[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
    }
    else{	/* a == 0. */
      B[k]	=aB[0];
    }
  }
  return(0);
}


int esigetbf(REAL *a0,REAL *gq0,int *n)
{
  /* Exception: at a=0 the routine returns:
     dr_q/a, dz_q/a, dB_q/a, dgF/a, dgY/a
     rather than
     dr_q=0, dz_q=0, dB_q=0, dgF=0, dgY=0 */
  int j0,j1,k;
  REAL x,X,xx,XX;
  REAL f0,f1,fq0,fq1;
  REAL a,q;

  for(k =0; k < *n; k++){
    a	=a0[k];
    q	=gq0[k];
    
    j1	=q*crgq0;
    if(q < 0.){
      j1--;
    }
    q	-=cgq0*j1;
    j0	=q*rhq;
    if(j0 >= Np){
      j0	-=Np;
      q	-=cgq0;
    }
    if(j0 < 0){
      j0	+=Np;
      q	+=cgq0;
    }
    j1	=j0+1;
    y	=(q-gq[j0])*rhq;
    Y	=1.-y;
   
    YY	=Y*Y;
    yy	=y*y;
    YF	=YY*(3.-2.*Y);
    YD	=YY*y*hq;
    yF	=yy*(3.-2.*y);
    yD	=-yy*Y*hq;
    
    dY	=3.*y*Y;
    dYD	=Y-dY;
    dyD	=y-dY;
    dY	*=2.*rhq;
    if(a < 0.){
      a	=-a;
    }
    i0	=(a-sa[0])*rha;
    if(a != 0.){
      if(i0 >= Na){
	i0	=Na-1;
      }
      i1	=i0+1;
      
      x	=(a-sa[i0])*rha;
      X	=1.-x;

      XX	=X*X;
      xx	=x*x;
      XF	=XX*(3.-2.*X);
      XD	=XX*x*ha;
      xF	=xx*(3.-2.*x);
      xD	=-xx*X*ha;
      
      dX	=3.*x*X;
      dXD	=X-dX;
      dxD	=x-dX;
      dX	*=2.*rha;
      
      j00	=Np1*i0+j0;
      j01	=j00+1;
      j10	=j00+Np1;
      j11	=j10+1;
      
      /* B, F */
      f0	=XF*aB [j00]+xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
      f1	=XF*aB [j01]+xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
      fq0	=XF*aBq[j00]+xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
      fq1	=XF*aBq[j01]+xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
      B[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;

      F[k]	=XF*aF[i0]+xF*aF[i1]+XD*daF[i0]+xD*daF[i1];
    }
    else{	/* a == 0. */
      B[k]	=aB[0];
      F[k]	=aF[0];
    }
  }
  return(0);
}


int esigetbfgrada(REAL *a0,REAL *gq0,int *n)
{
  /* Exception: at a=0 the routine returns:
     dr_q/a, dz_q/a, dB_q/a, dgF/a, dgY/a
     rather than
     dr_q=0, dz_q=0, dB_q=0, dgF=0, dgY=0 */
  int j0,j1,k;
  REAL x,X,xx,XX;
  REAL f0,f1,fq0,fq1;
  REAL a,q;

  for(k =0; k < *n; k++){
    a	=a0[k];
    q	=gq0[k];
    
    j1	=q*crgq0;
    if(q < 0.){

      j1--;
    }
    q	-=cgq0*j1;
    j0	=q*rhq;
    if(j0 >= Np){
      j0	-=Np;
      q	-=cgq0;
    }
    if(j0 < 0){
      j0	+=Np;
      q	+=cgq0;
    }
    j1	=j0+1;
    y	=(q-gq[j0])*rhq;
    Y	=1.-y;
   
    YY	=Y*Y;
    yy	=y*y;
    YF	=YY*(3.-2.*Y);
    YD	=YY*y*hq;
    yF	=yy*(3.-2.*y);
    yD	=-yy*Y*hq;
    
    dY	=3.*y*Y;
    dYD	=Y-dY;
    dyD	=y-dY;
    dY	*=2.*rhq;
    if(a < 0.){
      a	=-a;
    }
    i0	=(a-sa[0])*rha;
    if(a != 0.){
      if(i0 >= Na){
	i0	=Na-1;
      }
      i1	=i0+1;
      
      x	=(a-sa[i0])*rha;
      X	=1.-x;

      XX	=X*X;
      xx	=x*x;
      XF	=XX*(3.-2.*X);
      XD	=XX*x*ha;
      xF	=xx*(3.-2.*x);
      xD	=-xx*X*ha;
      
      dX	=3.*x*X;
      dXD	=X-dX;
      dxD	=x-dX;
      dX	*=2.*rha;
      
      j00	=Np1*i0+j0;
      j01	=j00+1;
      j10	=j00+Np1;
      j11	=j10+1;
      
      /* r,r'_a,r'_q */
      f0	=XF*sr [j00]+xF*sr [j10]+XD*sra [j00]+xD*sra [j10];
      f1	=XF*sr [j01]+xF*sr [j11]+XD*sra [j01]+xD*sra [j11];
      fq0	=XF*srq[j00]+xF*srq[j10]+XD*sraq[j00]+xD*sraq[j10];
      fq1	=XF*srq[j01]+xF*srq[j11]+XD*sraq[j01]+xD*sraq[j11];
      /*  r[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;  */
      rq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sr [j10]-sr [j00])+dXD*sra [j00]+dxD*sra [j10];
      f1	=dX*(sr [j11]-sr [j01])+dXD*sra [j01]+dxD*sra [j11];
      fq0	=dX*(srq[j10]-srq[j00])+dXD*sraq[j00]+dxD*sraq[j10];
      fq1	=dX*(srq[j11]-srq[j01])+dXD*sraq[j01]+dxD*sraq[j11];
      ra[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* z,z'_a,z'_q */
      f0	=XF*sz [j00]+xF*sz [j10]+XD*sza [j00]+xD*sza [j10];
      f1	=XF*sz [j01]+xF*sz [j11]+XD*sza [j01]+xD*sza [j11];
      fq0	=XF*szq[j00]+xF*szq[j10]+XD*szaq[j00]+xD*szaq[j10];
      fq1	=XF*szq[j01]+xF*szq[j11]+XD*szaq[j01]+xD*szaq[j11];
      /*  z[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;  */
      zq[k]=dY*(f1-f0)+dYD*fq0+dyD*fq1;
      f0	=dX*(sz [j10]-sz [j00])+dXD*sza [j00]+dxD*sza [j10];
      f1	=dX*(sz [j11]-sz [j01])+dXD*sza [j01]+dxD*sza [j11];
      fq0	=dX*(szq[j10]-szq[j00])+dXD*szaq[j00]+dxD*szaq[j10];
      fq1	=dX*(szq[j11]-szq[j01])+dXD*szaq[j01]+dxD*szaq[j11];
      za[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;
      
      /* B,B'_a,B'_q */
      f0	=XF*aB [j00]+xF*aB [j10]+XD*aBa [j00]+xD*aBa [j10];
      f1	=XF*aB [j01]+xF*aB [j11]+XD*aBa [j01]+xD*aBa [j11];
      fq0	=XF*aBq[j00]+xF*aBq[j10]+XD*aBaq[j00]+xD*aBaq[j10];
      fq1	=XF*aBq[j01]+xF*aBq[j11]+XD*aBaq[j01]+xD*aBaq[j11];
      B[k]=YF*f0+yF*f1+YD*fq0+yD*fq1;

      F[k]	=XF*aF[i0]+xF*aF[i1]+XD*daF[i0]+xD*daF[i1];
    }
    else{	/* a == 0. */
      ra[k]	=YF*sra[j0]+yF*sra[j1]+YD*sraq[j0]+yD*sraq[j1];
      rq[k]	=dY*(sra[j1]-sra[j0])+dYD*sraq[j0]+dyD*sraq[j1];
      za[k]	=YF*sza[j0]+yF*sza[j1]+YD*szaq[j0]+yD*szaq[j1];
      zq[k]	=dY*(sza[j1]-sza[j0])+dYD*szaq[j0]+dyD*szaq[j1];
      B[k]	=aB[0];
      F[k]	=aF[0];
    }
  }
  return(0);
}
