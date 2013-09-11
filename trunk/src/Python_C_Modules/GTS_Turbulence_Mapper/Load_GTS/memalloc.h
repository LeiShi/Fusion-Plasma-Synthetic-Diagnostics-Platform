/*! \file memalloc.h
  \brief Handy array memory allocation functions
*/

#ifndef erik_memalloc_c
#define erik_memalloc_c
#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#ifndef single			// compiler option determines variable size
#ifndef REAL_TYPEDEF
#define REAL_TYPEDEF
typedef double REAL;
#endif
#else
#ifndef REAL_TYPEDEF
#define REAL_TYPEDEF
typedef float REAL;
#endif
#endif


#define ERRORCODE -1

// function headers:
void *xmalloc(size_t size);
REAL **realAllocContiguous2d(size_t nrows,size_t ncols);
REAL ***realAllocContiguous3d(size_t npages,size_t nrows,size_t ncols);
void realFreeContiguous2d(REAL ***array2d);
void realFreeContiguous3d(REAL ****array3d);


// function definitions:

//! Outputs error and exits if memory allocation fails
/*! Taken from GNU C library documentation. */
void *xmalloc(size_t size){
  register void *value = PyMem_Malloc (size);
  if (value == 0){
    error("xmalloc: virtual memory exhausted\n");
    exit(-1);
  }
  return value;
}

//! Allocates a contiguous block of memory for a 2d array
/*! Access using either: ptr[npages][nrows][ncols]
     or ptr[i*nrows*ncols + j*ncols + k]
*/
REAL **realAllocContiguous2d(size_t nrows,size_t ncols){
  REAL  **ptr = (REAL **)PyMem_Malloc(nrows*sizeof(REAL *));
  ptr[0] = (REAL *)PyMem_Malloc(nrows*ncols*sizeof(REAL));

  int i;
  for(i=0;i<nrows;i++) ptr[i] = &(ptr[0][ncols*i]);

  return ptr;
}

//! Free the memory allocated by realAllocContiguous2d
/*! Pass a 2d array's address: realFreeContiguous2d(&array2d)
     when it returns, the pointer at that address has the value of null.
*/
void realFreeContiguous2d(REAL ***array2d){
  PyMem_Free((*array2d)[0]);
  PyMem_Free((*array2d));
  *array2d = NULL;
}

//! Allocate a contiguous block of memory for a 3d array
/*! Access like either: ptr[npages][nrows][ncols]
     or ptr[i*nrows*ncols + j*ncols + k]
*/
REAL ***realAllocContiguous3d(size_t npages,size_t nrows,size_t ncols){
  REAL  ***ptr = (REAL ***)PyMem_Malloc(npages*sizeof(REAL **));
  ptr[0] = (REAL **)PyMem_Malloc(npages*nrows*sizeof(REAL *));
  ptr[0][0] = (REAL *)PyMem_Malloc(npages*nrows*ncols*sizeof(REAL));

  int i,j;
  for(i=0;i<npages;i++){
    ptr[i] = &(ptr[0][nrows*i]);
    for(j=0;j<nrows;j++) ptr[i][j] = &(ptr[0][0][ncols*(nrows*i + j)]);
  }

  return ptr;
}

//! Free the memory allocated by realAllocContiguous3d
/*! Pass a 3d array's address: realFreeContiguous3d(&array3d)
   when it returns, the pointer at that address has the value of null.
*/
void realFreeContiguous3d(REAL ****array3d){
  PyMem_Free((*array3d)[0][0]);
  PyMem_Free((*array3d)[0]);
  PyMem_Free((*array3d));
  *array3d = NULL;
}


//! Free memory associated with a discontiguous 2d array
/*! Pass a 2d array's address: realFreeContiguous2d(&array2d)
   when it returns, the pointer at that address has the value of null.
*/
void realFreeDiscontiguous2d(int nrows,REAL ***array2d){
  int i;
  for(i=0;i<nrows;i++) PyMem_Free((*array2d)[i]);
  PyMem_Free(*array2d);
  *array2d = NULL;
}

//! Free memory associated with a discontiguous 3d array
/*! Pass a 3d array's address: realFreeContiguous3d(&array3d)
   when it returns, the pointer at that address has the value of null.
*/
void realFreeDiscontiguous3d(int npages,int nrows,REAL ****array3d){
  int i,j;
  for(i=0;i<npages;i++){
    for(j=0;j<nrows;j++) PyMem_Free((*array3d)[i][j]);
    PyMem_Free((*array3d)[i]);
  }
  PyMem_Free(*array3d);
  *array3d = NULL;
}



#endif
