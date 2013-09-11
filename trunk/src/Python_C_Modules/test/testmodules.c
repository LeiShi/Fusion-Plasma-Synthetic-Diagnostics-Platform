#include<Python.h>
#include<stdio.h>

/*
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL
#endif
*/

#include<numpy/arrayobject.h>

int age = 27;
char* name = "Lei Shi";

static PyObject*
test_set(PyObject* self, PyObject* args, PyObject* kws){
  int sts = 0;
  static char* kwlist[]={"name","age",NULL};
  if (!PyArg_ParseTupleAndKeywords(args,kws,"s|i",kwlist, &name, &age))
    return NULL;
  return Py_BuildValue("i", sts);
}

static PyObject*
test_cprint(PyObject* self, PyObject* args){
  int sts = 0;
  sts = printf("My name is %s, I'm %d years old!\n",name, age);
  return Py_BuildValue("i",sts);
}

static PyObject*
test_arraysum(PyObject* self, PyObject* args){

  PyObject* input;
  PyArrayObject* array;
  double sum = 0;
  if( !PyArg_ParseTuple(args,"O", &input))
    return NULL;
  array = (PyArrayObject*) PyArray_ContiguousFromObject(input,PyArray_DOUBLE,2,2);
  if(array == NULL)    
    return NULL;
  int m=array->dimensions[0];
  int n=array->dimensions[1];
  int i;
  for(i=0;i<m*n;i++){
    sum += *(double*)(array->data + i*sizeof(double));
    }
  return PyFloat_FromDouble(sum);
}

static PyObject*
test_arrayadd(PyObject* self, PyObject* args){
  PyObject *input1, *input2;
  PyArrayObject *array1,*array2,*result;
  
  if(!PyArg_ParseTuple(args,"OO",&input1,&input2))
    return NULL;
  array1 = (PyArrayObject*) PyArray_ContiguousFromObject(input1,PyArray_DOUBLE,0,0);
  array2 = (PyArrayObject*) PyArray_ContiguousFromObject(input2,PyArray_DOUBLE,0,0);
  if(array1->nd != array2->nd){
    PyErr_SetString(PyExc_ValueError,"arrays must have the same dimension number.");
    return NULL;
  }
  else{
    int i;
    for(i=0;i<array1->nd;i++){
      if(array1->dimensions[i] != array2->dimensions[i] || array1->strides[i]!=array2->strides[i]){
	PyErr_SetString(PyExc_ValueError,"arrays must have the same length in each corresponding dimension, and the memory arrangement must be the same.");
	return NULL;
      }
    }
    int total=1;
    int* dims= (int*)PyMem_New(int,array1->nd);
    for(i=0;i<array1->nd;i++){
      total *= array1->dimensions[i];
      dims[i] = array1->dimensions[i];
    }

    result = (PyArrayObject*) PyArray_FromDims(array1->nd,dims,PyArray_DOUBLE);
    PyMem_Del(dims);
    for(i=0;i<total;i++){
      *(double*)(result->data+i*sizeof(double)) = *(double*)(array1->data+i*sizeof(double)) + *(double*)(array2->data+i*sizeof(double));
    }
    
    return PyArray_Return(result);
    
  }
    
}


static PyMethodDef TestMethods[] = {
  {"set", (PyCFunction) test_set, METH_VARARGS|METH_KEYWORDS, "set name and age."}, 
  {"cprint", test_cprint, 0, "Print out the name and age."},
  {"arraysum",test_arraysum,METH_VARARGS,"sum up all the elements in an ndarray."},
  {"arrayadd",test_arrayadd,METH_VARARGS,"add two exact same shape arrays."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
inittest_C(void){
  (void) Py_InitModule("test_C", TestMethods);
  import_array();
}
