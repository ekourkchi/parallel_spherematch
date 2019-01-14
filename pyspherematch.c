
/*****************************************************************
//
//  NAME:		Ehsan Kourkchi
//
//  HOMEWORK:		Final Project
//
//  CLASS:		ICS 632
//
//  INSTRUCTOR:		Henri Casanova
//
//  DATE:		December 2015          
//
//
//  DESCRIPTION:	The very last wrapper code in C (it's calls different functions of Astrometry.net package.
//                        
//                      These functions were added by Ehsan
//          To be used internally in this code     1) static kdtree_t* esn_make_tree(PyArrayObject* x)
//                          match_sss_omp   :-     2) static PyObject* spherematch_match_esn_omp(PyObject* self, PyObject* args)
//                          match_sss_omp   :-     3) PyObject* spherematch_match_sss_omp(PyObject* self, PyObject* args)
//                          match_esn_omp   :-     4) PyObject* spherematch_match_sdr_omp(PyObject* self, PyObject* args)
// 
//****************************************************************/



/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include "Python.h"

#include <stdio.h>
#include <assert.h>

#include "numpy/arrayobject.h"

#include "kdtree.h"
#include "kdtree_fits_io.h"
#include "dualtree_rangesearch.h"
#include "dualtree_nearestneighbour.h"
#include "bl.h"



#include <omp.h>
#define PI 3.14159265359



/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_build(PyObject* self, PyObject* args) {
    int N, D;
    int i,j;
    int Nleaf, treeoptions, treetype;
    kdtree_t* kd;
    double* data;
    PyArrayObject* x;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x))
        return NULL;

    if (PyArray_NDIM(x) != 2) {
        PyErr_SetString(PyExc_ValueError, "array must be two-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(x) != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "array must contain doubles");
        return NULL;
    }

    N = (int)PyArray_DIM(x, 0);
    D = (int)PyArray_DIM(x, 1);

    if (D > 10) {
        PyErr_SetString(PyExc_ValueError, "maximum dimensionality is 10: maybe you need to transpose your array?");
        return NULL;
    }

    data = malloc(N * D * sizeof(double));
    for (i=0; i<N; i++) {
        for (j=0; j<D; j++) {
            double* pd = PyArray_GETPTR2(x, i, j);
            data[i*D + j] = *pd;
        }
    }

    Nleaf = 16;
    treetype = KDTT_DOUBLE;
    //treeoptions = KD_BUILD_SPLIT;
    treeoptions = KD_BUILD_BBOX;
    kd = kdtree_build(NULL, data, N, D, Nleaf,
                      treetype, treeoptions);
    return Py_BuildValue("k", kd);
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_free(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;

    if (!PyArg_ParseTuple(args, "l", &i)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;
    free(kd->data.any);
    kdtree_free(kd);
    return Py_BuildValue("");
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_write(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;
    char* fn;
    int rtn;

    if (!PyArg_ParseTuple(args, "ls", &i, &fn)) {
        PyErr_SetString(PyExc_ValueError, "need two args: kdtree identifier (int), filename (string)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;

    rtn = kdtree_fits_write(kd, fn, NULL);
    return Py_BuildValue("i", rtn);
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_open(PyObject* self, PyObject* args) {
    kdtree_t* kd;
    char* fn;
    char* treename = NULL;
    int n;

    n = PyTuple_Size(args);
    if (!((n == 1) || (n == 2))) {
        PyErr_SetString(PyExc_ValueError, "need one or two args: kdtree filename + optionally tree name");
        return NULL;
    }
    if (n == 1) {
        if (!PyArg_ParseTuple(args, "s", &fn)) {
            return NULL;
        }
    } else {
        if (!PyArg_ParseTuple(args, "ss", &fn, &treename)) {
            return NULL;
        }
    }
    kd = kdtree_fits_read(fn, treename, NULL);
    return Py_BuildValue("k", kd);
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_close(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;

    if (!PyArg_ParseTuple(args, "l", &i)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;
    kdtree_fits_close(kd);
    return Py_BuildValue("");
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_n(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;
    if (!PyArg_ParseTuple(args, "l", &i)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;
    return PyInt_FromLong(kdtree_n(kd));
}

/////////////////////////////////////////////////

struct dualtree_results2 {
    kdtree_t *kd1;
    kdtree_t *kd2;
    PyObject* indlist;
    anbool permute;
};

/////////////////////////////////////////////////

static void callback_dualtree2(void* v, int ind1, int ind2, double dist2) {
    struct dualtree_results2* dt = v;
    PyObject* lst;
    if (dt->permute) {
        ind1 = kdtree_permute(dt->kd1, ind1);
        ind2 = kdtree_permute(dt->kd2, ind2);
    }
    lst = PyList_GET_ITEM(dt->indlist, ind1);
    if (!lst) {
        lst = PyList_New(1);
        // SetItem steals a ref -- that's what we want.
        PyList_SetItem(dt->indlist, ind1, lst);
        PyList_SET_ITEM(lst, 0, PyInt_FromLong(ind2));
    } else {
        PyList_Append(lst, PyInt_FromLong(ind2));
    }
}

/////////////////////////////////////////////////

static PyObject* spherematch_match2(PyObject* self, PyObject* args) {
    int i, N;
    long p1, p2;
    struct dualtree_results2 dtresults;
    kdtree_t *kd1, *kd2;
    double rad;
    PyObject* indlist;
	anbool notself;
	anbool permute;
	
	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "lldbb", &p1, &p2, &rad, &notself, &permute)) {
        PyErr_SetString(PyExc_ValueError, "spherematch_c.match: need five args: two kdtree identifiers (ints), search radius (float), notself (boolean), permuted (boolean)");
        return NULL;
    }
    // Nasty!
    kd1 = (kdtree_t*)p1;
    kd2 = (kdtree_t*)p2;
    
    N = kdtree_n(kd1);
    indlist = PyList_New(N);
    assert(indlist);

    dtresults.kd1 = kd1;
    dtresults.kd2 = kd2;
    dtresults.indlist = indlist;
    dtresults.permute = permute;

    dualtree_rangesearch(kd1, kd2, 0.0, rad, notself, NULL,
                         callback_dualtree2, &dtresults,
                         NULL, NULL);

    // set empty slots to None, not NULL.
    for (i=0; i<N; i++) {
        if (PyList_GET_ITEM(indlist, i))
            continue;
        Py_INCREF(Py_None);
        PyList_SET_ITEM(indlist, i, Py_None);
    }

    return indlist;
}

/////////////////////////////////////////////////

struct dualtree_results {
    il* inds1;
    il* inds2;
    dl* dists;
};

/////////////////////////////////////////////////

static void callback_dualtree(void* v, int ind1, int ind2, double dist2) {
    struct dualtree_results* dtresults = v;
    il_append(dtresults->inds1, ind1);
    il_append(dtresults->inds2, ind2);
    dl_append(dtresults->dists, sqrt(dist2));
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
// original Function
/////////////////////////////////////////////////
/////////////////////////////////////////////////

static PyObject* spherematch_match(PyObject* self, PyObject* args) {
    size_t i, N;
    long p1, p2;
    kdtree_t *kd1, *kd2;
    double rad;
    struct dualtree_results dtresults;
    PyArrayObject* inds;
    npy_intp dims[2];
    PyArrayObject* dists;
	anbool notself;
	anbool permute;
	PyObject* rtn;
	
	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "lldbb", &p1, &p2, &rad, &notself, &permute)) {
        PyErr_SetString(PyExc_ValueError, "spherematch_c.match: need five args: two kdtree identifiers (ints), search radius (float), notself (boolean), permuted (boolean)");
        return NULL;
    }
	//printf("Notself = %i\n", (int)notself);
    // Nasty!
    kd1 = (kdtree_t*)p1;
    kd2 = (kdtree_t*)p2;

    dtresults.inds1 = il_new(256);
    dtresults.inds2 = il_new(256);
    dtresults.dists = dl_new(256);
    dualtree_rangesearch(kd1, kd2, 0.0, rad, notself, NULL,
                         callback_dualtree, &dtresults,
                         NULL, NULL);

    N = il_size(dtresults.inds1);
    dims[0] = N;
    dims[1] = 2;

    inds =  (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_INT);
    dims[1] = 1;
    dists = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
    for (i=0; i<N; i++) {
      int index;
      int* iptr;
      double* dptr;
      iptr = PyArray_GETPTR2(inds, i, 0);
      index = il_get(dtresults.inds1, i);
      if (permute)
        index = kdtree_permute(kd1, index);
      *iptr = index;
      iptr = PyArray_GETPTR2(inds, i, 1);
      index = il_get(dtresults.inds2, i);
      if (permute)
        index = kdtree_permute(kd2, index);
      *iptr = index;
      dptr = PyArray_GETPTR2(dists, i, 0);
      *dptr = dl_get(dtresults.dists, i);
    }

    il_free(dtresults.inds1);
    il_free(dtresults.inds2);
    dl_free(dtresults.dists);

    rtn = Py_BuildValue("(OO)", inds, dists);
    Py_DECREF(inds);
    Py_DECREF(dists);
    return rtn;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
// Start  Ehsan's work
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

/*
 * Generating a k-d tree for a givel Python 1D array
 * This is a piece of code taken out from the body of "spherematch_match" function
 * to make a better use of it
 * 
 */
static kdtree_t* esn_make_tree(PyArrayObject* x) { 
  
    int N, D;
    int i,j;
    int Nleaf, treeoptions, treetype;
    kdtree_t* kd;
    double* data;


    N = (int)PyArray_DIM(x, 0);
    D = (int)PyArray_DIM(x, 1);

    if (D > 10) {
        PyErr_SetString(PyExc_ValueError, "maximum dimensionality is 10: maybe you need to transpose your array?");
        return NULL;
    }

    data = malloc(N * D * sizeof(double));
    for (i=0; i<N; i++) {
        for (j=0; j<D; j++) {
            double* pd = PyArray_GETPTR2(x, i, j);
            data[i*D + j] = *pd;
        }
    }
    
    
    Nleaf = 16;
    treetype = KDTT_DOUBLE;
    //treeoptions = KD_BUILD_SPLIT;
    treeoptions = KD_BUILD_BBOX;
    kd = kdtree_build(NULL, data, N, D, Nleaf,
                      treetype, treeoptions);
    
    return kd;
    
}



/////////////////////////////////////////////////

static PyObject* spherematch_match_esn_omp(PyObject* self, PyObject* args) {
  
  
  int iam=0, np=1;
    
  
    int iter, n_iter = 1;
    
    long n_size;
    
    int ii,jj;
    int Nleaf, treeoptions, treetype;
    double* data;   
    int NN, DD;
    size_t i, N;
    long p1, p2;
    PyArrayObject *x, *y;
    kdtree_t *kd1, **kd2;
    double rad;
    struct dualtree_results dtresults;
    struct dualtree_results *dtresults_array;
    PyArrayObject* inds;
    npy_intp dims[2];
    PyArrayObject* dists;
	anbool notself;
	anbool permute;
	PyObject* rtn;
	
	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "O!O!dlbb", &PyArray_Type, &x, &PyArray_Type, &y, &rad, &n_size, &notself, &permute)) {
        PyErr_SetString(PyExc_ValueError, "spherematch_c.match: need five args: two kdtree identifiers (ints), search radius (float), notself (boolean), permuted (boolean)");
        return NULL;
    }

    
    
    // x is the small array (e.g. it has 100 points)
    kd1 = esn_make_tree(x);

 
    
    NN = (int)PyArray_DIM(y, 0);
    DD = (int)PyArray_DIM(y, 1);
    if (DD > 10) {
	      PyErr_SetString(PyExc_ValueError, "maximum dimensionality is 10: maybe you need to transpose your array?");
	      return NULL;
	  }
    
    data = malloc(NN * DD * sizeof(double));
    
    
    // nn, n_size indicate the size of each sub-array of 'y' for OpenMP threads
    int nn = n_size;
    
    // n_iter is the number of OpenMP iterations
    n_iter = NN / nn;
    
    
    
    if (NN-nn*n_iter != 0) n_iter++;
    

    
    // allocating memory for all OMP tasks
    dtresults_array = malloc(n_iter * sizeof(struct dualtree_results));
    kd2 = malloc(n_iter * sizeof(kdtree_t*));
    
    
    

    
    // main OMP loop
    #pragma omp parallel for shared(data, kd1, kd2, dtresults_array, y) \
	      private(ii, jj)
    for(iter=0; iter < n_iter; iter++) {
      

    
	  
	  for (ii=iter*nn; ii<NN && ii<(iter+1)*nn; ii++) {
	      for (jj=0; jj<DD; jj++) {
		  double* pd = PyArray_GETPTR2(y, ii, jj);
		  data[ii*DD + jj] = *pd;
	      }
	  }
	  
	  
	  Nleaf = 16;
	  treetype = KDTT_DOUBLE;
	  treeoptions = KD_BUILD_BBOX;
	  kd2[iter] = kdtree_build(NULL, data+(iter*nn*DD), ii-iter*nn, DD, Nleaf,
			    treetype, treeoptions);    
	  
	  
	  dtresults_array[iter].inds1 = il_new(256);
	  dtresults_array[iter].inds2 = il_new(256);
	  dtresults_array[iter].dists = dl_new(256);
	  dualtree_rangesearch(kd1, kd2[iter], 0.0, rad, notself, NULL,
			      callback_dualtree, &dtresults_array[iter],
			      NULL, NULL);
	  
    }    // OMP ends
    
    
    // Calculating the total number of pairs
    // Allocating memory to cook-up an output for the Python code
    N = 0;
    for(iter=0; iter < n_iter; iter++) {
        N += il_size(dtresults_array[iter].inds1);
    }
    

    
    
    dims[0] = N;

    dims[1] = 2;
    inds =  (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_INT);
    
    dims[1] = 1;
    dists = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
    
    
    // Extracting the information of pairs and storing them in the final array
    // Cooking up the output for Python, i.e. inds, dists
    // inds --> indices of pairs
    // dists --> the angular separation of points in pairs
    ii = 0;
    for(iter=0; iter < n_iter; iter++) {
      N = il_size(dtresults_array[iter].inds1);
      for (i=0; i<N; i++) {
        int index;
        int* iptr;
        double* dptr;
        
	iptr = PyArray_GETPTR2(inds, ii, 0);
        index = il_get(dtresults_array[iter].inds1, i);
        if (permute)
          index = kdtree_permute(kd1, index);
        *iptr = index;
        
	iptr = PyArray_GETPTR2(inds, ii, 1);
        index = il_get(dtresults_array[iter].inds2, i);
        if (permute)
          index = kdtree_permute(kd2[iter], index);
        *iptr = index + iter*nn;
	
        dptr = PyArray_GETPTR2(dists, ii, 0);
        *dptr = dl_get(dtresults_array[iter].dists, i);
      
      ii++;

      }

       il_free(dtresults_array[iter].inds1);
       il_free(dtresults_array[iter].inds2);
       dl_free(dtresults_array[iter].dists);
       
//         free(kd2[iter]->data.any);
       kdtree_free(kd2[iter]);
    }

    
    
  
    // packing (inds and dists)
    rtn = Py_BuildValue("(OO)", inds, dists);
    Py_DECREF(inds);
    Py_DECREF(dists);
    
    
    free(kd1->data.any);
    kdtree_free(kd1);
    
    
    free(dtresults_array);
    free(kd2);
    
    return rtn;

}

/////////////////////////////////////////////////



/*
 * This is a dedicated function optimized for calculating S values ... 
 * 
 * It finds all the pairs as the previous function does, in the next step, it performs a bunch 
 * of mathematical calculations to find the desired histogram, and returns the histogram as opposed to the all pairs 
 */

static PyObject* spherematch_match_sss_omp(PyObject* self, PyObject* args) {
  

    int iam=0, np=1;
    
  
    int iter, n_iter = 1;
    int n_bins;
    double powmin, powmax, delta;
    double* edges;
    int* hist_s;
    
    
    long check0;
    double d = 0;
    double checksum = 0;
    int m1,m2;
    double s1,s2,pp,rp;
    
    
    
    long n_size;
    PyObject* histo_cpython; 
    PyArrayObject *output, *output1;
    int ii,jj, low;
    int Nleaf, treeoptions, treetype;
    double* data;   
    int NN, DD, N_match, N_s;
    size_t i, N;
    long p1, p2;
    PyArrayObject *x, *y, *s;
    kdtree_t *kd1, **kd2;
    double rad;
    struct dualtree_results dtresults;
    struct dualtree_results *dtresults_array;
    PyArrayObject* inds;
    npy_intp dims[2];
	anbool notself;
	anbool permute;
	
	
	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "O!O!dO!iddilbb", &PyArray_Type, &x, &PyArray_Type, &y, &rad, &PyArray_Type, &s, &low, &powmin, &powmax, &n_bins, &n_size, &notself, &permute)) {
        PyErr_SetString(PyExc_ValueError, "spherematch_c.match: need five args: two kdtree identifiers (ints), search radius (float), notself (boolean), permuted (boolean)");
        return NULL;
    }

    
    
    kd1 = esn_make_tree(x);

 
    
    NN = (int)PyArray_DIM(y, 0);
    DD = (int)PyArray_DIM(y, 1);
    if (DD > 10) {
	      PyErr_SetString(PyExc_ValueError, "maximum dimensionality is 10: maybe you need to transpose your array?");
	      return NULL;
	  }
    
    data = malloc(NN * DD * sizeof(double));
    
    
    int nn = n_size;
    n_iter = NN / nn;
    
    
    
    if (NN-nn*n_iter != 0) n_iter++;
    

    
    dtresults_array = malloc(n_iter * sizeof(struct dualtree_results));
    kd2 = malloc(n_iter * sizeof(kdtree_t*));
    

    
    // main loop
    #pragma omp parallel for shared(data, kd1, kd2, dtresults_array, y) \
	      private(ii, jj)
    for(iter=0; iter < n_iter; iter++) {
      

    
	  
	  for (ii=iter*nn; ii<NN && ii<(iter+1)*nn; ii++) {
	      for (jj=0; jj<DD; jj++) {
		  double* pd = PyArray_GETPTR2(y, ii, jj);
		  data[ii*DD + jj] = *pd;
	      }
	  }
	  
	  
	  Nleaf = 16;
	  treetype = KDTT_DOUBLE;
	  treeoptions = KD_BUILD_BBOX;
	  kd2[iter] = kdtree_build(NULL, data+(iter*nn*DD), ii-iter*nn, DD, Nleaf,
			    treetype, treeoptions);    
	  
	  
	  dtresults_array[iter].inds1 = il_new(256);
	  dtresults_array[iter].inds2 = il_new(256);
	  dtresults_array[iter].dists = dl_new(256);
	  dualtree_rangesearch(kd1, kd2[iter], 0.0, rad, notself, NULL,
			      callback_dualtree, &dtresults_array[iter],
			      NULL, NULL);
	  
    } // OMP ends
    
    
    N_match = 0;
    for(iter=0; iter < n_iter; iter++) {
        N_match += il_size(dtresults_array[iter].inds1);
    }
    


    int* inds1 =  malloc(N_match * sizeof(int));
    int* inds2 = malloc(N_match * sizeof(int));
    double* dists = malloc(N_match * sizeof(double));
    double* out_array = malloc(N_match * sizeof(double));
    
    
    ii = 0;
    for(iter=0; iter < n_iter; iter++) {
      N = il_size(dtresults_array[iter].inds1);
      for (i=0; i<N; i++) {
        int index;
        int* iptr;
        double* dptr;
        
	
        index = il_get(dtresults_array[iter].inds1, i);
        if (permute)
          index = kdtree_permute(kd1, index);
        inds1[ii] = index;
        
	
        index = il_get(dtresults_array[iter].inds2, i);
        if (permute)
          index = kdtree_permute(kd2[iter], index);
        inds2[ii] = index + iter*nn;
	
        
        dists[ii] = dl_get(dtresults_array[iter].dists, i);
      
      ii++;

      }

       il_free(dtresults_array[iter].inds1);
       il_free(dtresults_array[iter].inds2);
       dl_free(dtresults_array[iter].dists);
       

       kdtree_free(kd2[iter]);
    }

    

    free(kd1->data.any);
    kdtree_free(kd1);
    
    
    free(dtresults_array);
    free(kd2);
    
    
    
    
    N_s = (int)PyArray_DIM(s, 0);
    DD = (int)PyArray_DIM(s, 1);

    double* s_array = malloc(N_s * sizeof(double));
    for (ii=0; ii<N_s; ii++) {
      double* ps = PyArray_GETPTR2(s, ii, 0);
      s_array[ii] = *ps;
    }
    

    // dists and inds are ready here
    // Start Calculating the desired histogram
    #pragma omp parallel for shared(out_array, dists, inds1, inds2, s_array) \
	      private(ii, d, m1, m2, s1, s2, pp, rp)
    for (ii=0; ii<N_match; ii++) {
     
      d = dists[ii];
      m1 = inds1[ii];
      m2 = inds2[ii];
      d  = acos(1.-d*d/2.);
      
      s1 = s_array[m1+low];
      s2 = s_array[m2];
      pp = fabs(s1-s2);
      rp = (s1+s2)*d/2.;
      out_array[ii] = sqrt(pp*pp+rp*rp);
      
      if (dists[ii] == 0)
	out_array[ii] = 0;
         
      
//        checksum += out_array[ii];
      
    } // OMP ends
//      printf("\n SSSSS checksum: %lf %d\n", checksum, N_match);
    


    delta = (powmax-powmin)/n_bins;
//     printf("delta: %lf\n", delta);
    
    edges = malloc((n_bins+1)*sizeof(double));
    hist_s = malloc(n_bins*sizeof(int));
    
    
    for (jj=0; jj<n_bins; jj++) {
      
      hist_s[jj] = 0;
      edges[jj] = pow(10., powmin);
      powmin += delta;
      
    }
    edges[jj] = pow(10., powmin);
    
    
     #pragma omp parallel for shared(out_array, hist_s)\
	      private(ii, jj, s1)
    for (ii=0; ii<N_match; ii++) {
      s1 = out_array[ii];
      for (jj=0; jj<n_bins; jj++) {
	 
	  if (s1 > edges[jj] && s1 <= edges[jj+1])
	    #pragma omp critical 
	      hist_s[jj]++;
	
      }
    }  // OMP ends

    

    
    
    //////////////////// Cooking Python output 1-D array
    
    dims[1] = 1;
    
    // cooking for hist_s
    dims[0] = n_bins;
    output = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
       
    // cooking for edges
    dims[0] = n_bins+1;
    output1 = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
    
    // filling for hist_s
    for (ii=0; ii<n_bins; ii++) {
      double* dptr;
      
      dptr = PyArray_GETPTR2(output, ii, 0);
      *dptr = hist_s[ii];   // This is where the output is injected 
    }
    
    // filling for edges
    for (ii=0; ii<n_bins+1; ii++) {
      double* dptr;
      
      dptr = PyArray_GETPTR2(output1, ii, 0);
      *dptr = edges[ii];   // This is where the output is injected 
      
    }
    
    
    histo_cpython = Py_BuildValue("(OO)", output, output1);
    
    
    // Free Py-memory 
    Py_DECREF(output);
    Py_DECREF(output1);
    
    
    
     //////////////////// END - cooking Python output 1-D array
    
    free(hist_s);
    free(edges);
    
    free(inds1);
    free(inds2);
    free(dists);
    free(s_array);
    free(out_array);
    
 
    return histo_cpython;


}


/////////////////////////////////////////////////


/*
 *  This is the same as the previous function, It takes two 'S' arrays for <DR> caculation 
 * 
 */

static PyObject* spherematch_match_sdr_omp(PyObject* self, PyObject* args) {
  
  
  
  int iam=0, np=1;
    
  
    int iter, n_iter = 1;
    int n_bins;
    double powmin, powmax, delta;
    double* edges;
    int* hist_s;
    
    
    long check0;
    double d = 0;
    double checksum = 0;
    int m1,m2;
    double s1,s2,pp,rp;
    
    
    
    long n_size;
    PyObject* histo_cpython; 
    PyArrayObject *output, *output1;
    int ii,jj, low;
    int Nleaf, treeoptions, treetype;
    double* data;   
    int NN, DD, N_match, N_s;
    size_t i, N;
    long p1, p2;
    PyArrayObject *x, *y, *s, *s_rand;
    kdtree_t *kd1, **kd2;
    double rad;
    struct dualtree_results dtresults;
    struct dualtree_results *dtresults_array;
    PyArrayObject* inds;
    npy_intp dims[2];
	anbool notself;
	anbool permute;
	
	
	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "O!O!dO!O!iddilbb", &PyArray_Type, &x, &PyArray_Type, &y, &rad, &PyArray_Type, &s, &PyArray_Type, &s_rand, &low, &powmin, &powmax, &n_bins, &n_size, &notself, &permute)) {
        PyErr_SetString(PyExc_ValueError, "spherematch_c.match: need five args: two kdtree identifiers (ints), search radius (float), notself (boolean), permuted (boolean)");
        return NULL;
    }

    
    
    kd1 = esn_make_tree(x);
//  kd2 = esn_make_tree(y);
 
    
    NN = (int)PyArray_DIM(y, 0);
    DD = (int)PyArray_DIM(y, 1);
    if (DD > 10) {
	      PyErr_SetString(PyExc_ValueError, "maximum dimensionality is 10: maybe you need to transpose your array?");
	      return NULL;
	  }
    
    data = malloc(NN * DD * sizeof(double));
    
    
    int nn = n_size;
    n_iter = NN / nn;
    
    
    
    if (NN-nn*n_iter != 0) n_iter++;
    
   
    
    dtresults_array = malloc(n_iter * sizeof(struct dualtree_results));
    kd2 = malloc(n_iter * sizeof(kdtree_t*));
    

    
    // main loop
    #pragma omp parallel for shared(data, kd1, kd2, dtresults_array, y) \
	      private(ii, jj)
    for(iter=0; iter < n_iter; iter++) {
      

    
	  
	  for (ii=iter*nn; ii<NN && ii<(iter+1)*nn; ii++) {
	      for (jj=0; jj<DD; jj++) {
		  double* pd = PyArray_GETPTR2(y, ii, jj);
		  data[ii*DD + jj] = *pd;
	      }
	  }
	  
	  
	  Nleaf = 16;
	  treetype = KDTT_DOUBLE;
	  treeoptions = KD_BUILD_BBOX;
	  kd2[iter] = kdtree_build(NULL, data+(iter*nn*DD), ii-iter*nn, DD, Nleaf,
			    treetype, treeoptions);    
	  
	  
	  dtresults_array[iter].inds1 = il_new(256);
	  dtresults_array[iter].inds2 = il_new(256);
	  dtresults_array[iter].dists = dl_new(256);
	  dualtree_rangesearch(kd1, kd2[iter], 0.0, rad, notself, NULL,
			      callback_dualtree, &dtresults_array[iter],
			      NULL, NULL);
	  
    } // OMP ends
    
    
    N_match = 0;
    for(iter=0; iter < n_iter; iter++) {
        N_match += il_size(dtresults_array[iter].inds1);
    }
    

    
    int* inds1 =  malloc(N_match * sizeof(int));
    int* inds2 = malloc(N_match * sizeof(int));
    double* dists = malloc(N_match * sizeof(double));
    double* out_array = malloc(N_match * sizeof(double));
    
    
    ii = 0;
    for(iter=0; iter < n_iter; iter++) {
      N = il_size(dtresults_array[iter].inds1);
      for (i=0; i<N; i++) {
        int index;
        int* iptr;
        double* dptr;
        
	
        index = il_get(dtresults_array[iter].inds1, i);
        if (permute)
          index = kdtree_permute(kd1, index);
        inds1[ii] = index;
        
	
        index = il_get(dtresults_array[iter].inds2, i);
        if (permute)
          index = kdtree_permute(kd2[iter], index);
        inds2[ii] = index + iter*nn;
	
        
        dists[ii] = dl_get(dtresults_array[iter].dists, i);
      
      ii++;
      }

       il_free(dtresults_array[iter].inds1);
       il_free(dtresults_array[iter].inds2);
       dl_free(dtresults_array[iter].dists);
       
       kdtree_free(kd2[iter]);
    }
    
    
  
   
    free(kd1->data.any);
    kdtree_free(kd1);
    
    
    free(dtresults_array);
    free(kd2);
    
    
    
    
    N_s = (int)PyArray_DIM(s, 0);
    DD = (int)PyArray_DIM(s, 1);
    double* s_array = malloc(N_s * sizeof(double));
    for (ii=0; ii<N_s; ii++) {
      double* ps = PyArray_GETPTR2(s, ii, 0);
      s_array[ii] = *ps;
    }
    
    
    N_s = (int)PyArray_DIM(s_rand, 0);
    DD = (int)PyArray_DIM(s_rand, 1);
    double* s_array_rand = malloc(N_s * sizeof(double));
    for (ii=0; ii<N_s; ii++) {
      double* ps = PyArray_GETPTR2(s_rand, ii, 0);
      s_array_rand[ii] = *ps;
    }
        
    

    #pragma omp parallel for shared(out_array, dists, inds1, inds2, s_array, s_array_rand) \
	      private(ii, d, m1, m2, s1, s2, pp, rp)
    for (ii=0; ii<N_match; ii++) {
     
      d = dists[ii];
      m1 = inds1[ii];
      m2 = inds2[ii];
      d  = acos(1.-d*d/2.);
      
      s1 = s_array[m1+low];
      s2 = s_array_rand[m2];
      pp = fabs(s1-s2);
      rp = (s1+s2)*d/2.;
      out_array[ii] = sqrt(pp*pp+rp*rp);
      
      if (dists[ii] == 0)
	out_array[ii] = 0;
         
            
    } // OMP ends
    


    delta = (powmax-powmin)/n_bins;
    
    edges = malloc((n_bins+1)*sizeof(double));
    hist_s = malloc(n_bins*sizeof(int));
    
    
    for (jj=0; jj<n_bins; jj++) {
      
      hist_s[jj] = 0;
      edges[jj] = pow(10., powmin);
      powmin += delta;
      
    }
    edges[jj] = pow(10., powmin);
    
    
     #pragma omp parallel for shared(out_array, hist_s)\
	      private(ii, jj, s1)
    for (ii=0; ii<N_match; ii++) {
      s1 = out_array[ii];
      for (jj=0; jj<n_bins; jj++) {
	 
	  if (s1 > edges[jj] && s1 <= edges[jj+1])
	    #pragma omp critical 
	      hist_s[jj]++;
	
      }
    } // OMP ends


    //////////////////// Cooking Python output 1-D array
    
    dims[1] = 1;
    
    // cooking for hist_s
    dims[0] = n_bins;
    output = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
       
    // cooking for edges
    dims[0] = n_bins+1;
    output1 = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
    
    // filling for hist_s
    for (ii=0; ii<n_bins; ii++) {
      double* dptr;
      
      dptr = PyArray_GETPTR2(output, ii, 0);
      *dptr = hist_s[ii];   // This is where the output is injected 
    }
    
    // filling for edges
    for (ii=0; ii<n_bins+1; ii++) {
      double* dptr;
      
      dptr = PyArray_GETPTR2(output1, ii, 0);
      *dptr = edges[ii];   // This is where the output is injected 
      
    }
    
    
    histo_cpython = Py_BuildValue("(OO)", output, output1);
    
    
    // Free Py-memory 
    Py_DECREF(output);
    Py_DECREF(output1);
    
    
    
     //////////////////// END - cooking Python output 1-D array
    
    free(hist_s);
    free(edges);
    
    free(inds1);
    free(inds2);
    free(dists);
    free(s_array);
    free(s_array_rand);
    free(out_array);
    
    
    
    return histo_cpython;


}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
// END  Ehsan's work
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////






static PyObject* spherematch_nn(PyObject* self, PyObject* args) {
    int i, NY;
    long p1, p2;
    kdtree_t *kd1, *kd2;
    npy_intp dims[1];
    PyArrayObject* inds;
    PyArrayObject* dist2s;
    int *pinds;
    double *pdist2s;
    double rad;
	anbool notself;
	int* tempinds;
	PyObject* rtn;

	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "lldb", &p1, &p2, &rad, &notself)) {
        PyErr_SetString(PyExc_ValueError, "need three args: two kdtree identifiers (ints), and search radius");
        return NULL;
    }
    // Nasty!
    kd1 = (kdtree_t*)p1;
    kd2 = (kdtree_t*)p2;

    NY = kdtree_n(kd2);

    dims[0] = NY;
    inds   = (PyArrayObject*)PyArray_SimpleNew(1, dims, PyArray_INT);
    dist2s = (PyArrayObject*)PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
    assert(PyArray_ITEMSIZE(inds) == sizeof(int));
    assert(PyArray_ITEMSIZE(dist2s) == sizeof(double));

	// YUCK!
	tempinds = (int*)malloc(NY * sizeof(int));
	double* tempdists = (double*)malloc(NY * sizeof(double));

    pinds   = tempinds; //PyArray_DATA(inds);
    //pdist2s = PyArray_DATA(dist2s);
	pdist2s = tempdists;

    dualtree_nearestneighbour(kd1, kd2, rad*rad, &pdist2s, &pinds, NULL, notself);

	// now we have to apply kd1's permutation array!
	for (i=0; i<NY; i++)
		if (pinds[i] != -1)
			pinds[i] = kdtree_permute(kd1, pinds[i]);


	pinds = PyArray_DATA(inds);
    pdist2s = PyArray_DATA(dist2s);

	for (i=0; i<NY; i++) {
		pinds[i] = -1;
		pdist2s[i] = HUGE_VAL;
	}
	// and apply kd2's permutation array!
	for (i=0; i<NY; i++) {
		if (tempinds[i] != -1) {
			int j = kdtree_permute(kd2, i);
			pinds[j] = tempinds[i];
			pdist2s[j] = tempdists[i];
		}
	}
	free(tempinds);
	free(tempdists);

	rtn = Py_BuildValue("(OO)", inds, dist2s);
	Py_DECREF(inds);
	Py_DECREF(dist2s);
	return rtn;
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_bbox(PyObject* self, PyObject* args) {
  PyArrayObject* bbox;
  PyObject* rtn;
  npy_intp dims[2];
  long i;
  double *bb;
  anbool ok;
  kdtree_t* kd;
  int j, D;

  if (!PyArg_ParseTuple(args, "l", &i)) {
    PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
    return NULL;
  }
  // Nasty!
  kd = (kdtree_t*)i;
  D = kd->ndim;
  dims[0] = D;
  dims[1] = 2;
  bbox = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  {
    double bblo[D];
    double bbhi[D];
    ok = kdtree_get_bboxes(kd, 0, bblo, bbhi);
    if (!ok) {
        Py_RETURN_NONE;
    }
    assert(ok);
    bb = PyArray_DATA(bbox);
    for (j=0; j<D; j++) {
      bb[j*2 + 0] = bblo[j];
      bb[j*2 + 1] = bbhi[j];
    }
  }
  rtn = Py_BuildValue("O", bbox);
  Py_DECREF(bbox);
  return rtn;
}


/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_print(PyObject* self, PyObject* args) {
  long i;
  kdtree_t* kd;
  if (!PyArg_ParseTuple(args, "l", &i)) {
    PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
    return NULL;
  }
  // Nasty!
  kd = (kdtree_t*)i;
  kdtree_print(kd);
  Py_RETURN_NONE;
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_rangesearch(PyObject* self,
                                                PyObject* args) {
    double* X;
    PyObject* rtn;
    npy_intp dims[1];
    long i;
    kdtree_t* kd;
    int D, N;
    PyObject* pyO;
    PyObject* pyI;
    PyObject* pyInds;
    PyObject* pyDists = NULL;
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_DOUBLE);
    int req = NPY_C_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED
        | NPY_ELEMENTSTRIDES;
    double radius;
    kdtree_qres_t* res;
    int getdists, sortdists;
    int opts;

    if (!PyArg_ParseTuple(args, "lOdii", &i, &pyO, &radius,
                          &getdists, &sortdists)) {
        PyErr_SetString(PyExc_ValueError, "need five args: kdtree identifier (int), query point (numpy array of floats), radius (double), get distances (int 0/1), sort distances (int 0/1)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;
    D = kd->ndim;

    if (sortdists) {
        getdists = 1;
    }

    Py_INCREF(dtype);
    pyI = PyArray_FromAny(pyO, dtype, 1, 1, req, NULL);
    if (!pyI) {
        PyErr_SetString(PyExc_ValueError, "Failed to convert query point array to np array of float");
        Py_XDECREF(dtype);
        return NULL;
    }
    N = (int)PyArray_DIM(pyI, 0);
    if (N != D) {
        PyErr_SetString(PyExc_ValueError, "Query point must have size == dimension of tree");
        Py_DECREF(pyI);
        Py_DECREF(dtype);
        return NULL;
    }

    X = PyArray_DATA(pyI);

    opts = 0;
    if (getdists) {
        opts |= KD_OPTIONS_COMPUTE_DISTS;
    }
    if (sortdists) {
        opts |= KD_OPTIONS_SORT_DISTS;
    }

    res = kdtree_rangesearch_options(kd, X, radius*radius, opts);
    N = res->nres;
    dims[0] = N;
    res->inds = realloc(res->inds, N * sizeof(uint32_t));
    pyInds = PyArray_SimpleNewFromData(1, dims, NPY_UINT32, res->inds);
    res->inds = NULL;

    if (getdists) {
        res->sdists = realloc(res->sdists, N * sizeof(double));
        pyDists = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, res->sdists);
        res->sdists = NULL;
    }

    kdtree_free_query(res);

    Py_DECREF(pyI);
    Py_DECREF(dtype);
    if (getdists) {
        rtn = Py_BuildValue("(OO)", pyInds, pyDists);
        Py_DECREF(pyDists);
    } else {
        rtn = Py_BuildValue("O", pyInds);
    }
    Py_DECREF(pyInds);
    return rtn;
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_get_data(PyObject* self, PyObject* args) {
  PyArrayObject* pyX;
  double* X;
  PyObject* rtn;
  npy_intp dims[2];
  long i;
  kdtree_t* kd;
  int k, D, N;
  //npy_int* I;
  npy_uint32* I;
  PyObject* pyO;
  PyObject* pyI;
  // this is the type returned by kdtree_rangesearch
  PyArray_Descr* dtype = PyArray_DescrFromType(NPY_UINT32);
  int req = NPY_C_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED | NPY_ELEMENTSTRIDES;

  if (!PyArg_ParseTuple(args, "lO", &i, &pyO)) {
    PyErr_SetString(PyExc_ValueError, "need two args: kdtree identifier (int), index array (numpy array of ints)");
    return NULL;
  }
  // Nasty!
  kd = (kdtree_t*)i;
  D = kd->ndim;

  Py_INCREF(dtype);
  pyI = PyArray_FromAny(pyO, dtype, 1, 1, req, NULL);
  if (!pyI) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert index array to np array of int");
    Py_XDECREF(dtype);
    return NULL;
  }
  N = (int)PyArray_DIM(pyI, 0);

  dims[0] = N;
  dims[1] = D;

  pyX = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  X = PyArray_DATA(pyX);
  I = PyArray_DATA(pyI);

  for (k=0; k<N; k++) {
    kdtree_copy_data_double(kd, I[k], 1, X);
    X += D;
  }
  Py_DECREF(pyI);
  Py_DECREF(dtype);
  rtn = Py_BuildValue("O", pyX);
  Py_DECREF(pyX);
  return rtn;
}

/////////////////////////////////////////////////

static PyObject* spherematch_kdtree_permute(PyObject* self, PyObject* args) {
  PyArrayObject* pyX;
  npy_int* X;
  PyObject* rtn;
  npy_intp dims[1];
  long i;
  kdtree_t* kd;
  long k, N;
  npy_int* I;
  PyObject* pyO;
  PyObject* pyI;
  PyArray_Descr* dtype = PyArray_DescrFromType(NPY_INT);
  int req = NPY_C_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED | NPY_ELEMENTSTRIDES;

  if (!PyArg_ParseTuple(args, "lO", &i, &pyO)) {
    PyErr_SetString(PyExc_ValueError, "need two args: kdtree identifier (int), index array (numpy array of ints)");
    return NULL;
  }
  // Nasty!
  kd = (kdtree_t*)i;

  Py_INCREF(dtype);
  pyI = PyArray_FromAny(pyO, dtype, 1, 1, req, NULL);
  if (!pyI) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert index array to np array of int");
    Py_XDECREF(dtype);
    return NULL;
  }
  N = PyArray_DIM(pyI, 0);

  dims[0] = N;

  pyX = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
  X = PyArray_DATA(pyX);
  I = PyArray_DATA(pyI);

  for (k=0; k<N; k++) {
    npy_int ii = I[k];
    //printf("Permute: ii=%i\n", ii);
    X[k] = kdtree_permute(kd, ii);
  }
  Py_DECREF(pyI);
  Py_DECREF(dtype);
  rtn = Py_BuildValue("O", pyX);
  Py_DECREF(pyX);
  return rtn;
}


/////////////////////////////////////////////////

static PyObject* spherematch_nn2(PyObject* self, PyObject* args) {
  int i, j, NY, N;
  long p1, p2;
  kdtree_t *kd1, *kd2;
  npy_intp dims[1];
  PyObject* I;
  PyObject* J;
  PyObject* dist2s;
  PyObject* counts = NULL;
  int *pi;
  int *pj;
  int *pc = NULL;
  double *pd;
  double rad;
  anbool notself;
  anbool docount;
  int* tempinds;
  int* tempcount = NULL;
  int** ptempcount = NULL;
  double* tempd2;
  PyObject* rtn;

  // So that ParseTuple("b") with a C "anbool" works
  assert(sizeof(anbool) == sizeof(unsigned char));

  if (!PyArg_ParseTuple(args, "lldbb", &p1, &p2, &rad, &notself, &docount)) {
    PyErr_SetString(PyExc_ValueError, "need five args: two kdtree identifiers (ints), search radius, notself (bool) and docount (bool)");
    return NULL;
  }
  // Nasty!
  kd1 = (kdtree_t*)p1;
  kd2 = (kdtree_t*)p2;

  // quick check for no-overlap case
  if (kdtree_node_node_mindist2_exceeds(kd1, 0, kd2, 0, rad*rad)) {
    // allocate empty return arrays
    dims[0] = 0;
    I = PyArray_SimpleNew(1, dims, NPY_INT);
    J = PyArray_SimpleNew(1, dims, NPY_INT);
    dist2s = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	if (docount) {
	  counts = PyArray_SimpleNew(1, dims, NPY_INT);
	  rtn = Py_BuildValue("(OOOO)", I, J, dist2s, counts);
	  Py_DECREF(counts);
	} else {
	  rtn = Py_BuildValue("(OOO)", I, J, dist2s);
	}
    Py_DECREF(I);
    Py_DECREF(J);
    Py_DECREF(dist2s);
    return rtn;
  }

  NY = kdtree_n(kd2);

  tempinds = (int*)malloc(NY * sizeof(int));
  tempd2 = (double*)malloc(NY * sizeof(double));
  if (docount) {
	tempcount = (int*)calloc(NY, sizeof(int));
    ptempcount = &tempcount;
  }

  dualtree_nearestneighbour(kd1, kd2, rad*rad, &tempd2, &tempinds, ptempcount, notself);

  // count number of matches
  N = 0;
  for (i=0; i<NY; i++)
    if (tempinds[i] != -1)
      N++;

  // allocate return arrays
  dims[0] = N;
  I = PyArray_SimpleNew(1, dims, NPY_INT);
  J = PyArray_SimpleNew(1, dims, NPY_INT);
  dist2s = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  pi = PyArray_DATA(I);
  pj = PyArray_DATA(J);
  pd = PyArray_DATA(dist2s);
  if (docount) {
    counts = PyArray_SimpleNew(1, dims, NPY_INT);
    pc = PyArray_DATA(counts);
  }

  j = 0;
  for (i=0; i<NY; i++) {
    if (tempinds[i] == -1)
      continue;
    pi[j] = kdtree_permute(kd1, tempinds[i]);
    pj[j] = kdtree_permute(kd2, i);
    pd[j] = tempd2[i];
    if (docount)
      pc[j] = tempcount[i];
    j++;
  }

  free(tempinds);
  free(tempd2);
  free(tempcount);

  if (docount) {
    rtn = Py_BuildValue("(OOOO)", I, J, dist2s, counts);
    Py_DECREF(counts);
  } else {
    rtn = Py_BuildValue("(OOO)", I, J, dist2s);
  }
  Py_DECREF(I);
  Py_DECREF(J);
  Py_DECREF(dist2s);
  return rtn;
}

/////////////////////////////////////////////////

static PyMethodDef spherematchMethods[] = {
    { "kdtree_build", spherematch_kdtree_build, METH_VARARGS,
      "build kdtree" },
    { "kdtree_write", spherematch_kdtree_write, METH_VARARGS,
      "save kdtree to file" },
    { "kdtree_open", spherematch_kdtree_open, METH_VARARGS,
      "open kdtree from file" },
    { "kdtree_close", spherematch_kdtree_close, METH_VARARGS,
      "close kdtree opened with kdtree_open" },
    { "kdtree_free", spherematch_kdtree_free, METH_VARARGS,
      "free kdtree" },

    { "kdtree_bbox", spherematch_kdtree_bbox, METH_VARARGS,
      "get bounding-box of this tree" },
    { "kdtree_n", spherematch_kdtree_n, METH_VARARGS,
      "N pts in tree" },

    { "kdtree_print", spherematch_kdtree_print, METH_VARARGS,
      "Describe kdtree" },

    { "kdtree_rangesearch", spherematch_kdtree_rangesearch, METH_VARARGS,
      "Rangesearch in a single kd-tree" },

    {"kdtree_get_positions", spherematch_kdtree_get_data, METH_VARARGS,
     "Retrieve the positions of given indices in this tree (np array of ints)" },

	{"kdtree_permute", spherematch_kdtree_permute, METH_VARARGS,
	 "Apply kd-tree permutation array to (get from kd-tree indices back to original)"},

    { "match", spherematch_match, METH_VARARGS,
      "find matching data points" },
    { "match2", spherematch_match2, METH_VARARGS,
      "find matching data points" },
    { "nearest", spherematch_nn, METH_VARARGS,
      "find nearest neighbours" },

    { "nearest2", spherematch_nn2, METH_VARARGS,
      "find nearest neighbours (different return values)" },

/////////////////////////////////////////////////
/////////////////////////////////////////////////

    { "match_esn_omp", spherematch_match_esn_omp, METH_VARARGS,
      "Ehsan OMP (top-level) match_esn_omp implementation ...." },
      
    
    { "match_sss_omp", spherematch_match_sss_omp, METH_VARARGS,
      "Ehsan OMP (top-level) match_sss_omp implementation ...." }, 
      
    { "match_sdr_omp", spherematch_match_sdr_omp, METH_VARARGS,
      "Ehsan OMP (top-level) match_sdr_omp implementation ...." }, 
      
/////////////////////////////////////////////////
/////////////////////////////////////////////////
    
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initspherematch_c(void) {
    Py_InitModule("spherematch_c", spherematchMethods);
    import_array();
}

