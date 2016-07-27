//
// GPU routines for SPD Linear Algebra benchmark
//
// see matinv.c for more info
//
// Alan Gray, EPCC, August 2014
// (C) EPCC, The University of Edinburgh 2014

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "cublas_v2.h"

#include "magma.h"
#include "magma_lapack.h"


void checkCUDAError(const char *msg);

// print GPU device number
extern "C" void printDeviceNumber(){

  int device;

  cudaGetDevice(&device);
  checkCUDAError("getDevice");  
  
  printf("GPU device number %d will be used (if called upon)\n\n",device);
  

}

// square matrix-matrix multiplication on the GPU
extern "C" void matmult_gpu(double* mat1, double* mat2, 
			   double* result, int size){

double alpha=1.0; 
double beta=0.0; 
int ld=2;  
cublasHandle_t handle; 
cublasCreate(&handle);


cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,size, size, size, &alpha, mat1,ld, mat2, ld, &beta, result, ld);

cublasDestroy(handle);

 return;
  
  
}


// matrix inversion on the GPU
extern "C" void invert_gpu(double* mat1, int size){


   
    double t1, t2;
    int info;
    magma_init();
    
    
    magma_dpotrf(MagmaLower , size,  mat1, size, &info);
//    if(info!=0){
//      printf("Error:MAGMA  Cholesky factorisation - return value %d \n",info);
//      exit(1);
//    }
 
    magma_dpotri(MagmaLower, size, mat1, size, &info);
//     if(info!=0){
//       printf("Error: MAGMA Inverse calculation - return value %d \n",info);
//        exit(1);
//     }  

    magma_finalize();

    return;


}


/* check for CUDA errors */
void checkCUDAError(const char *msg)
{
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) 
        {
                fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                cudaGetErrorString( err) );
                fflush(stdout);
                fflush(stderr);
                exit(EXIT_FAILURE);
        }                         
}

