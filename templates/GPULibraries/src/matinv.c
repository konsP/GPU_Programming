//
// GPU Linear Algebra Library training exercise template file
//
// Based on SPD Linear Algebra benchmark
//  -- generates NxN Symmetric Positive Definite matrix
//  -- inverts matrix using cholesky factorisation
//  -- multiplies original matrix by inverse 
//  -- verifies by comparing with identity matrix 
//  
//
//  Alan Gray, EPCC, August 2014
// (C) EPCC, The University of Edinburgh 2014



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mkl_lapack.h>
#include <mkl_cblas.h>
#include <string.h>

#include <sys/time.h>


//OPTIONS:

//matrix is size NxN
#define N 2048



int main(int argc, char *argv[])
{
  int i, j, k, iter, nthreads;
  double time0, time1, time2;
  
  int info;
  char uplo='L';
  int _n=N;

  printf("\nStarting Benchmark \n\n");

  printDeviceNumber();

  // allocate memory
  double *matrix = (double*) malloc (N*N*sizeof(double));
  double *matrix_save = (double*) malloc (N*N*sizeof(double));
  double *eyecalc = (double*) malloc (N*N*sizeof(double));

  if ((!matrix) || (!matrix_save) || (!eyecalc)) {
    printf("Error. Memory allocation failed.\n");
    exit(1);
  }

  printf("Creating random %dx%d matrix..\n",N,N);
  for (i = 0; i < N; i++) 
    for (j = 0; j < N; j++) 
      matrix[i*N+j] = ((double) rand())/RAND_MAX;            
  printf("...Done\n\n");

  //copy to matrix_save
  memcpy(matrix_save,matrix,N*N*sizeof(double));

  printf("Making matrix symmetric positive definite...\n");
  //A=A+A'
  for (i = 0; i < N; i++) 
    for (j = 0; j < N; j++) 
      matrix[i*N+j] = matrix_save[i*N+j]+matrix_save[j*N+i];

  //A=A+N*eye(N)
  for (i = 0; i < N; i++)
    matrix[i*N+i] += (double) N;

  printf("...Done\n\n");

  //copy to matrix_save
  memcpy(matrix_save,matrix,N*N*sizeof(double));

  time0 = omp_get_wtime(); 


  printf("\n\nPerforming Matrix Inversion using host CPU...\n");

  // invert_gpu(matrix, N)
  /*
  //Replace matrix with it's Cholesky factorisation
  dpotrf_( &uplo, &_n, matrix, &_n, &info );
  if(info!=0){
    printf("Error: Cholesky factorisation - return value %d \n",info);
    exit(1);
  }
  
  //Replace (cholesky factorisation of) matrix with it's inverse
  dpotri_( &uplo, &_n, matrix, &_n, &info );
  if(info!=0){
    printf("Error: Inverse calculation - return value %d \n",info);
    exit(1);
  }
  */

  /* call invert_gpu(double* mat1, int size) -- executes on the GPU */
  invert_gpu(matrix,_n);


  time1 = omp_get_wtime(); 
  printf("...Matrix inversion: %1.5f seconds\n\n",time1-time0);  

  // fill other half of symmetric matrix before dgemm
  // (note: in real life we would not do this but use dsymm BLAS call
  // for matrix multiplication)
   for (j = 0; j < N; j++) 
     for (i = j; i < N; i++) 
       matrix[i*N+j]=matrix[j*N+i];






  time0 = omp_get_wtime();    

  printf("\n\nMultiplying by original matrix directly...\n");
  /*
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      eyecalc[i*N+j]=0.;
      for (k = 0; k < N; k++) {
  	eyecalc[i*N+j]+=matrix[i*N+k]*matrix_save[k*N+j];
      }
    }
  }
  */
  
  //call to cblas_dgemm
  
   double alpha = 1.0;
   double beta = 0.0;
   
   // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,N, N,N, alpha, matrix, N, matrix_save, N, beta, eyecalc, N);
  
   matmult_gpu(matrix, matrix_save, eyecalc, N);
 

  time1 = omp_get_wtime(); 
  printf("...Matrix multiplication : %1.5f seconds\n\n",time1-time0);  




  //verify resulting matrix against identity matrix
  double tmpdiff;
  double diff=0.;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (i==j)
	tmpdiff = fabs(eyecalc[i*N+j] - 1.);
      else
	tmpdiff = fabs(eyecalc[i*N+j] - 0.);
       if (tmpdiff > diff) diff = tmpdiff;      
      
    }
  }

  printf("\n\nMax diff of resulting matrix from I is %1.3e\n\n",diff);


  //clean up
  free(matrix);
  free(matrix_save);
  free(eyecalc);
  
  printf ("\nBenchmark Completed.\n\n"); 

  return 0;
}



