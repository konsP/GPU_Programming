/*
 * This is a CUDA code that performs an iterative reverse edge 
 * detection algorithm.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2013 
 */




/*
	Update your both your kernel, and the code responsible for specifying the decomposition such that that a 2D decomposition is over both rows and columns.
	The original code uses 256 threads per block in a 1D CUDA decomposition. 
	Replace this with 16 threads in each of the X and Y directions of the 2D CUDA decomposition, 
	to give a total of 256 threads per block. Ensure that the number of blocks is specified appropriately in each direction.
	Ensure that you retain memory coalescing
	Again, measure performance and compare to the previous versions.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/types.h>
#include <sys/time.h>

#include "reconstruct.h"

/* Data buffer to read edge data into */
float edge[N][N];

/* Data buffer for the resulting image */
float img[N][N];

/* Work buffers, with halos */
float host_input[N+2][N+2];
float gpu_output[N+2][N+2];
float host_output[N+2][N+2];


int main(int argc, char *argv[])
{
  int x, y;
  int i;
  int errors;

  double start_time_inc_data, end_time_inc_data;
  double cpu_start_time, cpu_end_time;

  float *d_input, *d_output, *d_edge, *d_temp;

  size_t memSize = (N+2) * (N+2) * sizeof(float);

  printf("Image size: %dx%d\n", N, N);
  printf("ITERATIONS: %d\n", ITERATIONS);




#define THREADSPERBLOCK 256

if ( N%THREADSPERBLOCK != 0 ){
    printf("Error: THREADSPERBLOCK must exactly divide N\n");
    exit(1);
 }




  /* allocate memory on device */
  cudaMalloc(&d_input, memSize);
  cudaMalloc(&d_output, memSize);
  cudaMalloc(&d_edge, memSize);

  /* read in edge data */
  datread("edge2048x2048.dat", (void *)edge, N, N);

  /* zero buffer so that halo is zeroed */
  for (y = 0; y < N+2; y++) {
    for (x = 0; x < N+2; x++) {
      host_input[y][x] = 0.0;
    }
  }

  /* copy input to buffer with halo */
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
       host_input[y+1][x+1] = edge[y][x];
    }
  }



  /* CUDA decomposition */
    dim3 blocksPerGrid(N/16,N/16,1);
    dim3 threadsPerBlock(16,16,1);

    printf("Blocks: %d %d %d\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);
   printf("Threads per block: %d %d %d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);


  /*
   * copy to all the GPU arrays. d_output doesn't need to have this data but
   * this will zero its halo
   */
  
    start_time_inc_data = get_current_time();
  	cudaMemcpy( d_input, host_input, memSize, cudaMemcpyHostToDevice);
  	cudaMemcpy( d_output, host_input, memSize, cudaMemcpyHostToDevice);
  	cudaMemcpy( d_edge, host_input, memSize, cudaMemcpyHostToDevice);
  
  /* We can simply reverse the roles of the input and output buffers in device memory after each iteration. 
  	In order to do this you will need to:
	remove the cudaMemcpy calls from inside the main loop.
	replace them with code to swap the pointers d_input and  d_output (you will need to declare a new temporary pointer).
	add a new cudaMemcpy call after the end of the loop (in between the two calls to get_current_time()) to copy the final result back from 
	the GPU to the gpu_output buffer in host memory. 
    
    (Note: remember that the 
	buffer pointers will have been swapped at the end of the loop, so the output 
	from the last iteration will now be pointed to by d_input!).
*/


  /* run on GPU */
  for (i = 0; i < ITERATIONS; i++) {
    /* run the kernel */
    
    inverseEdgeDetect<<< blocksPerGrid, threadsPerBlock >>>(d_output, d_input, d_edge);
 
    cudaThreadSynchronize();
    
    cudaMalloc(&d_temp, memSize);
    
    d_temp =   d_output;
	d_output = d_input;
   	d_input =  d_temp;
    
  }
  
  
  																							 /* copy the data back from the output buffer on the device */
  																							 /* cudaMemcpy(gpu_output, d_output, memSize, cudaMemcpyDeviceToHost);*/

  																							  /* copy the new data to the input buffer on the device */
 																							  /* cudaMemcpy( d_input, gpu_output, memSize, cudaMemcpyHostToDevice);*/

  end_time_inc_data = get_current_time();
  
  
  /*copy the final result back from 
	the GPU to the gpu_output buffer in host memory*/
  cudaMemcpy(gpu_output, d_input, memSize, cudaMemcpyDeviceToHost);

  checkCUDAError("Main loop");

  /*
   * run on host for comparison
   */
  cpu_start_time = get_current_time();
  for (i = 0; i < ITERATIONS; i++) {

    /* perform stencil operation */
    for (y = 0; y < N; y++) {
      for (x = 0; x < N; x++) {
	host_output[y+1][x+1] = (host_input[y+1][x] + host_input[y+1][x+2] +
				 host_input[y][x+1] + host_input[y+2][x+1] \
				 - edge[y][x]) * 0.25;
      }
    }
    
    /* copy output back to input buffer */
    for (y = 0; y < N; y++) {
      for (x = 0; x < N; x++) {
	host_input[y+1][x+1] = host_output[y+1][x+1];
      }
    }
  }
  cpu_end_time = get_current_time();

/* Maximum difference allowed between host result and GPU result */
#define MAX_DIFF 0.01

  /* check that GPU result matches host result */
  errors = 0;
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      float diff = fabs(gpu_output[y+1][x+1] - host_output[y+1][x+1]);
      if (diff >= MAX_DIFF) {
        errors++;
        //printf("Error at %d,%d (CPU=%f, GPU=%f)\n", x, y,	\
  	  //     host_output[y+1][x+1],				\
		   //	      gpu_output[y+1][x+1]);
      }
    }
  }

  if (errors == 0) 
    printf("\n\n ***TEST PASSED SUCCESSFULLY*** \n\n\n");
  else
    printf("\n\n ***ERROR: TEST FAILED*** \n\n\n");

  /* copy result to output buffer */
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      img[y][x] = gpu_output[y+1][x+1];
    }
  }

  /* write PGM */
  pgmwrite("output.pgm", (void *)img, N, N);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_edge);
  cudaFree(d_temp);

  printf("GPU Time (Including Data Transfer): %fs\n", \
	 end_time_inc_data - start_time_inc_data);
  printf("CPU Time                          : %fs\n", \
	 cpu_end_time - cpu_start_time);

  return 0;
}

