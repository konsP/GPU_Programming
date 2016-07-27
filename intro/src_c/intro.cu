/*
 * This is a simple CUDA code that negates an array of integers.
 * It introduces the concepts of device memory management, and
 * kernel invocation.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010 
 */

#include <stdio.h>
#include <stdlib.h>

/* Forward Declaration*/
/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* The actual array negation kernel (basic single block version) */
__global__ void negate(int *d_a)
{
    /* Part 2B: negate an element of d_a */
  	/* 2B) Implement the actual kernel function to negate an array element as follows: */
    /* int idx = threadIdx.x; */
    /* d_a[idx] = -1 * d_a[idx];*/
  
  	int idx = threadIdx.x;
	d_a[idx] = -1 * d_a[idx];
}

/* Multi-block version of kernel for part 2C */
__global__ void negate_multiblock(int *d_a)
{
    /* Part 2C: negate an element of d_a, using multiple blocks this time */
  	/* 2C) Implement the kernel again, this time allowing multiple thread blocks. */
    /* It will be very similar to the previous kernel implementation except that */
    /* the array index will be computed differently: */
    /* int idx = threadIdx.x + (blockIdx.x * blockDim.x);*/

	/*Remember to also change the kernel invocation to invoke negate_multiblock this time.*/
    /* With this version you can change NUM_BLOCKS and THREADS_PER_BLOCK to have different values — */
    /* so long as they still multiply to give the array size.*/
  
  	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  	d_a[idx] = -1 * d_a[idx];
}

/* The number of integer elements in the array */
#define ARRAY_SIZE 10240

/*
 * The number of CUDA blocks and threads per block to use.
 * These should always multiply to give the array size.
 * For the single block kernel, NUM_BLOCKS should be 1 and
 * THREADS_PER_BLOCK should be the array size
 */


#define NUM_BLOCKS 20
#define THREADS_PER_BLOCK 512

//#define NUM_BLOCKS 3125
//#define THREADS_PER_BLOCK 32

/* Main routine */
int main(int argc, char *argv[])
{
    int *h_a, *h_out;
    int *d_a;

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * allocate memory on host
     * h_a holds the input array, h_out holds the result
     */
    h_a = (int *) malloc(sz);
    h_out = (int *) malloc(sz);

    /*
     * allocate memory on device
     */
    /* Part 1A: allocate device memory: use the existing pointer <code>d_a</code> and the variable
<code>sz</code> (which has already been assigned the size of the array in bytes).*/
  
  	cudaMalloc(&d_a, sz); 

    /* initialise host arrays */
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = i;
        h_out[i] = 0;
    }

    /* copy input array from host to GPU */
    /* Part 1B: copy host array h_a to device array d_a -- Copy the array <code>h_a</code> on the host to <code>d_a</code> on the device.*/
  
  	cudaMemcpy(d_a, h_a, sz,cudaMemcpyHostToDevice); 

    /* run the kernel on the GPU */
    /* Part 2A: configure and launch kernel (un-comment and complete) */
    /*Configure and launch the kernel using a 1D grid and a single thread block (NUM_BLOCKS and THREADS_PER_BLOCK are already defined for this case).*/
     dim3 blocksPerGrid(NUM_BLOCKS,1,1); 
     dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1); 
     negate_multiblock<<< blocksPerGrid , threadsPerBlock>>>(d_a); 

    /* wait for all threads to complete and check for errors */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* copy the result array back to the host */
    /* Part 1C: copy device array d_a to host array h_out  Copy <code>d_a</code> on the device back to <code>h_out</code> on the host.*/
  
  	cudaMemcpy(h_out, d_a, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", h_out[i]);
    }
    printf("\n\n");
    

    /* free device buffer */
    /* Part 1D: free d_a */
  	cudaFree(d_a);

    /* free host buffers */
    free(h_a);
    free(h_out);

    return 0;
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
