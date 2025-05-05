#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "include/helper_cuda.h"
#include <cuda_runtime.h>

#define NPROBS 3

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

#define DBG_CHECK { printf("DBG_CHECK: file %s at line %d\n", __FILE__, __LINE__ ); }
// #define DEBUG
// #define BLK_DISPACH

#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype int

#define PRINT_ACCESS_PATTERN

/*
__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__global__
void example_kernel(int n, dtype *a, dtype* b, dtype* c)
{
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());

  // [ ... ]
}
*/

            /* |========================================| */
            /* |         Put here your kernels          | */
            /* |========================================| */

__global__
void kernelA_opt2 (int len, dtype* a, dtype *b, dtype *c) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tsize = len / (gridDim.x*blockDim.x);
	if (len % (gridDim.x*blockDim.x) != 0) tsize++;
	
	int accessindex = tid*tsize;
	for (int i=0; i<tsize; i++) {
		if (accessindex < len)
#ifndef PRINT_ACCESS_PATTERN
			c[accessindex] = a[accessindex] + b[accessindex];
#else
			c[accessindex] = (dtype)tid;  // to test access pattern
// 			printf("tid %d writing to c[%d]\n", tid, accessindex);
#endif
		accessindex++;
	}

}

__global__
void kernelB_opt1 (int len, dtype* a, dtype *b, dtype *c, int h) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int nthreads = gridDim.x*blockDim.x;
	int tsize = (len+nthreads-1) / (nthreads);
	
	int accessindex = (tid < h) ? (tid+nthreads-h)*tsize : (tid-h)*tsize ;
	for (int i=0; i<tsize; i++) {
		if (accessindex < len) 
#ifndef PRINT_ACCESS_PATTERN
			c[accessindex] = a[accessindex] + b[accessindex];
#else
			c[accessindex] = (dtype)tid;  // to test access pattern
// 			printf("tid %d writing to c[%d]\n", tid, accessindex);
#endif
		accessindex++;
	}

}

__global__
void kernelC_opt2 (int len, dtype* a, dtype *b, dtype *c) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int nthreads = gridDim.x*blockDim.x;
	int tsize = ((len%nthreads) == 0) ? len/nthreads : (len/nthreads)+1;
	
	int accessindex = (tid/2)*2*tsize ;
	if ( tid%2 == 1 ) accessindex++;
	for (int i=0; i<tsize; i++) {
		if (accessindex < len) 
#ifndef PRINT_ACCESS_PATTERN
			c[accessindex] = a[accessindex] + b[accessindex];
#else
			c[accessindex] = (dtype)tid;  // to test access pattern
// 			printf("tid %d writing to c[%d]\n", tid, accessindex);
#endif
		accessindex+=2;
	}
	

}


double check(int len, dtype *vec_a, dtype *vec_b) {
  double error = 0.0f;
  for (int i=0; i<len; i++)
    error += (float)fabs(vec_a[i] - vec_b[i]);

  return error;
}





int main(int argc, char *argv[]) {

  printf("====================================== Problem computations ======================================\n");
// =========================================== Set-up the problem ============================================

  if (argc < 2) {
    printf("Usage: lab2_ex1 n\n");
    return(1);
  }
  printf("argv[1] = %s\n", argv[1]);

  // ---------------- set-up the problem size -------------------

  int n = atoi(argv[1]), len = (1<<n), i;

  printf("n = %d --> len = 2^(n) = %d\n", n, len);
  printf("dtype = %s\n", XSTR(dtype));


  // ------------------ set-up the timers ---------------------

  TIMER_DEF;
  float error, cputime, gputime, errorvec[NPROBS], timevec[NPROBS];


  // ------------------- set-up the problem -------------------

  dtype *a, *b, *CPU_c, *GPU_c;
  a = (dtype*)malloc(sizeof(dtype)*len);
  b = (dtype*)malloc(sizeof(dtype)*len);
  CPU_c = (dtype*)malloc(sizeof(dtype)*len);
  GPU_c = (dtype*)malloc(sizeof(dtype)*len);
  time_t t;
  srand((unsigned) time(&t));

  int typ = (strcmp( XSTR(dtype) ,"int")==0);
  if (typ) {
      // here we generate random ints
      int rand_range = (1<<11);
      printf("rand_range= %d\n", rand_range);
      for (i=0; i<len; i++) {
          a[i] = rand()/(rand_range);
          b[i] = rand()/(rand_range);
          GPU_c[i] = (dtype)0;
      }
  } else {
      // here we generate random floats
      for (i=0; i<len; i++) {
        a[i] = (dtype)rand()/((dtype)RAND_MAX);
        b[i] = (dtype)rand()/((dtype)RAND_MAX);
        GPU_c[i] = (dtype)0;
      }
  }


// ======================================== Running the computations =========================================

 
  TIMER_START;
  for (i=0; i<len; i++)
    CPU_c[i] = a[i] + b[i];
  TIMER_STOP;
  error = 0.0;
  cputime = TIMER_ELAPSED;


  // ---------------- allocing GPU vectors -------------------
  dtype *dev_a, *dev_b, *dev_c;

  checkCudaErrors( cudaMalloc(&dev_a, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_b, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_c, len*sizeof(dtype)) );


  // ------------ copy data from host to device --------------

  checkCudaErrors( cudaMemset(dev_c, 0, sizeof(dtype)*len) );
  checkCudaErrors( cudaMemcpy(dev_a, a, sizeof(dtype)*len, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dev_b, b, sizeof(dtype)*len, cudaMemcpyHostToDevice) );

			
	//int blk_size = 32;
	//int grd_size = ((len%blk_size)==0) ? (len/blk_size) :  ((len/blk_size)+1) ;
	int blk_size = 128;
	int grd_size = 1;
	printf("blk_size = %d, grd_size = %d\n", blk_size, grd_size);
	
    for (int pn=0; pn<NPROBS; pn++) {

      // ------------------- problem set-up ----------------------

      switch(pn) {
        case 0:
          printf("kernelA_opt2\n");
          TIMER_START;
          kernelA_opt2<<<grd_size, blk_size>>>(len, dev_a, dev_b, dev_c);
          checkCudaErrors( cudaDeviceSynchronize() );
          TIMER_STOP;
          break;
        case 1:
          printf("kernelB_opt1\n");
          TIMER_START;
          kernelB_opt1<<<grd_size, blk_size>>>(len, dev_a, dev_b, dev_c, blk_size/2);
          checkCudaErrors( cudaDeviceSynchronize() );
          TIMER_STOP;
          break;
        case 2:
          printf("kernelC_opt2\n");
          TIMER_START;
          kernelC_opt2<<<grd_size, blk_size>>>(len, dev_a, dev_b, dev_c);
          checkCudaErrors( cudaDeviceSynchronize() );
          TIMER_STOP;
          break;
      }

      gputime = 0.0f;
      gputime += TIMER_ELAPSED;
      timevec[pn] = gputime;
      printf("Elapsed time: %lf\n", gputime);

      // ----------- copy results from device to host ------------

      checkCudaErrors( cudaMemcpy(GPU_c, dev_c, sizeof(dtype)*len, cudaMemcpyDeviceToHost) );

#ifdef PRINT_ACCESS_PATTERN
      // ------------------- Print GPU vector --------------------

      printf("GPU_c: ");
      int typ = (strcmp( XSTR(dtype) ,"int")==0);
      if (typ) {
          // here we print ints
          for (i = 0; i < len; i++) printf("%d ", GPU_c[i]);
      } else {
          // here we print floats
          for (i = 0; i < len; i++) printf("%lf ", GPU_c[i]);
      }
      printf("\n");
#else
      // ------------- Compare GPU and CPU solution --------------
      error = check(len, CPU_c, GPU_c);
      printf("Error between CPU and GPU is %lf\n", error);
      errorvec[pn] = error;
#endif

      checkCudaErrors( cudaMemset(dev_c, 0, sizeof(dtype)*len) );

    }


  // ----------------- free GPU variable ---------------------

  checkCudaErrors( cudaFree(dev_a) );
  checkCudaErrors( cudaFree(dev_b) );
  checkCudaErrors( cudaFree(dev_c) );

  // ---------------------------------------------------------


// ============================================ Print the results ============================================

  printf("================================== Times and results of my code ==================================\n");
  for (int pn=0; pn<NPROBS; pn++) {
    printf("Vector len = %d:\t%5.3f\t%5.3f\n", len, errorvec[pn], timevec[pn]);
  }

  return(0);
}
