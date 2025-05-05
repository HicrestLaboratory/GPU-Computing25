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

#define RUN_SOLUTIONS

#define BLK_SIZE 32
#define GRD_SIZE 2
#define CEIL_DIV( N, D ) ((( N ) % ( D )) == 0) ? (( N )/( D )) : ((( N )/( D ))+1)

#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype float

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



__global__
void Problem_solution_layout1(int n, dtype *a, dtype* b, dtype* c)
{
#ifdef BLK_DISPACH
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());
#endif

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
//   printf("%d = tid = blockIdx.x*blockDim.x + threadIdx.x = %d * %d + %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
  int th_tot = gridDim.x*blockDim.x;
  int values2th = ((n % th_tot) == 0) ? (n/th_tot) : (n/th_tot)+1 ;
  int val_coord;

  for (int i=0; i<values2th; i++) {
    val_coord = tid + i*th_tot;
    if (val_coord < n)
        c[val_coord] = a[val_coord] + b[val_coord]; // (dtype)tid;
  }
}

__global__
void Problem_solution_layout2(int n, dtype *a, dtype* b, dtype* c)
{
#ifdef BLK_DISPACH
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());
#endif

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
//   printf("%d = tid = blockIdx.x*blockDim.x + threadIdx.x = %d * %d + %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
  int th_tot = gridDim.x*blockDim.x;
  int values2th = ((n % th_tot) == 0) ? (n/th_tot) : (n/th_tot)+1 ;
  int val_coord;

  for (int i=0; i<values2th; i++) {
    val_coord = tid*values2th + i;
    if (val_coord < n)
        c[val_coord] = a[val_coord] + b[val_coord]; // (dtype)tid;
  }
}





int main(int argc, char *argv[]) {

  printf("======================================= Device properties ========================================\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  int dev, SM_num;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:                              %d bit\n",
           deviceProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:                     %7.3f GB/s\n",
           2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);

    printf("  Multiprocessors:                                %3d\n",
           deviceProp.multiProcessorCount);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);

    SM_num = deviceProp.multiProcessorCount;

  }

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
  float error, cputime, gputime;
  float errors[2], gputimes[2];

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
//   errors[0] = 0.0;
  cputime = TIMER_ELAPSED;


  // ---------------- allocing GPU vectors -------------------
  dtype *dev_a, *dev_b, *dev_c;

  checkCudaErrors( cudaMalloc(&dev_a, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_b, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_c, len*sizeof(dtype)) );


  // ------------ copy date from host to device --------------

  checkCudaErrors( cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dev_c, 0, len*sizeof(dtype)) );

  // ------------ computation solution with Layout 1 -----------
  TIMER_START;
			
  {
      dim3 block_size(BLK_SIZE, 1, 1);
      dim3 grid_size(CEIL_DIV( len , BLK_SIZE ), 1, 1);
      printf("block_size = %d, grid_size = %d, elements per thread = %f\n", block_size.x, grid_size.x, (float)len/(block_size.x*grid_size.x));
      Problem_solution_layout1<<<grid_size, block_size>>>(len, dev_a, dev_b, dev_c);
  }

  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  gputime = TIMER_ELAPSED;
  // ----------- copy results from device to host ------------

  checkCudaErrors( cudaMemcpy(GPU_c, dev_c, len*sizeof(dtype), cudaMemcpyDeviceToHost) );

  // ------------- Compare GPU and CPU solution --------------

  error = 0.0f;
  for (int i = 0; i < len; i++)
    error += (float)fabs(CPU_c[i] - GPU_c[i]);

  errors[0] = error;
  gputimes[0] = gputime;

  // ------------------- reset gpu buffers ---------------------

  checkCudaErrors( cudaMemset(dev_a, 0, len*sizeof(dtype)) );
  checkCudaErrors( cudaMemset(dev_b, 0, len*sizeof(dtype)) );
  checkCudaErrors( cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dev_c, 0, len*sizeof(dtype)) );

  // ------------ computation solution with Layout 2 -----------
  TIMER_START;

  {
      dim3 block_size(BLK_SIZE, 1, 1);
      dim3 grid_size(CEIL_DIV( len , BLK_SIZE ), 1, 1);
      printf("block_size = %d, grid_size = %d, elements per thread = %f\n", block_size.x, grid_size.x, (float)len/(block_size.x*grid_size.x));
      Problem_solution_layout2<<<grid_size, block_size>>>(len, dev_a, dev_b, dev_c);
  }

  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  gputime = TIMER_ELAPSED;
  // ----------- copy results from device to host ------------

  checkCudaErrors( cudaMemcpy(GPU_c, dev_c, len*sizeof(dtype), cudaMemcpyDeviceToHost) );

  // ------------- Compare GPU and CPU solution --------------

  error = 0.0f;
  for (int i = 0; i < len; i++)
    error += (float)fabs(CPU_c[i] - GPU_c[i]);

  errors[1] = error;
  gputimes[1] = gputime;

// ============================================ Print the results ============================================

  printf("================================== Times and results of my code ==================================\n");
  printf("\nVector len = %d, CPU time = %5.3f\n\n", len, cputime);
  printf("\t\tError\tTime\n");
  for (int i=0; i<2; i++)
    printf("Layout %d:\t%5.3f\t%5.3f\n", i, errors[i], gputimes[i]);

// ===================================== Fine best BLK and GRD dimensions ====================================

  printf("===================================== Fine best BLK and GRD dimensions ====================================\n");

  int blk_size, grd_size;
  float compareDimError[5][8], compareDimTime[5][8];

#ifdef LAYOUT1
  void (*kernel)(int,dtype*,dtype*,dtype*) = Problem_solution_layout1;
#else
  void (*kernel)(int,dtype*,dtype*,dtype*) = Problem_solution_layout2;
#endif

  for(int i=5; i<10; i++) {
    blk_size = 1 << i;
    for (int j=0; j<8; j++) {
      grd_size = 1 << j;

      checkCudaErrors( cudaMemset(dev_a, 0, len*sizeof(dtype)) );
      checkCudaErrors( cudaMemset(dev_b, 0, len*sizeof(dtype)) );
      checkCudaErrors( cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemset(dev_c, 0, len*sizeof(dtype)) );

      // ------------ computation solution with Layout 2 -----------
      TIMER_START;

      {
          dim3 block_size(blk_size, 1, 1);
          dim3 grid_size(grd_size, 1, 1);
          printf("block_size = %d, grid_size = %d, elements per thread = %f\n", block_size.x, grid_size.x, (float)len/(block_size.x*grid_size.x));
          kernel<<<grid_size, block_size>>>(len, dev_a, dev_b, dev_c);
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP;
      gputime = TIMER_ELAPSED;
      // ----------- copy results from device to host ------------

      checkCudaErrors( cudaMemcpy(GPU_c, dev_c, len*sizeof(dtype), cudaMemcpyDeviceToHost) );

      // ------------- Compare GPU and CPU solution --------------

      error = 0.0f;
      for (int z = 0; z < len; z++)
        error += (float)fabs(CPU_c[z] - GPU_c[z]);

      compareDimError[i-5][j] = error;
      compareDimTime[i-5][j] = gputime;
    }
  }

  char *kernel_string;
  if (kernel == Problem_solution_layout1)
    kernel_string = (char*)"Problem_solution_layout1";
  else if (kernel == Problem_solution_layout2)
    kernel_string = (char*)"Problem_solution_layout2";


  printf("\nVector len = %d, CPU time = %5.5f, kernel = %s\n\n\t", len, cputime, kernel_string);
  for (int j=0; j<8; j++)
    printf("%d\t\t", 1 << j);
  printf("\n");
  for (int i=0; i<5; i++) {
    printf("%d:\t", 1 << (i+5));
    for (int j=0; j<8; j++)
      printf("%.5f(%4.2f)\t", compareDimTime[i][j], compareDimError[i][j]);
    printf("\n");
  }
  printf("\n");


  // ----------------- free GPU variable ---------------------

  checkCudaErrors( cudaFree(dev_a) );
  checkCudaErrors( cudaFree(dev_b) );
  checkCudaErrors( cudaFree(dev_c) );

  // ---------------------------------------------------------

  return(0);
}
