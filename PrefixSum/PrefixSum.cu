#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "include/helper_cuda.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

#define DBG_CHECK { printf("DBG_CHECK: file %s at line %d\n", __FILE__, __LINE__ ); }
#define DEBUG  // without debug (with random imputs) the kernel does not work

#define NPROBS 11
#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype float

#define BLK_SIZE 256

int verbose;

__global__ void Kernel1_sol (int len, dtype* g_idata, dtype* g_odata){
    __shared__ dtype sdata[BLK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i= blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
        sdata[tid] = g_idata[i];
    else
        sdata[tid] = (dtype)0;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if(tid % (2*s) == 0 /*&& tid+s < len*/){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

__global__ void Kernel2_sol (int len, dtype* g_idata, dtype* g_odata){
    __shared__ dtype sdata[BLK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i= blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        sdata[tid] = g_idata[i];
        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s=1; s < blockDim.x; s *= 2) {
          int index = 2 * s * tid;

          if (index < blockDim.x && index + s < len)
            sdata[index] += sdata[index + s];
          __syncthreads();
        }

        // write result for this block to global mem
        if(tid == 0)
            g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void Kernel3_sol (int len, dtype* g_idata, dtype* g_odata){
    __shared__ dtype sdata[BLK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i= blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        sdata[tid] = g_idata[i];
        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
          if (tid < s && tid + s < len) {
            sdata[tid] += sdata[tid + s];
          }
          __syncthreads();
        }


        // write result for this block to global mem
        if(tid == 0)
            g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void Kernel4_sol (int len, dtype* g_idata, dtype* g_odata){
    __shared__ dtype sdata[BLK_SIZE];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if (i < len) {
        if ((i+blockDim.x)<len)
          sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
        else
          sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }


    // write result for this block to global mem
    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__device__ __forceinline__ void warpReduce(volatile dtype* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}


__global__ void Kernel5_sol (int len, dtype* g_idata, dtype* g_odata){
    __shared__ dtype sdata[BLK_SIZE];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if (i < len) {
        if ((i+blockDim.x)<len)
          sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
        else
          sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid);


    // write result for this block to global mem
    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__device__ __forceinline__ void warpReduce(volatile dtype* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void Kernel6_sol(int len, dtype *g_idata, dtype *g_odata) {
  __shared__ dtype sdata[BLK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  if (i < len) {
    if (i+blockDim.x < len)
      sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    else
      sdata[tid] = g_idata[i];
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();


  if (blockSize >= 1024) {
    if (tid < 512) { sdata[tid] += sdata[tid + 512]; }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
    __syncthreads();
  }

  if (tid < 32) warpReduce<blockSize>(sdata, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__global__ void Kernel7_sol(int len, dtype *g_idata, dtype *g_odata) {
  __shared__ dtype sdata[BLK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;

  sdata[tid] = 0;
  while (i < len) {
    if (i+blockSize < len)
      sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    else
      sdata[tid] += g_idata[i];
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >= 512) {
    if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
    __syncthreads();
  }

  if (tid < 32) warpReduce<blockSize>(sdata, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}


__global__ void Kernel_CUB(int len, dtype *g_idata, dtype *g_odata) {
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<dtype, BLK_SIZE> BlockScan;
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Obtain input item for each thread
    dtype thread_data;
    unsigned int tid = threadIdx.x;
    unsigned int i= blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
        thread_data = g_idata[i];
    else
        thread_data = (dtype)0;
    __syncthreads();

    // Collectively compute the block-wide inclusive prefix sum
    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

    if (tid == BLK_SIZE-1)
      g_odata[blockIdx.x] = thread_data;
}

__global__ void Kernel_CUB2(int len, dtype *g_idata, dtype *g_odata) {
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<dtype, BLK_SIZE> BlockScan;
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Obtain input item for each thread
    dtype thread_data;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    if (i < len) {
      if (i+blockDim.x < len)
        thread_data = g_idata[i] + g_idata[i+blockDim.x];
      else
        thread_data = g_idata[i];
    } else {
      thread_data = 0;
    }
    __syncthreads();

    // Collectively compute the block-wide inclusive prefix sum
    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

    if (tid == BLK_SIZE-1)
      g_odata[blockIdx.x] = thread_data;
}

__global__ void Kernel_CUB3(int len, dtype *g_idata, dtype *g_odata) {
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<dtype, BLK_SIZE> BlockScan;
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Obtain input item for each thread
    dtype thread_data;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = BLK_SIZE*2*gridDim.x;

    thread_data = 0;
    while (i < len) {
      if (i+BLK_SIZE < len)
        thread_data += g_idata[i] + g_idata[i+BLK_SIZE];
      else
        thread_data += g_idata[i];
      i += gridSize;
    }
    __syncthreads();

    // Collectively compute the block-wide inclusive prefix sum
    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

    if (tid == BLK_SIZE-1)
      g_odata[blockIdx.x] = thread_data;
}

dtype execute_reduction_kernel (int len, dtype* a, void(*reduction_kernel)(int,dtype*,dtype*), float* Bandwidth, float* CompTime) {
  int grd_size = len/BLK_SIZE;
  int kernel_len = len;
  dtype *GPU_tmp_b;

  // ------------------- allocating GPU vectors ----------------------
  dtype *dev_a, *dev_tmp_b, *dev_b;

  checkCudaErrors( cudaMalloc(&dev_a, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_tmp_b, grd_size*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_b, sizeof(dtype)) );
  size_t bandwidth_numerator = 0;

  // ----------------- copy date from host to device -----------------

  checkCudaErrors( cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dev_tmp_b, 0, grd_size*sizeof(dtype)) );
  checkCudaErrors( cudaMemset(dev_b, 0, sizeof(dtype)) );

  // ---------- compute GPU_tmp_b with the reduction kernel ----------
  TIMER_DEF;
  TIMER_START;

  {

      dim3 block_size(BLK_SIZE, 1, 1);
      dim3 grid_size(grd_size, 1, 1);
      printf("%d: block_size = %d, grid_size = %d, kernel_len = %d\n", __LINE__, block_size.x, grid_size.x, kernel_len);
      reduction_kernel<<<grid_size, block_size>>>(kernel_len, dev_a, dev_tmp_b);
      bandwidth_numerator += kernel_len*sizeof(dtype);
      kernel_len = grd_size;
      if (reduction_kernel != Kernel1_sol && reduction_kernel != Kernel2_sol && reduction_kernel != Kernel3_sol && reduction_kernel != Kernel_CUB) kernel_len /= 2;
      bandwidth_numerator += kernel_len*sizeof(dtype);
  }


  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  *CompTime += TIMER_ELAPSED;

  if (verbose > 0) {
    GPU_tmp_b = (dtype*) malloc(sizeof(dtype)*kernel_len);
    checkCudaErrors( cudaMemcpy(GPU_tmp_b, dev_tmp_b, kernel_len*sizeof(dtype), cudaMemcpyDeviceToHost) );
    dtype CPUsum_of_GPU_tmp_b = 0.0f;

    if (verbose > 1) printf("GPU_tmp_b: ");
    for (int i=0; i<kernel_len; i++) {
        if (verbose > 1) {
          if (i==0 || GPU_tmp_b[i] != GPU_tmp_b[i-1])
            printf("%f ... ", GPU_tmp_b[i]);
        }
        CPUsum_of_GPU_tmp_b += GPU_tmp_b[i];
    }
    if (verbose > 1) printf("\n");
    printf("CPUsum_of_GPU_tmp_b = %f\n", CPUsum_of_GPU_tmp_b);

    free(GPU_tmp_b);
  }

  while (grd_size > BLK_SIZE) {
    dtype* dev_tmp_b2;
    checkCudaErrors( cudaMalloc(&dev_tmp_b2, sizeof(dtype)*(kernel_len/BLK_SIZE)) );
    checkCudaErrors( cudaMemset(dev_tmp_b2, 0, sizeof(dtype)*(kernel_len/BLK_SIZE)) );
    // ---------- compute GPU_tmp_b with the reduction kernel ----------
    TIMER_START;

    {
        dim3 block_size(BLK_SIZE, 1, 1);
        grd_size = kernel_len / BLK_SIZE;
        dim3 grid_size(grd_size, 1, 1);
        printf("%d: block_size = %d, grid_size = %d, kernel_len = %d\n", __LINE__, block_size.x, grid_size.x, kernel_len);
        reduction_kernel<<<grid_size, block_size>>>(kernel_len, dev_tmp_b, dev_tmp_b2);
        bandwidth_numerator += kernel_len*sizeof(dtype);
        kernel_len = grd_size;
        if (reduction_kernel != Kernel1_sol && reduction_kernel != Kernel2_sol && reduction_kernel != Kernel3_sol && reduction_kernel != Kernel_CUB) kernel_len /= 2;
        bandwidth_numerator += kernel_len*sizeof(dtype);
    }


    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP;
    *CompTime += TIMER_ELAPSED;
    checkCudaErrors( cudaFree(dev_tmp_b) );
    dev_tmp_b = dev_tmp_b2;


    if (verbose > 0) {
        GPU_tmp_b = (dtype*) malloc(sizeof(dtype)*kernel_len);
        checkCudaErrors( cudaMemcpy(GPU_tmp_b, dev_tmp_b, kernel_len*sizeof(dtype), cudaMemcpyDeviceToHost) );
        dtype CPUsum_of_GPU_tmp_b = 0.0f;

        if (verbose > 1) printf("GPU_tmp_b: ");
        for (int i=0; i<kernel_len; i++) {
          if (verbose > 1) {
            if (i==0 || GPU_tmp_b[i] != GPU_tmp_b[i-1])
              printf("%f ... ", GPU_tmp_b[i]);
          }
            CPUsum_of_GPU_tmp_b += GPU_tmp_b[i];
        }
        if (verbose > 1) printf("\n");
        printf("CPUsum_of_GPU_tmp_b = %f\n", CPUsum_of_GPU_tmp_b);

        free(GPU_tmp_b);
    }

  }

  // ------------ compute GPU_b with the reduction kernel ------------
  TIMER_START;

  {
      dim3 block_size(BLK_SIZE, 1, 1);
      dim3 grid_size(1, 1, 1);
      printf("%d: block_size = %d, grid_size = %d, kernel_len = %d\n", __LINE__, block_size.x, grid_size.x, kernel_len);
      reduction_kernel<<<grid_size, block_size>>>(kernel_len, dev_tmp_b, dev_b);
      bandwidth_numerator += (kernel_len+1)*sizeof(dtype);
  }


  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  *CompTime += TIMER_ELAPSED;
  *Bandwidth = bandwidth_numerator / ((*CompTime)*1e+9);

  // --------------- copy results from device to host ----------------

  dtype GPU_b;
  checkCudaErrors( cudaMemcpy(&GPU_b, dev_b, sizeof(dtype), cudaMemcpyDeviceToHost) );

  checkCudaErrors( cudaFree(dev_a) );
  checkCudaErrors( cudaFree(dev_b) );
  checkCudaErrors( cudaFree(dev_tmp_b) );
  return(GPU_b);
}



int main(int argc, char *argv[]) {

  printf("====================================== Problem computations ======================================\n");
// =========================================== Set-up the problem ============================================

  if (argc < 3) {
    printf("Usage: lab3_ex1 n v\n");
    return(1);
  }
  printf("argv[1] = %s\n", argv[1]);
  printf("argv[2] = %s\n", argv[2]);

  // ---------------- set-up the problem size -------------------

  int n = atoi(argv[1]), len = (1<<n), i;
  verbose = atoi(argv[2]);

  printf("n = %d --> len = 2^(n) = %d\n", n, len);
  printf("verbose = %d\n", verbose);
  printf("dtype = %s\n", XSTR(dtype));

//   // BUG check: code to generalize
//   if ((len%BLK_SIZE) != 0) {
//     printf("Now the code only support len (mod BLK_SIZE) == 0 (BLK_SIZE = %d)\n", BLK_SIZE);
//     exit(42);
//   }

  // ------------------ set-up the timers ---------------------

  TIMER_DEF;
  const char* lables[NPROBS] = {"CPU check", "Kernel 1", "Kernel 2", "Kernel 3", "Kernel 4", "Kernel 5", "Kernel 6", "Kernel 7", "CUB test", "CUB test 2", "CUB test 3"};
  float errors[NPROBS], Times[NPROBS], Bandwidths[NPROBS], error;
  for (i=0; i<NPROBS; i++) {
    errors[i] = 1<<30;
    Bandwidths[i] = 0;
    Times[i] = 0;
  }


  // ------------------- set-up the problem -------------------

  dtype *a, GPU_b = 0.0f;
  double CPU_b = 0.0f;
  a = (dtype*)malloc(sizeof(dtype)*len);
  time_t t;
  srand((unsigned) time(&t));


#ifdef ONES_INIT
  for (i=0; i<len; i++)
    a[i] = 0.125f;
#endif

#ifdef DEBUG
  if (verbose > 1) {
    printf("a: ");
    for (i=0; i<len; i++)
        if (i==0 || a[i] != a[i-1])
          printf("%f ... ", a[i]);
    printf("\n");
  }
#endif
  // ======================================== Running the computations =========================================

  /* [ ... ]
   */

  // ========================== CPU computation =========================

  TIMER_START;
  for (i=0; i<len; i++) {
    CPU_b += a[i];
//     if (CPU_b == 16777216.0f) {
//       printf("ERROR: float overflow\n");
//       exit(42);
//     }
  }
  TIMER_STOP;

  errors[0] = 0.0f;
  Bandwidths[0] = 0.0f;
  Times[0] = TIMER_ELAPSED;

  printf("CPU_b = %f\n", CPU_b);

  printf("=========================== GPU Kernel 1 ===========================\n");
  // =========================== GPU Kernel 1 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel1_sol, &Bandwidths[1], &Times[1]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[1] = error;

  printf("GPU_b = %f\n", GPU_b);

  printf("=========================== GPU Kernel 2 ===========================\n");
  // =========================== GPU Kernel 2 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel2_sol, &Bandwidths[2], &Times[2]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[2] = error;

  printf("GPU_b = %f\n", GPU_b);

  printf("=========================== GPU Kernel 3 ===========================\n");
  // =========================== GPU Kernel 3 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel3_sol, &Bandwidths[3], &Times[3]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[3] = error;

  printf("GPU_b = %f\n", GPU_b);

  printf("=========================== GPU Kernel 4 ===========================\n");
  // =========================== GPU Kernel 4 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel4_sol, &Bandwidths[4], &Times[4]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[4] = error;

  printf("GPU_b = %f\n", GPU_b);

    printf("=========================== GPU Kernel 5 ===========================\n");
  // =========================== GPU Kernel 5 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel5_sol, &Bandwidths[5], &Times[5]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[5] = error;

  printf("GPU_b = %f\n", GPU_b);

  printf("=========================== GPU Kernel 6 ===========================\n");
  // =========================== GPU Kernel 6 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel6_sol<BLK_SIZE>, &Bandwidths[6], &Times[6]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[6] = error;

  printf("GPU_b = %f\n", GPU_b);

  printf("=========================== GPU Kernel 7 ===========================\n");
  // =========================== GPU Kernel 7 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel7_sol<BLK_SIZE>, &Bandwidths[7], &Times[7]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[7] = error;

  printf("GPU_b = %f\n", GPU_b);

  printf("=========================== GPU CUB test ===========================\n");
  // =========================== GPU CUB test ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel_CUB, &Bandwidths[8], &Times[8]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[8] = error;

  printf("GPU_b = %f\n", GPU_b);

    printf("=========================== GPU CUB test 2 ===========================\n");
  // =========================== GPU CUB test 2 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel_CUB2, &Bandwidths[9], &Times[9]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[9] = error;

  printf("GPU_b = %f\n", GPU_b);

    printf("=========================== GPU CUB test 3 ===========================\n");
  // =========================== GPU CUB test 3 ===========================

  GPU_b = execute_reduction_kernel(len, a, Kernel_CUB3, &Bandwidths[10], &Times[10]);

  // ------------- Compare GPU and CPU solution --------------

  error = (float)fabs(CPU_b - GPU_b);
  errors[10] = error;

  printf("GPU_b = %f\n", GPU_b);

  printf("Solution\n %9s\t%9s\t%9s\t%9s\n", "type", "error", "time (s)", "bandwidth (GB/s)");
  for (int i=0; i<NPROBS; i++) {
    printf("%12s:\t%9.7f\t%9.7f\t%9.7f\n", lables[i], errors[i], Times[i], Bandwidths[i]);
  }
  printf("\n");

  printf("GPU times: n Kernel1_time Kernel2_time ... on stderr\n");
  fprintf(stderr, "%d, ", n);
  for (i=1; i<NPROBS; i++)
    fprintf(stderr, "%f, ", Times[i]);
  fprintf(stderr, "\n");

  return(0);
}
