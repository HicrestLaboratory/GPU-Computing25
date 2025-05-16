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

#define NPROBS 8
#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype float

#define RUN_SOLUTIONS

#define PRINT_MATRIX(A, N, M, ST ) {  \
      int i, j;  \
      printf("%s:\n", ( ST ));  \
      for (i=0; i< ( N ); i++) {  \
        printf("\t");  \
        for (j=0; j< ( M ); j++)  \
          printf("%6.3f ", A[i*( M ) + j]);  \
        printf("\n");  \
      }  \
      printf("\n\n");  \
}

float matrix_error (int n, int m, const dtype* A, const dtype* B) {
  int i, j;
  dtype error = (dtype)0;
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      error += fabs(B[i*m + j] - A[i*m + j]);

  return(error);
}

#define CEIL_DIV( N, D ) ((( N ) % ( D )) == 0) ? (( N )/( D )) : ((( N )/( D ))+1)

#define BLK_EDGE 32     // sgemm_naive
#define BLOCKSIZE 32    // sgemm_global_memory_coalescing, sgemm_shared_memory_cache_blocking

// --------------------- sgemm_block_tiling

#define BN 64
#define BK 16
#define BM 64

#define TM 4

// ------------------ sgemm_2D_block_tiling

#define BN2D 128
#define BK2D 16
#define BM2D 128
#define TM2D 4
#define TN2D 4

// ----------------------- sgemm_warptiling

#define BNWARP 128
#define BKWARP 8
#define BMWARP 128
#define TMWARPS 2
#define TNWARPS 2

#define WARPNUM 32
#define WARPSN 4
#define WARPSM 8
#define WN 32
#define WM 16

#define WARPSIZE 32
#define WSUBN 8
#define WSUBM 4
#define WNITER 2
#define WMITER 2

// ----------------------------------------


int verbose;

__global__ void sgemm_naive(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < N && y < M) {
    dtype tmp = 0.0;
    for (int i = 0; i < K; ++i)
      tmp += A[x * K + i] * B[i * M + y];

    // C = α*(A@B)+β*C
    C[x * M + y] = alpha * tmp + beta * C[x * M + y];
  }
}

__global__ void sgemm_global_memory_coalescing(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {
  // compute position in C that this thread is responsible for
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < N && y < M) {
    dtype tmp = 0.0;
    for (int i = 0; i < K; ++i)
      tmp += A[x * K + i] * B[i * M + y];

    // C = α*(A@B)+β*C
    C[x * M + y] = alpha * tmp + beta * C[x * M + y];
  }
}

__global__ void sgemm_shared_memory_cache_blocking(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {

  int cRow = blockIdx.x;
  int cCol = blockIdx.y;
  int threadRow = threadIdx.x / BLOCKSIZE;
  int threadCol = threadIdx.x % BLOCKSIZE;
  __shared__ dtype As[BLOCKSIZE*BLOCKSIZE], Bs[BLOCKSIZE*BLOCKSIZE];

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * M + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  // the outer loop advances A along the columns and B along
  // the rows until we have fully calculated the result in C.
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B from
    // global memory into shared memory.
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * M + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();

    // advance pointers onto next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * M;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
      tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }

  C[threadRow * M + threadCol] = alpha * tmp + beta * C[threadRow * M + threadCol];

}

__global__ void sgemm_block_tiling(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {

  int cRow = blockIdx.x;
  int cCol = blockIdx.y;
//   ------------------------
  int threadRow = threadIdx.x / BK / TM;
  int threadCol = threadIdx.x % BM;
//   ------------------------
  int innerRowA = threadIdx.x / BK;
  int innerColA = threadIdx.x % BK;
  int innerRowB = threadIdx.x / BM;
  int innerColB = threadIdx.x % BM;
//   ------------------------
  __shared__ dtype As[BN*BK], Bs[BK*BM];

  // advance pointers to the starting positions
  A += cRow * BN * K;                    // row=cRow, col=0
  B += cCol * BM;                        // row=0, col=cCol
  C += (cRow * BN * M) + (cCol * BM);    // row=cRow, col=cCol


  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches (same as before)
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BM + innerColB] = B[innerRowB * M + innerColB];
    __syncthreads();

    // advance blocktile for outer loop
    A += BK;
    B += BK * M;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float Btmp = Bs[dotIdx * BM + threadCol];
      for (uint resIdx = 0; resIdx < TM; resIdx++)
        threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
    __syncthreads();

  }

  // TEST OK --------------------------------
//   if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
//     printf("As:\n");
//     for (int i=0; i<BN; i++) {
//       for (int j=0; j<BK; j++)
//         printf("%f ", As[i * BK + j]);
//       printf("\n");
//     }
//     printf("Bs:\n");
//     for (int i=0; i<BM; i++) {
//       for (int j=0; j<BK; j++)
//         printf("%f ", Bs[j * BM + i]);
//       printf("\n");
//     }
//     printf("\n");
//   }
//   __syncthreads();
  // ----------------------------------------

  // TEST OK --------------------------------
//   if (blockIdx.x == 0 && blockIdx.y == 0) {
//     for (int th=0; th<blockDim.x; th++) {
//       if (th == threadIdx.x) {
//         printf("threadResults[%d]:\t", threadIdx.x);
//         for (int i=0; i<TM; i++) {
//           printf("%f ", threadResults[i]);
//         }
//         printf("\n");
//       }
//       __syncthreads();
//     }
//   }
//   __syncthreads();
  // ----------------------------------------

// Loop over threadResults
  for (uint resIdx = 0; resIdx < TM; resIdx++) {
    C[(threadRow * TM + resIdx) * M + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * M + threadCol];
//     C[(threadRow * TM + resIdx) * M + threadCol] = threadIdx.x; // NOTE test BUG
  }

}



__global__ void sgemm_2D_block_tiling(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {

  int cRow = blockIdx.x;
  int cCol = blockIdx.y;
//   ------------------------
  int threadRow = threadIdx.x / (BM2D / TM2D);
  int threadCol = threadIdx.x % (BM2D / TM2D);
//   ------------------------
  int innerRowA = threadIdx.x / BK2D;
  int innerColA = threadIdx.x % BK2D;
  int innerRowB = threadIdx.x / BM2D;
  int innerColB = threadIdx.x % BM2D;
//   ------------------------
  int strideA = blockDim.x / BK2D;
  int strideB = blockDim.x / BM2D;
//   ------------------------
  __shared__ dtype As[BN2D*BK2D], Bs[BK2D*BM2D];

  // advance pointers to the starting positions
  A += cRow * BN2D * K;                    // row=cRow, col=0
  B += cCol * BM2D;                        // row=0, col=cCol
  C += (cRow * BN2D * M) + (cCol * BM2D);    // row=cRow, col=cCol


  // allocate thread-local cache for results in registerfile
  float threadResults[TM2D * TN2D] = {0.0};
  // register caches for As and Bs
  float regM[TM2D] = {0.0};
  float regN[TN2D] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK2D) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BN2D; loadOffset += strideA)
      As[(innerRowA + loadOffset) * BK2D + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
    for (uint loadOffset = 0; loadOffset < BK2D; loadOffset += strideB)
      Bs[(innerRowB + loadOffset) * BM2D + innerColB] = B[(innerRowB + loadOffset) * M + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK2D;     // move BK2D columns to right
    B += BK2D * M; // move BK2D rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK2D; dotIdx++) {
      // load relevant As & Bs entries into registers
      for (uint i = 0; i < TN2D; ++i)
        regN[i] = As[(threadRow * TN2D + i) * BK2D + dotIdx];
      for (uint i = 0; i < TM2D; ++i)
        regM[i] = Bs[dotIdx * BM2D + threadCol * TM2D + i];

      // perform outer product on register cache, accumulate
      // into threadResults
      for (uint resIdxM = 0; resIdxM < TM2D; resIdxM++)
        for (uint resIdxN = 0; resIdxN < TN2D; resIdxN++)
          threadResults[resIdxN * TN2D + resIdxM] += regM[resIdxM] * regN[resIdxN];
    }
    __syncthreads();

  }

  for (uint resIdxM = 0; resIdxM < TM2D; resIdxM++)
    for (uint resIdxN = 0; resIdxN < TN2D; resIdxN++)
      C[(threadRow * TN2D + resIdxN) * M + threadCol * TM2D + resIdxM] = alpha * threadResults[resIdxN * TN2D + resIdxM] + beta * C[(threadRow * TN2D + resIdxN) * M + threadCol * TM2D + resIdxM];

}

// BUG
__global__ void sgemm_vectorize_SMEM(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {
/*
  int cRow = blockIdx.x;
  int cCol = blockIdx.y;
//   ------------------------
  int threadRow = threadIdx.x / (BM / TM);
  int threadCol = threadIdx.x % (BM / TM);
//   ------------------------
  int innerRowA = threadIdx.x / BK;
  int innerColA = (threadIdx.x % BK) / 4;
  int innerRowB = threadIdx.x / BM;
  int innerColB = (threadIdx.x % BM) / 4;
//   ------------------------
  int strideA = blockDim.x / BK;
  int strideB = blockDim.x / BM;
//   ------------------------
  __shared__ dtype As[BN*BK], Bs[BK*BM];

  // advance pointers to the starting positions
  A += cRow * BN * K;                    // row=cRow, col=0
  B += cCol * BM;                        // row=0, col=cCol
  C += (cRow * BN * M) + (cCol * BM);    // row=cRow, col=cCol


  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    float4 tmp =
        reinterpret_cast<float4*>((float4*)&A[innerRowA * K + innerColA * 4])[0];
    // transpose A during the GMEM to SMEM transfer
    As[(innerColA * 4 + 0) * BN + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BN + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BN + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BN + innerRowA] = tmp.w;

// -------------------- TEST --------------------
    reinterpret_cast<float4*>((float4*)&Bs[innerRowB * BM + innerColB * 4])[0] =
        reinterpret_cast<float4*>((float4*)&B[innerRowB * M + innerColB * 4])[0];
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//     for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB)
//       Bs[(innerRowB + loadOffset) * BM + innerColB*4] = B[(innerRowB + loadOffset) * M + innerColB*4];
// ----------------------------------------------
    __syncthreads();


    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * M; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // load relevant As & Bs entries into registers
      for (uint i = 0; i < TN; ++i)
        regN[i] = As[dotIdx * BN + (threadRow * TN + i)];
      for (uint i = 0; i < TM; ++i)
        regM[i] = Bs[dotIdx * BM + (threadCol * TM + i)];

      // perform outer product on register cache, accumulate
      // into threadResults
      for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
        for (uint resIdxN = 0; resIdxN < TN; resIdxN++) {
          threadResults[resIdxN * TM + resIdxM] += regM[resIdxM] * regN[resIdxN];
        }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
    for (uint resIdxN = 0; resIdxN < TN; resIdxN++) {
      int C_index = (threadRow * TN + resIdxN) * M + threadCol * TM + resIdxM;
      C[C_index] = alpha * threadResults[resIdxN * TN + resIdxM] + beta * C[C_index];
//       C[C_index] = C_index;  // NOTE OK
      C[C_index] = (innerRowA+1);  // BUG 8 righe e solo 2 colonne (OK, innerColA è diviso per 4)
//       C[C_index] = (innerRowB+1) +(innerColB+1)*0.001;  // BUG a cazzo di cane
    }
*/
}



__global__ void sgemm_warptiling(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {

  int cRow = blockIdx.x;
  int cCol = blockIdx.y;
//   ------------------------
  int innerRowA = threadIdx.x / BKWARP;
  int innerColA = threadIdx.x % BKWARP;
  int innerRowB = threadIdx.x / BMWARP;
  int innerColB = threadIdx.x % BMWARP;
//   ------------------------

  if (warpSize != WARPSIZE) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
      printf("ERROR: The build-in warpSize (%d) and the macro defined WARPSIZE (%d) are different [code line %d]\n", warpSize, WARPSIZE, __LINE__);
    asm("trap;");
  }

  int warpNum  = threadIdx.x / warpSize;
  int warpRow = warpNum / (BMWARP / WM);
  int warpCol = warpNum % (BMWARP / WM);

  int warpId  = threadIdx.x % warpSize;
  int threadRowInWarp = warpId / WSUBM;
  int threadColInWarp = warpId % WSUBM;


//   if (blockIdx.x == 0 && blockIdx.y == 0)
//     printf("[%d]: threadIdx.x = %d, warpNum = %d, warpId = %d, warpRow = %d, warpCol = %d, threadRowInWarp = %d, threadColInWarp = %d\n", __LINE__, threadIdx.x, warpNum, warpId, warpRow, warpCol, threadRowInWarp, threadColInWarp);
//   ------------------------
  __shared__ dtype As[BNWARP*BKWARP], Bs[BKWARP*BMWARP];

  // advance pointers to the starting positions
  A += cRow * BNWARP * K;                    // row=cRow, col=0
  B += cCol * BMWARP;                        // row=0, col=cCol
  C += (cRow * BNWARP * M) + (cCol * BMWARP);    // row=cRow, col=cCol

//   if (threadIdx.x == 0)
//     printf("[%d] cRow = %d, cCol = %d, Cincrement = %d\n", __LINE__, cRow, cCol, (cRow * BNWARP * M) + (cCol * BMWARP));

  // allocate thread-local cache for results in registerfile
  dtype threadResults[(WNITER * TNWARPS) * (WMITER * TMWARPS)] = {0.0};

// TEST OK
//   for (int i=0; i<(WNITER * TNWARPS) * (WMITER * TMWARPS); i++)
//     threadResults[i] = (dtype)warpNum;

  // register caches for As and Bs
  dtype regM[(WMITER * TMWARPS)] = {0.0};
  dtype regN[(WNITER * TNWARPS)] = {0.0};

  // BUG in this cycle
  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BKWARP) {
    // populate the SMEM caches
    // --------------------------------------------------------------------------------------------------
//     for (uint loadOffset = 0; loadOffset < BNWARP; loadOffset += strideA)
//       As[(innerRowA + loadOffset) * BKWARP + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
//     for (uint loadOffset = 0; loadOffset < BKWARP; loadOffset += strideB)
//       Bs[(innerRowB + loadOffset) * BMWARP + innerColB] = B[(innerRowB + loadOffset) * M + innerColB];
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    As[innerRowA * BKWARP + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BMWARP + innerColB] = B[innerRowB * M + innerColB];
    // --------------------------------------------------------------------------------------------------
    __syncthreads();

    // TEST OK --------------------------------
//     if (threadIdx.x == 0 && bkIdx == 0) {
//       printf("As:\n");
//       for (int i=0; i<BNWARP; i++) {
//         for (int j=0; j<BKWARP; j++)
//           printf("%f ", As[i * BKWARP + j]);
//         printf("\n");
//       }
//       printf("Bs:\n");
//       for (int i=0; i<BMWARP; i++) {
//         for (int j=0; j<BKWARP; j++)
//           printf("%f ", Bs[j * BMWARP + i]);
//         printf("\n");
//       }
//       printf("\n");
//     }
//     __syncthreads();
    // ----------------------------------------

    // advance blocktile
    A += BKWARP;     // move BKWARP columns to right
    B += BKWARP * M; // move BKWARP rows down


    // dotIdx loops over contents of SMEM
    for (uint dotIdx = 0; dotIdx < BKWARP; dotIdx++) {

      // populate registers for this thread's part of the warptile
      for (uint wSubRowIdx = 0; wSubRowIdx < WNITER; wSubRowIdx++)
        for (uint i = 0; i < TNWARPS; ++i) {
          regN[wSubRowIdx * TNWARPS + i] =
              As[(warpRow * WN + wSubRowIdx * (WSUBN*TNWARPS) + threadRowInWarp * TNWARPS + i) * BKWARP + dotIdx ];  // BUG
//               threadIdx.x; // (TEST N)
        }

      for (uint wSubColIdx = 0; wSubColIdx < WMITER; wSubColIdx++)
        for (uint i = 0; i < TMWARPS; ++i) {
          regM[wSubColIdx * TMWARPS + i] =
              Bs[(dotIdx * BMWARP) + warpCol * WM + wSubColIdx * (WSUBM*TMWARPS) + threadColInWarp * TMWARPS + i];  // BUG
//               threadIdx.x; // (TEST M)
        }

      // TEST OK --------------------------------
//       if (threadIdx.x == 0 && bkIdx == 0) {
//         printf("[%d] regN:\t", dotIdx);
//         for (uint wSubRowIdx = 0; wSubRowIdx < WNITER; wSubRowIdx++)
//           for (uint i = 0; i < TNWARPS; ++i)
//             printf("%f ", regN[wSubRowIdx * TNWARPS + i]);
//         printf("\n");
//       }
//       if (threadIdx.x == 0 && bkIdx == 0) {
//         printf("[%d] regM:\t", dotIdx);
//         for (uint wSubColIdx = 0; wSubColIdx < WMITER; wSubColIdx++)
//           for (uint i = 0; i < TMWARPS; ++i)
//             printf("%f ", regM[wSubColIdx * TMWARPS + i]);
//         printf("\n\n");
//       }
//       __syncthreads();
      // ----------------------------------------


      // execute warptile matmul. Later this will map well to
      // warp-wide matrix instructions, executed on tensor cores.
      for (uint wSubRowIdx = 0; wSubRowIdx < WNITER; wSubRowIdx++)
        for (uint wSubColIdx = 0; wSubColIdx < WMITER; wSubColIdx++)
          // calculate per-thread results with register-cache locality
          for (uint resIdxM = 0; resIdxM < TMWARPS; resIdxM++)
            for (uint resIdxN = 0; resIdxN < TNWARPS; resIdxN++) {
              // ------ BUG -------
              threadResults[(wSubRowIdx * TNWARPS + resIdxN) * (WMITER * TMWARPS) + (wSubColIdx * TMWARPS) + resIdxM] +=
                  regM[wSubColIdx * TNWARPS + resIdxM] * regN[wSubRowIdx * TMWARPS + resIdxN];
// TEST
//                threadResults[(wSubRowIdx * TNWARPS + resIdxN) * (WMITER * TMWARPS) + (wSubColIdx * TMWARPS) + resIdxM] =
//                   (dtype)warpNum;                              // NOTE OK
//                   regM[wSubColIdx * TNWARPS + resIdxM];        // NOTE OK (TEST M)
//                   regN[wSubRowIdx * TMWARPS + resIdxN];        // NOTE OK (TEST N)
//                   (wSubRowIdx * TNWARPS + resIdxM) * (WMITER * TMWARPS) + (wSubColIdx * TMWARPS) + resIdxN;  // NOTE OK
            }

      // TEST OK --------------------------------
//       if (threadIdx.x == 0 && bkIdx == 0) {
//         printf("[%d] threadResults:\n", dotIdx);
//         for (int i=0; i<(WNITER * TNWARPS); i++) {
//           for (int j=0; j<(WMITER * TMWARPS); j++)
//               printf("%f ", threadResults[i * (WMITER * TMWARPS) + j]);
//           printf("\n");
//         }
//         printf("\n\n");
//       }
//       __syncthreads();
      // ----------------------------------------
    }
  }


//   if (warpNum == 1) {
  for (uint wSubRowIdx = 0; wSubRowIdx < WNITER; wSubRowIdx++) {
    for (uint wSubColIdx = 0; wSubColIdx < WMITER; wSubColIdx++) {
      // calculate per-thread results with register-cache locality
      for (uint resIdxN = 0; resIdxN < TNWARPS; resIdxN++) {
        for (uint resIdxM = 0; resIdxM < TMWARPS; resIdxM++) {
          int result_row = wSubRowIdx * TNWARPS + resIdxN;
          int result_col = wSubColIdx * TMWARPS + resIdxM;
          int result_index = result_row * (WMITER * TMWARPS) + result_col;
          dtype tmp = threadResults[result_index];

          int C_row = warpRow * WN + wSubRowIdx * (WSUBN*TNWARPS) + threadRowInWarp * TNWARPS + resIdxN;
          int C_col = warpCol * WM + wSubColIdx * (WSUBM*TMWARPS) + threadColInWarp * TMWARPS + resIdxM;
          int C_index = (C_row) * M + (C_col);

          C[C_index] = alpha * tmp + beta * C[C_index];
//           C[C_index] = C_row + C_col * 0.001;    // NOTE OK
//           C[C_index] = C_index;                  // NOTE OK
//           C[C_index] = warpNum;                  // NOTE OK
//           C[C_index] = threadIdx.x;              // NOTE OK
//           C[C_index] = result_index+1;           // NOTE OK
//           C[C_index] = tmp;                      // NOTE OK
        }
      }
    }
  }
//   }
}

dtype* execute_gemm_kernel (int n, int k, int m, float alpha, dtype* A, dtype* B, float beta, void (*gemm_kernel)(int, int, int, float, const dtype*, const dtype*, float, dtype*), float* Bandwidth, float* CompTime, double* Flops) {
  int grd_sizeX, grd_sizeY;
  int blk_sizeX, blk_sizeY;

  // ---------------------------------
  if (gemm_kernel == sgemm_naive) {
    grd_sizeX = CEIL_DIV(n, BLK_EDGE);
    grd_sizeY = CEIL_DIV(m, BLK_EDGE);

    blk_sizeX = BLK_EDGE;
    blk_sizeY = BLK_EDGE;
  } else {
    if (gemm_kernel == sgemm_global_memory_coalescing || gemm_kernel == sgemm_shared_memory_cache_blocking) {
      grd_sizeX = CEIL_DIV(n, BLOCKSIZE);
      grd_sizeY = CEIL_DIV(m, BLOCKSIZE);

      blk_sizeX = BLOCKSIZE * BLOCKSIZE;
      blk_sizeY = 1;
    } else {
      if (gemm_kernel == sgemm_block_tiling) {
        if ( ((BN * BM) / TM) != (BN * BK) ) {
          fprintf(stderr, "[%d] ERROR: ((BN * BM) / TM) = %d != %d = (BN * BK)\n", __LINE__, ((BN * BM) / TM), (BN * BK));
          exit(42);
        }

        grd_sizeX = CEIL_DIV(n, BN);
        grd_sizeY = CEIL_DIV(m, BM);

        blk_sizeX = (BN * BM) / TM;
        blk_sizeY = 1;
      } else {
        if (gemm_kernel == sgemm_2D_block_tiling || gemm_kernel == sgemm_vectorize_SMEM) {
          grd_sizeX = CEIL_DIV(n, BN2D);
          grd_sizeY = CEIL_DIV(m, BM2D);

          blk_sizeX = (BN2D / TN2D) * (BM2D / TM2D);
          blk_sizeY = 1;
        } else {
          if (gemm_kernel == sgemm_warptiling) {
            grd_sizeX = CEIL_DIV(n, BNWARP);
            grd_sizeY = CEIL_DIV(m, BMWARP);

            blk_sizeX = BNWARP * BKWARP;
            blk_sizeY = 1;

            unsigned int flag = 0U, e = 0U;
            (WARPNUM * 32 == blk_sizeX)           ? (e &= (1U<<0)) : (flag &= (1U<<0));

            ((WARPSN * WARPSM) == WARPNUM)        ? (e &= (1U<<1)) : (flag &= (1U<<1));
            ((WSUBN * WNITER) == WN)              ? (e &= (1U<<2)) : (flag &= (1U<<2));
            ((WN * WARPSN) == BNWARP)             ? (e &= (1U<<3)) : (flag &= (1U<<3));

            ((WSUBM * WMITER) == WM)              ? (e &= (1U<<4)) : (flag &= (1U<<4));
            ((WM * WARPSM) == BMWARP)             ? (e &= (1U<<5)) : (flag &= (1U<<5));

            if (flag != 0U) {
              fprintf(stderr, "ERROR: line %d, e = %u, flag = %u\n", __LINE__, e, flag);
              fprintf(stderr, "ERROR: WARPSN = %d, WSUBN = %d, WNITER = %d, WN = %d, WARPSM = %d, WSUBM = %d, WMITER = %d, WM = %d\n", WARPSN, WSUBN, WNITER, WN, WARPSM, WSUBM, WMITER, WM);
              exit(42);
            }
          }
        }
      }
    }
  }
  // ---------------------------------

  // ------------------- allocating GPU vectors ----------------------
  dtype *dev_A, *dev_B, *dev_C;

  checkCudaErrors( cudaMalloc(&dev_A, n*k*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_B, k*m*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_C, n*m*sizeof(dtype)) );
  size_t bandwidth_numerator = ((n*k) + (k*m) + (n*m))*sizeof(dtype);

  // ----------------- copy date from host to device -----------------

  checkCudaErrors( cudaMemcpy(dev_A, A, n*k*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dev_B, B, k*m*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dev_C, 0, n*m*sizeof(dtype)) );

  // ---------- compute GPU_tmp_b with the reduction kernel ----------
  TIMER_DEF;
  TIMER_START;

  {
      dim3 block_size(blk_sizeX, blk_sizeY, 1);
      dim3 grid_size(grd_sizeX, grd_sizeY, 1);
      printf("%d: block_size = (%d, %d), grid_size = (%d, %d)\n", __LINE__, block_size.x, block_size.y, grid_size.x, grid_size.y);
      gemm_kernel<<<grid_size, block_size>>>(n, k, m, alpha, (const dtype*)dev_A, (const dtype*)dev_B, beta, dev_C);
  }


  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  *CompTime += TIMER_ELAPSED;
  *Bandwidth = bandwidth_numerator / ((*CompTime)*1e+9);
  *Flops  = (n) / ((*CompTime)*1e+9);
  *Flops *= m;
  *Flops *= k;
  *Flops *= 2;

  // --------------- copy results from device to host ----------------

  dtype *GPU_C = (dtype*)malloc(sizeof(dtype)*n*m);
  checkCudaErrors( cudaMemcpy(GPU_C, dev_C, n*m*sizeof(dtype), cudaMemcpyDeviceToHost) );

  if (verbose > 1)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C form execute_gemm_kernel")

  checkCudaErrors( cudaFree(dev_A) );
  checkCudaErrors( cudaFree(dev_B) );
  checkCudaErrors( cudaFree(dev_C) );

  return(GPU_C);
}


int main(int argc, char *argv[]) {

  printf("====================================== Problem computations ======================================\n");
  // =========================================== Set-up the problem ============================================

  if (argc < 3) {
    printf("Usage: lab3_ex1 e v [CPU_ON = 1]\n");
    return(1);
  }
  printf("argv[1] = %s\n", argv[1]);
  printf("argv[2] = %s\n", argv[2]);
  if (argc > 3)
    printf("argv[3] = %s\n", argv[3]);

  // ---------------- set-up the problem size -------------------


  int e = atoi(argv[1]), n = (1<<(e/2)), k = n, m = n, i, j, CPU_ON = 1;
  float alpha = 1.0f, beta = 1.0f;
  verbose = atoi(argv[2]);
  if (argc > 3)
    CPU_ON = atoi(argv[3]);

  // BUG check: code to generalize
  if ((e%2) != 0) {
    printf("Now the code only support squared matrices. So, since the generated matrix will have dimensions 2^(e/2) x 2^(e/2), e must be even\n");
    exit(42);
  }

  printf("e = %d --> n = k = m = 2^(e/2) = %d\n", e, n);
  printf("alpha = %f, beta = %f\n", alpha, beta);
  printf("CPU_ON = %d\n", CPU_ON);
  printf("verbose = %d\n", verbose);
  printf("dtype = %s\n", XSTR(dtype));

  // ======================================== Get the device properties ========================================
  printf("======================================= Device properties ========================================\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  int dev;
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

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);

    printf("  Peak Arithmetic Intensity:                     %7.3f GFLOPS/s\n",
           2.0*deviceProp.memoryClockRate*(_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount)/1.0e6);
    printf("  Shared Memory per SM:                     %d B\n", deviceProp.sharedMemPerMultiprocessor);
  }

  // ------------------ set-up the timers ---------------------

  TIMER_DEF;
  const char* lables[NPROBS] = {"CPU check", "Kernel 1", "Kernel 2", "Kernel 3", "Kernel 4", "Kernel 5", "Kernel 6", "Kernel '10'"};
  float errors[NPROBS], Times[NPROBS], Bandwidths[NPROBS], error;
  double Flops[NPROBS];
  for (i=0; i<NPROBS; i++) {
    errors[i] = 1<<30;
    Bandwidths[i] = 0;
    Flops[i] = 0;
    Times[i] = 0;
  }


  // ------------------- set-up the problem -------------------

  dtype *A, *B, *GPU_C, *CPU_C;
  A = (dtype*)malloc(sizeof(dtype)*n*k);
  B = (dtype*)malloc(sizeof(dtype)*k*m);
  CPU_C = (dtype*)malloc(sizeof(dtype)*n*m);
//   GPU_C = (dtype*)malloc(sizeof(dtype)*n*m);

  time_t t;
  srand((unsigned) time(&t));


  for (i=0; i<(n*k); i++)
    A[i] = ((dtype)(i/m)/(dtype)m) + 1.0f;
  for (i=0; i<(k*m); i++)
    B[i] = (dtype)(1);

#ifdef DEBUG
  if (verbose > 0) {
    PRINT_MATRIX(A, n, k, "A")

    PRINT_MATRIX(B, k, m, "B")
  }
#endif
  // ======================================== Running the computations =========================================

  /* [ ... ]
   */

  // ========================== CPU computation =========================
  if (CPU_ON) {

    TIMER_START;
    for (i=0; i<n; i++)
      for (j=0; j<m; j++)
        for (int h=0; h<k; h++)
          CPU_C[i*m +j] += A[i*k + h] * B[h*m + j];
    TIMER_STOP;

    Times[0] = TIMER_ELAPSED;
    errors[0] = 0.0f;
    Bandwidths[0] = 0.0f;
    Flops[0]  = (n) / (Times[0]*1e+9);
    Flops[0] *= m;
    Flops[0] *= k;


    if (verbose > 0)
      PRINT_MATRIX(CPU_C, n, m, "CPU_C")

  } else {
    Times[0] = -1.0f;
    errors[0] = -1.0f;
    Bandwidths[0] = -1.0f;
    Flops[0] = -1.0f;
  }
  printf("=========================== GPU Kernel 1 ===========================\n");
  // =========================== GPU Kernel 1 ===========================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_naive, &Bandwidths[1], &Times[1], &Flops[1]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[1] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);

  printf("=========================== GPU Kernel 2 ===========================\n");
  // =========================== GPU Kernel 2 ===========================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_global_memory_coalescing, &Bandwidths[2], &Times[2], &Flops[2]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[2] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);

    printf("=========================== GPU Kernel 3 ===========================\n");
  // =========================== GPU Kernel 3 ===========================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_shared_memory_cache_blocking, &Bandwidths[3], &Times[3], &Flops[3]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[3] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);

    printf("=========================== GPU Kernel 4 ===========================\n");
  // =========================== GPU Kernel 4 ===========================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_block_tiling, &Bandwidths[4], &Times[4], &Flops[4]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[4] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);

    printf("=========================== GPU Kernel 5 ===========================\n");
  // =========================== GPU Kernel 5 ===========================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_2D_block_tiling, &Bandwidths[5], &Times[5], &Flops[5]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[5] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);

/*
    printf("=========================== GPU Kernel 6 ===========================\n");
  // =========================== GPU Kernel 6 ===========================

#ifdef RUN_SOLUTIONS

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_vectorize_SMEM, &Bandwidths[6], &Times[6], &Flops[6]);

#else

#endif

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[6] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);
*/

  printf("=========================== GPU Kernel 7 ===========================\n");
  // =========================== GPU Kernel 7 ===========================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_warptiling, &Bandwidths[7], &Times[7], &Flops[7]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[7] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);


  printf("\n\n");
  if (!(CPU_ON)) printf("CPU check not lunched!!\n");
  printf("Solution\n %9s\t%9s\t%9s\t%16s\t%16s\n", "type", "error", "time (s)", "flops (GFLOPS/s)", "bandwidth (GB/s)");
  for (int i=0; i<NPROBS; i++) {
    if ((i != 6))
      printf("%12s:\t%9.6f\t%9.6f\t%16.6lf\t%16.6f\n", lables[i], errors[i], Times[i], Flops[i], Bandwidths[i]);
  }
  printf("\n");

  printf("GPU times: e Kernel1_time Kernel1_flops Kernel2_time Kernel2_flops ... on stderr\n");
  fprintf(stderr, "%d\t", e);
  for (i=1; i<NPROBS; i++)
    fprintf(stderr, "%f\t%f\t", Times[i], Flops[i]);
  fprintf(stderr, "\n");

  return(0);
}
