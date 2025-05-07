/* ----------------------- Original code-base -----------------------
 *  This code simplify the CUDA matrix transposition 
 *  code publically available at:
 *            https://github.com/Martina-Scheffler/GPU-Computing.git
 * ------------------------------------------------------------------
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include "cublas_v2.h"
#include "include/matrix_generation.h"

using namespace std;

#define NUM_REPS 10

int tileDimension = 4;
int blockRows = 1;

__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia = 0.0f, ib = 0.0f;
    ib += ia + tid; 
}

__global__ void transposeSimple(int* A, int* A_T, int tileDimension, int blockRows){
    int x = blockIdx.x * tileDimension + threadIdx.x;
    int y = blockIdx.y * tileDimension + threadIdx.y;
    int width = gridDim.x * tileDimension;

    for(int i = 0; i < tileDimension; i += blockRows){
        A_T[x * width + (y + i)] = A[(y + i) * width + x];
    }
}

bool checkCorrectness(int* A, int* A_T, int size){
    float* res = (float*) malloc(size * size * sizeof(float));
    float* A_copy = (float*) malloc(size * size * sizeof(float));

    float *dev_A_check, *dev_A_T_check;
    cudaMalloc(&dev_A_check, size * size * sizeof(float));
    cudaMalloc(&dev_A_T_check, size * size * sizeof(float));

    for (int i = 0; i < size * size; i++) A_copy[i] = (float) A[i];

    cudaMemcpy(dev_A_check, A_copy, size * size * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, &alpha, dev_A_check, size, &beta, dev_A_check, size, dev_A_T_check, size);
    cublasDestroy(handle);

    cudaMemcpy(res, dev_A_T_check, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < size * size; i++){
        if (A_T[i] != (int) res[i]) correct = false;
    }

    cudaFree(dev_A_check);
    cudaFree(dev_A_T_check);
    free(A_copy);
    free(res);

    return correct;
}

int main(int argc, char* argv[]){
    if (argc < 2){
        throw runtime_error("Please enter an integer N to generate a matrix of size 2^N x 2^N.");
    }

    if (argc >= 4){
        tileDimension = atoi(argv[2]);
        blockRows = atoi(argv[3]);
    }

    int size = pow(2, atoi(argv[1]));
    int N = size * size;

    cout << "Size: " << size << endl;

    int* A = generate_continous_matrix(size);
    int* A_T = (int*) malloc(N * sizeof(int));

    dim3 nBlocks(size / tileDimension, size / tileDimension);
    dim3 nThreads(tileDimension, blockRows);

    cout << "Blocks: " << nBlocks.x << " x " << nBlocks.y << endl;
    cout << "Threads: " << nThreads.x << " x " << nThreads.y << endl;

    int *dev_A, *dev_A_T;
    cudaMalloc(&dev_A, N * sizeof(int));
    cudaMalloc(&dev_A_T, N * sizeof(int));
    cudaMemcpy(dev_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    warm_up_gpu<<<nBlocks, nThreads>>>();

    cudaEventRecord(start);
    for (int l = 0; l < NUM_REPS; l++){
        transposeSimple<<<nBlocks, nThreads>>>(dev_A, dev_A_T, tileDimension, blockRows);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= NUM_REPS;

    printf("Kernel Time (avg): %f ms\n", milliseconds);

    cudaMemcpy(A_T, dev_A_T, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (!checkCorrectness(A, A_T, size)){
        printf("Incorrect Result!\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_A);
    cudaFree(dev_A_T);
    free(A);
    free(A_T);

    return 0;
}

