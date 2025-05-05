#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>

#include "../../Laboratory_II/ex5/include/my_time_lib.h"

#define dtype double
#define WARMUP 2
#define NITER 10

// Function to fill a pre-allocated matrix with random values
void fill_matrix(int rows, int cols, dtype *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*cols +j] = (float)(rand() % 100) / 10.0;
        }
    }
}

void init_matrix(int rows, int cols, dtype *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*cols +j] = val;
        }
    }
}

void my_block_gemm(int n, int k, int m, int block_n, dtype *A, dtype *B, dtype *C) {

    for (int blk_i=0; blk_i<n; blk_i+=block_n)
        for (int blk_j=0; blk_j<m; blk_j+=block_n)
            for (int i=0; i<block_n; i ++)
                for (int j=0; j<block_n; j++)
                    for (int blk_h=0; blk_h<k; blk_h+=block_n)
                        for (int h=0; h<block_n; h++)
                            C[(blk_i + i)*m + (blk_j + j)] += A[(blk_i + i)*k + (blk_h + h)] * B[(blk_h + h)*m + (blk_j + j)];
}

void my_gemm(int n, int k, int m, dtype *A, dtype *B, dtype *C) {

    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            for (int h=0; h<k; h++)
                C[i*m +j] += A[i*k +h] * B[h*m +j];
}


int main(int argc, char *argv[]) {

    if (argc < 1) {
        printf("Usage: %s n blk_n\n\nWhere:\tmatrix A has size nxn\n\tmatrix B has size nxn\n\tresult matrix C has size nxn\n\tblock_n is the size of the block multiplication\n", argv[0]);
        return(1);
    }

    // Generate random matrices
    dtype *A, *B, *C;
    double timers[NITER];
    int n = atoi(argv[1]);
    int k = n;
    int m = n;
    int block_n = atoi(argv[2]);
    fprintf(stdout, "Input sizes are %d x %d x %d\n", n, k, m);

    A = (dtype*)malloc(sizeof(dtype)*n*k);
    B = (dtype*)malloc(sizeof(dtype)*k*m);
    C = (dtype*)malloc(sizeof(dtype)*n*m);

    srand(time(NULL));
    fill_matrix(n, k, A);
    fill_matrix(k, m, B);
    init_matrix(n, m, C, 0.0);


    int nflop = n * k * m;;
    double iter_time, a_mean, flops;
    fprintf(stdout, "\nEach gemm required n*k*m = %d floating point operations.\n", nflop);

    TIMER_DEF(0);

    fprintf(stdout, "\nMy GEMM implementation:\n");
    for (int i=-WARMUP; i<NITER; i++) {

        // Perform C = A * B using BLAS
        TIMER_START(0);
        my_block_gemm(n, k, m, block_n, A, B, C);
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        if ( i == -WARMUP ) {
            dtype *test_C = (dtype*)malloc(sizeof(dtype)*n*m);
            init_matrix(n, m, test_C, 0.0);

            int flag = 0;
            my_gemm(n, k, m, A, B, test_C);
            for (int k=0; k<n*m && flag == 0; k++) {
                if (test_C[k] != C[k]) flag = 1;
            }

            if (flag == 0) {
                fprintf(stdout, "Correctness check: PASSED\n");
            } else {
                fprintf(stdout, "Correctness check: ERROR\n");
                fprintf(stderr, "Correctness check: ERROR\n");
            }

            free(test_C);
        }

        printf("Iteration %d tooks %lfs\n", i, iter_time);
        init_matrix(n, m, C, 0.0);
    }

    a_mean = arithmetic_mean(timers, NITER);
    fprintf(stdout, "Arithmetic Mean: %lf\n", a_mean);

    flops = nflop / a_mean;
    fprintf(stdout, "My GEMM implementation acheved %lf MFLOP/s\n", flops / 1.e6);

    return(0);
}

