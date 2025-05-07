#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP 2
#define NITER 10

#include "../Laboratory_II/ex5/include/my_time_lib.h"

typedef struct {
    int row;
    int col;
    float val;
} COOEntry;

// Function to generate random sparse matrix in COO format
void generate_random_coo(COOEntry *coo, int nnz, int rows, int cols) {
    for (int i = 0; i < nnz; ++i) {
        coo[i].row = rand() % rows;
        coo[i].col = rand() % cols;
        coo[i].val = (float)(rand() % 10 + 1);
    }
}

// Function to convert COO to CSR
void coo_to_csr(COOEntry *coo, int *row_ptr, int *col_idx, float *csr_val, int nnz, int rows) {
    for (int i = 0; i <= rows; ++i)
        row_ptr[i] = 0;

    for (int i = 0; i < nnz; ++i)
        row_ptr[coo[i].row + 1]++;

    for (int i = 0; i < rows; ++i)
        row_ptr[i + 1] += row_ptr[i];

    int *temp = (int *)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; ++i)
        temp[i] = row_ptr[i];

    for (int i = 0; i < nnz; ++i) {
        int row = coo[i].row;
        int idx = temp[row]++;
        col_idx[idx] = coo[i].col;
        csr_val[idx] = coo[i].val;
    }

    free(temp);
}

// COO SpMV
void spmv_coo(COOEntry *coo, int nnz, float *x, float *y) {
    for (int i = 0; i < nnz; ++i) {
        y[coo[i].row] += coo[i].val * x[coo[i].col];
    }
}

// CSR SpMV
void spmv_csr(int *row_ptr, int *col_idx, float *csr_val, int rows, float *x, float *y) {
    for (int i = 0; i < rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            y[i] += csr_val[j] * x[col_idx[j]];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <rows> <cols> <nnz>\n", argv[0]);
        return 1;
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int nnz = atoi(argv[3]);
    double timers[NITER];

    srand(time(NULL));

    // Allocate and generate COO
    COOEntry *coo = (COOEntry *)malloc(nnz * sizeof(COOEntry));
    generate_random_coo(coo, nnz, rows, cols);

    // Allocate CSR
    int *row_ptr = (int *)malloc((rows + 1) * sizeof(int));
    int *col_idx = (int *)malloc(nnz * sizeof(int));
    float *csr_val = (float *)malloc(nnz * sizeof(float));
    coo_to_csr(coo, row_ptr, col_idx, csr_val, nnz, rows);

    // Input vector and result vectors
    float *x = (float *)malloc(cols * sizeof(float));
    for (int i = 0; i < cols; ++i) x[i] = 1.0f;

    float *y_coo = (float *)calloc(rows, sizeof(float));
    float *y_csr = (float *)calloc(rows, sizeof(float));

    // Verify results
    spmv_coo(coo, nnz, x, y_coo);
    spmv_csr(row_ptr, col_idx, csr_val, rows, x, y_csr);
    int correct = 1;
    for (int i = 0; i < rows; ++i) {
        if (abs(y_coo[i] - y_csr[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    printf("SpMV verification: %s\n", correct ? "SUCCESS" : "FAILURE");

    TIMER_DEF(0);

    // Perform SpMV using COO
    fprintf(stdout, "\nPerform SpMV using COO:\n");
    for (int i=-WARMUP; i<NITER; i++) {

        // Perform C = A * B using BLAS
        TIMER_START(0);
        spmv_coo(coo, nnz, x, y_coo);
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
    }

    double a_mean = arithmetic_mean(timers, NITER);
    fprintf(stdout, "Arithmetic Mean: %lf\n", a_mean);

    int flops = 2 * nnz; // this is an upper-bound
    fprintf(stdout, "My GEMM implementation acheved %lf MFLOP/s\n", flops / 1.e6);

    // Perform SpMV using CSR
    fprintf(stdout, "\nPerform SpMV using CSR:\n");
    for (int i=-WARMUP; i<NITER; i++) {

        // Perform C = A * B using BLAS
        TIMER_START(0);
        spmv_csr(row_ptr, col_idx, csr_val, rows, x, y_csr);
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
    }

    a_mean = arithmetic_mean(timers, NITER);
    fprintf(stdout, "Arithmetic Mean: %lf\n", a_mean);

    fprintf(stdout, "My GEMM implementation acheved %lf MFLOP/s\n", flops / 1.e6);

    // Free memory
    free(coo);
    free(row_ptr);
    free(col_idx);
    free(csr_val);
    free(x);
    free(y_coo);
    free(y_csr);

    return 0;
}

