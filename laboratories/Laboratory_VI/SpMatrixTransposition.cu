#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

#include <cusparse.h>
#include "cublas_v2.h"

#include "include/mmio.h"
#include "include/import_sparse_matrix.h"
#include "../Laboratory_II/ex5/include/my_time_lib.h"

using namespace std;

#define WARMUP 2
#define NUM_REPS 10

#define NKERNEL 4


__global__
void transpose_COO(int* row_indices, int* column_indices, int* row_indices_tp, int* col_indices_tp, int nnz){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    

    while (idx < nnz){
        // swap row and columns        
        row_indices_tp[idx] = column_indices[idx];
        col_indices_tp[idx] = row_indices[idx];        
        
        idx += gridDim.x * blockDim.x;
    }
}

__global__
void transpose_COO_kernelA (int* row_indices, int* column_indices, int* row_indices_tp, int* col_indices_tp, int nnz) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tsize = nnz / (gridDim.x*blockDim.x);
	if (nnz % (gridDim.x*blockDim.x) != 0) tsize++;

	int accessindex = tid*tsize;
	for (int i=0; i<tsize; i++) {
		if (accessindex < nnz) {
            // swap row and columns
            row_indices_tp[accessindex] = column_indices[accessindex];
            col_indices_tp[accessindex] = row_indices[accessindex];
        }
		accessindex++;
	}

}

__global__
void transpose_COO_kernelB (int* row_indices, int* column_indices, int* row_indices_tp, int* col_indices_tp, int nnz) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int nthreads = gridDim.x*blockDim.x;
	int tsize = (nnz+nthreads-1) / (nthreads);
    int h = nthreads/2;

	int accessindex = (tid < h) ? (tid+nthreads-h)*tsize : (tid-h)*tsize ;
	for (int i=0; i<tsize; i++) {
		if (accessindex < nnz) {
            // swap row and columns
            row_indices_tp[accessindex] = column_indices[accessindex];
            col_indices_tp[accessindex] = row_indices[accessindex];
        }
		accessindex++;
	}

}

__global__
void transpose_COO_kernelC (int* row_indices, int* column_indices, int* row_indices_tp, int* col_indices_tp, int nnz) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int nthreads = gridDim.x*blockDim.x;
	int tsize = ((nnz%nthreads) == 0) ? nnz/nthreads : (nnz/nthreads)+1;

	int accessindex = (tid/2)*2*tsize ;
	if ( tid%2 == 1 ) accessindex++;
	for (int i=0; i<tsize; i++) {
		if (accessindex < nnz) {
            // swap row and columns
            row_indices_tp[accessindex] = column_indices[accessindex];
            col_indices_tp[accessindex] = row_indices[accessindex];
        }
		accessindex+=2;
	}


}


int my_mtx_reader(char *matpath, int** rows, int** cols, float** values) {
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    double *val;
    float *fval;

    if ((f = fopen(matpath, "r")) == NULL) exit(1);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));
    fval = (float *) malloc(nz * sizeof(float));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
	fval[i] = (float) val[i];
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
//     for (i=0; i<nz; i++) fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);

    *rows = I;
    *cols = J;
    *values = fval;
    free(val);

	return(nz);
}

typedef struct {
    int blocksize;
    int gridsize;
    double runtime;
} ResultStruct;

ResultStruct* transpose_own_COO(char* file, int kernelnum){

    // load COO from file
    int rows, columns, nnz;
    int *row_indices, *col_indices;
    float* values;

    ResultStruct *result = (ResultStruct*)malloc(sizeof(ResultStruct));

    nnz = my_mtx_reader(file, &row_indices, &col_indices, &values);

//    fprintf(stdout, "Input print (%d nnz):\n", nnz);
//    for (int i=0; i<nnz; i++) fprintf(stdout, "%d %d %20.19g\n", row_indices[i]+1, col_indices[i]+1, values[i]);

    // create arrays on device
    int *dev_row_indices, *dev_col_indices;
    int *dev_row_indices_tp, *dev_col_indices_tp;

    // allocate memory on device
    cudaMalloc(&dev_row_indices, nnz * sizeof(int));
    cudaMalloc(&dev_col_indices, nnz * sizeof(int));
    cudaMalloc(&dev_row_indices_tp, nnz * sizeof(int));
    cudaMalloc(&dev_col_indices_tp, nnz * sizeof(int));


    // copy entries to device
    cudaMemcpy(dev_row_indices, row_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events to use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // try different grid and block sizes and find fastest
    double min_time = INFINITY;
    int min_blocks;
    int min_threads;
    int possible_blocks = ceil(nnz / 1024.);
    int nBlocks;
    int nThreads;

    float milliseconds;
    double timers[NUM_REPS];
    int nMemOp = 4*nnz*sizeof(int)*8; // Each elements require to read the two indices from the input
                                      // matrix and write the two indices in the output matrix (8 is for bit)

    for (int i=1; i<=possible_blocks; i++) {
        nBlocks = i;

        for (int j=4; j<=1024; j*=2) {
            nThreads = j;

            fprintf(stdout, "\nKernel set-up: nBlocks = %d, nThreads = %d\n", nBlocks, nThreads);

            // invoke kernel NUM_REPS times
            for (int k=-WARMUP; k<NUM_REPS; k++){
                // start CUDA timer
                cudaEventRecord(start, 0);

                switch(kernelnum) {
                    case 1:
                        transpose_COO_kernelA<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, dev_row_indices_tp, dev_col_indices_tp, nnz);
                        break;
                    case 2:
                        transpose_COO_kernelC<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, dev_row_indices_tp, dev_col_indices_tp, nnz);
                        break;
                    case 3:
                        transpose_COO_kernelC<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, dev_row_indices_tp, dev_col_indices_tp, nnz);
                        break;
                    default:
                        transpose_COO<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, dev_row_indices_tp, dev_col_indices_tp, nnz);
                }

                // stop CUDA timer
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);

                // Calculate elapsed time
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);

                if (k == -WARMUP) printf("\nWarm-up cycles: ");
                if (k == 0) printf("\nIteration cycles: ");

                printf("%f ", k, milliseconds);
                fflush(stdout);

                if( k >= 0) timers[i] = (double) milliseconds;
            }

            double a_mean = arithmetic_mean(timers, NUM_REPS);
            fprintf(stdout, "\nArithmetic Mean: %lf\n", a_mean);
            fprintf(stdout, "Memory Bandwidth: %lf Kb/s\n", (double)nMemOp / a_mean);

            if (a_mean < min_time){
                min_time = a_mean;
                min_blocks = i;
                min_threads = j;
            }
        }
    }

    // find best configuration
    printf("\nBest configuration: %d, %d, %lf\n", min_blocks, min_threads, min_time);

    fprintf(stdout, "\nmilliseconds, rows, columns, nnz, min_blocks, min_threads\n");
    fprintf(stdout, "%lf, %d, %d, %d, %d, %d\n", milliseconds, rows, columns, nnz, min_blocks, min_threads);
    result->blocksize = min_threads;
    result->gridsize  = min_blocks;
    result->runtime   = milliseconds;

    // copy back
    cudaMemcpy(row_indices, dev_row_indices_tp, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col_indices, dev_col_indices_tp, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // save result to file
    transposed_coo_to_file(file, columns, rows, nnz, row_indices, col_indices, values);

    // free device memory
    cudaFree(dev_row_indices);
    cudaFree(dev_col_indices);

    // free host memory
    free(row_indices);
    free(col_indices);
    free(values);

    // free timer events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return(result);
}


int main(int argc, char* argv[]){
    printf("Started script\n");

    // Strategy 2: own COO transpose kernel
    printf("Use COO format and own kernel.\n");

    // check which test matrix to use
    if (argc < 2){
        fprintf(stderr, "Please choose a test matrix\n");
    }

    fprintf(stdout, "argv1: %s\n", argv[1]);

    printf("Transposing matrix %d\n", argv[1]);

    ResultStruct *result_vec[NKERNEL];
    const char* kernelnames[NKERNEL] = {"transpose_COO", "transpose_COO_kernelA", "transpose_COO_kernelB", "transpose_COO_kernelC"};

    for (int kernelnum=0; kernelnum<NKERNEL; kernelnum++) {
        switch(kernelnum) {
            case 1:
                printf("Use transpose_COO_kernelA kernel.\n");
                break;
            case 2:
                printf("Use transpose_COO_kernelB kernel.\n");
                break;
            case 3:
                printf("Use transpose_COO_kernelC kernel.\n");
                break;
            default:
                printf("Use transpose_COO kernel.\n");
        }
        printf("Use %s kernel.\n", kernelnames[kernelnum]);
        result_vec[kernelnum] = transpose_own_COO(argv[1], kernelnum);
    }

    fprintf(stdout, "\nkernelname milliseconds, min_blocks, min_threads\n");
    for (int i=0; i<NKERNEL; i++) {
        fprintf(stdout, "%20s, %lf, %d, %d\n", kernelnames[i], result_vec[i]->runtime, result_vec[i]->gridsize, result_vec[i]->blocksize);
    }
    
    return 0;
}

