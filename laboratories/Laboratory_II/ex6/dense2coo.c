#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "include/my_time_lib.h"

#define dtype double
#define WARMUP 2
#define NITER 10
#define NTEST 5

/* 
 * Fills an already allocated n x m dense matrix with values.
 * Each entry has a probability p of being 0; if not zero, a random value in [0,10) is used.
 */
void fill_dense_matrix(double *matrix, int n, int m, double p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double r = (double)rand() / RAND_MAX;
            if (r < p)
                matrix[i*m + j] = 0.0;
            else
                matrix[i*m + j] = ((double)rand() / RAND_MAX) * 10.0;
        }
    }
}

/*
 * Counts nonzero entries in a dense matrix.
 */
int count_nonzeros(double *matrix, int n, int m) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (matrix[i*m + j] != 0.0)
                count++;
        }
    }
    return count;
}

/*
 * Converts a dense matrix to COO format.
 * The COO representation is stored in three arrays:
 *   row_indices, col_indices, and values.
 * The arrays must be allocated in main with a size large enough to hold all nonzeros.
 */
void dense_to_coo(double *matrix, int n, int m, int *row_indices, int *col_indices, double *values) {
    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (matrix[i*m + j] != 0.0) {
                row_indices[idx] = i;
                col_indices[idx] = j;
                values[idx] = matrix[i*m + j];
                idx++;
            }
        }
    }
}

/*
 * Converts a COO format (given as three arrays and the number of nonzeros) 
 * back to a dense matrix.
 * The dense matrix is assumed to be allocated in main.
 * This function first zeros the matrix, then fills in the nonzero entries.
 */
void coo_to_dense(int *row_indices, int *col_indices, double *values, int nnz, double *matrix, int n, int m) {
    // Initialize the dense matrix to zeros.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i*m + j] = 0.0;
        }
    }
    // Fill in the nonzero values.
    for (int k = 0; k < nnz; k++) {
        int i = row_indices[k];
        int j = col_indices[k];
        matrix[i*m + j] = values[k];
    }
}


int main(void) {
    srand(time(NULL));

    int n = 100, m = 100, nnz;
    
    double p_values[] = {0.1, 0.3, 0.5, 0.7, 0.9};
    
    double dense2coo_times[NTEST][NITER];
    double coo2dense_times[NTEST][NITER];
    double dense2coo_avgs[NTEST];
    double coo2dense_avgs[NTEST];

    double *dense = malloc(n * m * sizeof(double));
    double *new_dense = malloc(n * m * sizeof(double));

    TIMER_DEF(0);

    for (int i = 0; i < NTEST; i++) {
        double p = p_values[i];
	fill_dense_matrix(dense, n, m, p);
        nnz = count_nonzeros(dense, n, m);
        fprintf(stdout, "\nTest with p = %lf... %d nnz\n", p, nnz);

	for (int j = -WARMUP; j < NITER; j++) {


            /* Allocate COO arrays in main */
            int *row_indices = malloc(nnz * sizeof(int));
            int *col_indices = malloc(nnz * sizeof(int));
            double *values   = malloc(nnz * sizeof(double));

            /* Benchmark dense to COO conversion */
	    TIMER_START(0);
	    dense_to_coo(dense, n, m, row_indices, col_indices, values);
	    TIMER_STOP(0);

	    fprintf(stdout, "\tIteration %d took %lfs\n", j, TIMER_ELAPSED(0));
	    dense2coo_times[i][j] = TIMER_ELAPSED(0);


            /* Benchmark COO to dense conversion */
            TIMER_START(0);
	    coo_to_dense(row_indices, col_indices, values, nnz, new_dense, n, m);
	    TIMER_STOP(0);

	    fprintf(stdout, "\tIteration %d took %lfs\n", j, TIMER_ELAPSED(0));
	    coo2dense_times[i][j] = TIMER_ELAPSED(0);

	    if(j==NITER-1) {
                    dense2coo_avgs[i] = arithmetic_mean(dense2coo_times[i], NITER);
		    coo2dense_avgs[i] = arithmetic_mean(coo2dense_times[i], NITER);
                    fprintf(stdout, "Average of iteration %d is %lfs\n", j, dense2coo_avgs[i]);
		    fprintf(stdout, "Average of iteration %d is %lfs\n", j, coo2dense_avgs[i]);
            }

            if(j<0) {
                int flag=0;
                int k = 0, h = 0;
                while(k<n && flag==0) {
                    while(h<m && flag==0) {
                        if(dense[k*m+h] != new_dense[k*m+h]) flag=1;
                        h++;
                    }
                    k++;
                }

                if(flag==1) {
                    fprintf(stderr, "Error: input and output are different:\tdense[%d][%d] = %lf != %lf = new_dense[%d][%d]\n", k, h, dense[k*m+h], new_dense[k*m+h], k, h);
                } else {
                    fprintf(stdout, "\tCorrectness check passed!\n");
                }
            }

            /* Free allocated memory for this test */
            free(row_indices);
            free(col_indices);
            free(values);
        }
    }

    free(dense);
    free(new_dense);

    return(0);
}

