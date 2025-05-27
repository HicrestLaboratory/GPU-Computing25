#include <stdio.h>
#include <stdlib.h>
#include "include/mmio.h"

void coo_to_csr(int M, int nz, int *I, int *J, double *val,
                int **row_ptr_out, int **col_ind_out, float **values_out) {
    int *row_ptr = (int *)calloc(M + 1, sizeof(int));
    int *col_ind = (int *)malloc(nz * sizeof(int));
    float *values = (float*)malloc(nz * sizeof(float));

    // Step 1: Count number of entries in each row
    for (int i = 0; i < nz; i++) {
        row_ptr[I[i] + 1]++;
    }

    // Step 2: Cumulative sum to get row_ptr
    for (int i = 0; i < M; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Step 3: Fill col_ind and values arrays
    int *temp_row_ptr = (int *)malloc((M + 1) * sizeof(int));
    for (int i = 0; i <= M; i++) {
        temp_row_ptr[i] = row_ptr[i];
    }

    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dest = temp_row_ptr[row];

        col_ind[dest] = J[i];
        values[dest] = (float)val[i];

        temp_row_ptr[row]++;
    }

    free(temp_row_ptr);

    *row_ptr_out = row_ptr;
    *col_ind_out = col_ind;
    *values_out = values;
}

// Sparse Matrix-Vector Multiplication: y = A * x
// A is in CSR (Compressed Sparse Row) format
void spmv_csr(
    int rows,
    const int *row_ptr,   // Row pointer array of size (rows + 1)
    const int *col_ind,   // Column indices array (non-zero count size)
    const float *values,  // Non-zero values array
    const float *x,       // Dense vector input
    float *y              // Result vector (output)
) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y[i] += values[j] * x[col_ind[j]];
        }
    }
}

// Example usage
int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    double *val;

    if (argc < 2)
        {
                fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
                exit(1);
        }
    else
    {
        if ((f = fopen(argv[1], "r")) == NULL)
            exit(1);
    }

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


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

     if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);

    int *row_ptr, *col_ind;
    float *values, *x, *y;

    x = (float*)malloc(sizeof(float)*N);
    y = (float*)malloc(sizeof(float)*N);
    for (int i=0; i<N; i++) x[i] = 1.0;

    coo_to_csr(M, nz, I, J, val, &row_ptr, &col_ind, &values);


    spmv_csr(M, row_ptr, col_ind, values, x, y);

    // Print the result
    printf("Result y = A * x:\n");
    for (int i = 0; i < N; i++) {
        printf("y[%d] = %.2f\n", i, y[i]);
    }

    return 0;
}

