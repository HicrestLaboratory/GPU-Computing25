#include "matrix_parser.hpp"

template<typename IdxType, typename ValType>
struct COO<IdxType, ValType>* malloc_coo (int n, int m, int nnz) {
    struct COO<IdxType, ValType> *coo = (struct COO<IdxType, ValType>*)malloc(sizeof(struct COO<IdxType, ValType>));

    coo->nnz = nnz;
    coo->nrows = n;
    coo->ncols = m;
    coo->rows_idx = (IdxType*) malloc(nnz * sizeof(IdxType));
    coo->cols_idx = (IdxType*) malloc(nnz * sizeof(IdxType));
    coo->values   = (ValType*) malloc(nnz * sizeof(ValType));

    return(coo);
}

template<typename IdxType, typename ValType>
void free_coo (struct COO<IdxType, ValType> *coo) {
    free(coo->rows_idx);
    free(coo->cols_idx);
    free(coo->values);
    free(coo);
    return;
}

int mtxfile_check (FILE *input_file, MM_typecode *matcode) {
    if (mm_read_banner(input_file, matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(*matcode) && mm_is_matrix(*matcode) &&
            mm_is_sparse(*matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(*matcode));
        return(1);
    }

    return(0);
}

void get_mtx_dims (FILE *f, int *m, int *n, int *nnz) {
    int ret_code;
    if ((ret_code = mm_read_mtx_crd_size(f, m, n, nnz)) !=0) exit(1);
    return;
}

template<typename IdxType, typename ValType>
struct COO<IdxType, ValType>* my_mtx_to_coo (FILE* inputfile, MM_typecode *matcode, int verbose=0) {
    if (mtxfile_check (inputfile, matcode) != 0) exit(__LINE__);

    int m, n, nnz;
    get_mtx_dims(inputfile, &m, &n, &nnz);
    struct COO<IdxType, ValType> *coo = malloc_coo<IdxType, ValType>(n, m, nnz);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i=0; i<nnz; i++)
    {
        fscanf(inputfile, "%d %d %lg\n", &(coo->rows_idx[i]), &(coo->cols_idx[i]), &(coo->values[i]));
        coo->rows_idx[i]--;  /* adjust from 1-based to 0-based */
        coo->cols_idx[i]--;
    }

    if (inputfile !=stdin) fclose(inputfile);

    /************************/
    /* now write out matrix */
    /************************/

    mm_write_banner(stdout, *matcode);
    mm_write_mtx_crd_size(stdout, m, n, nnz);

    if (verbose == 1) {
        for (int i=0; i<nnz; i++)
            fprintf(stdout, "%d %d %20.19g\n", coo->rows_idx[i]+1, coo->cols_idx[i]+1, coo->values[i]);
    }

    return(coo);
}

template<typename ValType>
struct DENSE<ValType>* malloc_dense (int n, int m) {
    struct DENSE<ValType> *M = (struct DENSE<ValType>*)malloc(sizeof(struct DENSE<ValType>));

    M->nrows  = n;
    M->ncols  = m;
    M->values = (ValType*) malloc(n * m * sizeof(ValType));

    return(M);
}

template<typename ValType>
void free_dense (struct DENSE<ValType> *M) {
    free(M->rows_idx);
    free(M->cols_idx);
    free(M->values);
    free(M);
    return;
}

template<typename IdxType, typename ValType>
struct DENSE<ValType>* my_coo_to_dense(struct COO<IdxType, ValType> *coo) {

    int n = coo->nrows, m = coo->ncols;
    struct DENSE<ValType> *M = malloc_dense<ValType>(n, m);

    for(int i=0; i<n*m; i++) M->values[i] = 0;
    for(int i=0; i<coo->nnz; i++) M->values[coo->rows_idx[i] * m + coo->cols_idx[i]] = coo->values[i];
    return(M);
}

template<typename IdxType, typename ValType>
struct DENSE<ValType>* my_mtx_to_dense (FILE* inputfile, MM_typecode *matcode, int verbose=0) {
    struct COO<IdxType, ValType> *coo = my_mtx_to_coo<IdxType, ValType>(inputfile, matcode, verbose);
    struct DENSE<ValType> *M = my_coo_to_dense(coo);
    free_coo(coo);
    return(M);
}

template<typename IdxType, typename ValType>
struct CSR<IdxType, ValType>* malloc_csr(int nrows, int ncols, int nnz) {
    struct CSR<IdxType, ValType>* csr = (struct CSR<IdxType, ValType>*) malloc(sizeof(struct CSR<IdxType, ValType>));

    csr->nnz     = nnz;
    csr->nrows   = nrows;
    csr->ncols   = ncols;
    csr->rows_ptr = (IdxType*) malloc((nrows + 1) * sizeof(IdxType)); // nrows + 1 for CSR format
    csr->cols_idx = (IdxType*) malloc(nnz * sizeof(IdxType));
    csr->values   = (ValType*) malloc(nnz * sizeof(ValType));

    return csr;
}

template<typename IdxType, typename ValType>
void free_csr (struct CSR<IdxType, ValType> *csr) {
    free(csr->rows_idx);
    free(csr->cols_idx);
    free(csr->values);
    free(csr);
    return;
}

template<typename IdxType, typename ValType>
struct CSR<IdxType, ValType>* my_coo_to_csr(struct COO<IdxType, ValType>* coo) {
    int n = coo->nrows;
    int m = coo->ncols;
    int nnz = coo->nnz;

    struct CSR<IdxType, ValType>* csr = malloc_csr<IdxType, ValType>(n, m, nnz);

    // Step 1: Initialize row_ptr to 0
    for (int i = 0; i <= n; i++) csr->rows_ptr[i] = 0;

    // Step 2: Count number of entries in each row
    for (int i = 0; i < nnz; i++) csr->rows_ptr[coo->rows_idx[i] + 1]++;

    // Step 3: Cumulative sum to get row_ptr
    for (int i = 0; i < n; i++) csr->rows_ptr[i + 1] += csr->rows_ptr[i];

    // Step 4: Fill cols_idx and values
    // Temp array to hold current position in each row
    IdxType* curr = (IdxType*) malloc(n * sizeof(IdxType));
    for (int i = 0; i < n; i++) curr[i] = csr->rows_ptr[i];

    for (int i = 0; i < nnz; i++) {
        int row = coo->rows_idx[i];
        int dest = curr[row];

        csr->cols_idx[dest] = coo->cols_idx[i];
        csr->values[dest] = coo->values[i];
        curr[row]++;
    }

    free(curr);
    return csr;
}

template<typename IdxType, typename ValType>
struct CSR<IdxType, ValType>* my_mtx_to_csr (FILE* inputfile, MM_typecode *matcode, int verbose=0) {
    struct COO<IdxType, ValType> *coo = my_mtx_to_coo<IdxType, ValType>(inputfile, matcode, verbose);
    struct CSR<IdxType, ValType> *csr = my_coo_to_csr(coo);
    free_coo(coo);
    return(csr);
}


// Entry point
void* my_mtx_parser(int argc, char* argv[], const char* str_outtype, int verbose) {
    MM_typecode matcode;
    FILE* f;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [filename]\n", argv[0]);
        exit(1);
    }

    if ((f = fopen(argv[1], "r")) == NULL) {
        perror("fopen failed");
        exit(1);
    }

    if (strcmp(str_outtype, "dense") == 0) {
        return (void*) my_mtx_to_dense<IDXTYPE, VALTYPE>(f, &matcode, verbose);
    } else if (strcmp(str_outtype, "coo") == 0) {
        return (void*) my_mtx_to_coo<IDXTYPE, VALTYPE>(f, &matcode, verbose);
    } else if (strcmp(str_outtype, "csr") == 0) {
        return (void*) my_mtx_to_csr<IDXTYPE, VALTYPE>(f, &matcode, verbose);
    } else {
        fprintf(stderr, "Unknown type: %s\n", str_outtype);
        exit(__LINE__);
    }

    return nullptr;
}

