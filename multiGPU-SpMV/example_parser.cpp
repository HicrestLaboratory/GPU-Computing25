#include "include/matrix_parser.hpp"
#include <iostream>

int naiv_partitioning (COO<IDXTYPE, VALTYPE>* M, IDXTYPE row, IDXTYPE col, int nproc) {
    int rows_per_proc = (M->nrows % nproc == 0) ? (M->nrows/nproc) : (M->nrows/nproc)+1 ;
    return(row/rows_per_proc);
}

int main(int argc, char* argv[]) {
    COO<IDXTYPE, VALTYPE>* coo = (COO<IDXTYPE, VALTYPE>*) my_mtx_parser(argc, argv, "coo");
    fprintf(stdout, "Loaded COO matrix with %d non-zeros.\n", coo->nnz);
//     free(coo);

    CSR<IDXTYPE, VALTYPE>* csr = (CSR<IDXTYPE, VALTYPE>*) my_mtx_parser(argc, argv, "csr");
    fprintf(stdout, "Loaded COO matrix with %d non-zeros.\n", csr->nnz);
//     free(csr);

    DENSE<VALTYPE>* M = (DENSE<VALTYPE>*) my_mtx_parser(argc, argv, "dense");
    fprintf(stdout, "Loaded COO matrix with %d rows and %d columns.\n", M->nrows, M->ncols);
//     free(M);

    fprintf(stdout, "------------------------ TEST filtering ------------------------\n");
    COO<IDXTYPE, VALTYPE>* filtered_coo = (COO<IDXTYPE, VALTYPE>*) my_mtx_parser(argc, argv, "coo", 0, naiv_partitioning, 0, 4); // Simulating I am process 0 over 4
    fprintf(stdout, "Loaded filtered COO matrix with %d non-zeros.\n", filtered_coo->nnz);

    return 0;
}

