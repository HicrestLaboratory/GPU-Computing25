#include "include/matrix_parser.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    COO<IDXTYPE, VALTYPE>* coo = (COO<IDXTYPE, VALTYPE>*) my_mtx_parser(argc, argv, "coo");
    fprintf(stdout, "Loaded COO matrix with %d non-zeros.\n", coo->nnz);
    free(coo);

    CSR<IDXTYPE, VALTYPE>* csr = (CSR<IDXTYPE, VALTYPE>*) my_mtx_parser(argc, argv, "csr");
    fprintf(stdout, "Loaded COO matrix with %d non-zeros.\n", csr->nnz);
    free(csr);

    DENSE<VALTYPE>* M = (DENSE<VALTYPE>*) my_mtx_parser(argc, argv, "dense");
    fprintf(stdout, "Loaded COO matrix with %d rows and %d columns.\n", M->nrows, M->ncols);
    free(M);

    return 0;
}

