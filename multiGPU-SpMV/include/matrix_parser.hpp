#ifndef MATRIX_PARSER_HPP
#define MATRIX_PARSER_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mmio.h"

#define IDXTYPE int
#define VALTYPE float

// COO
template<typename IdxType, typename ValType>
struct COO {
    int nnz, nrows, ncols;
    IdxType* rows_idx;
    IdxType* cols_idx;
    ValType* values;
};

// DENSE
template<typename ValType>
struct DENSE {
    int nrows, ncols;
    ValType* values;
};

// CSR
template<typename IdxType, typename ValType>
struct CSR {
    int nnz, nrows, ncols;
    IdxType* rows_ptr;
    IdxType* cols_idx;
    ValType* values;
};

// Main unified parser
void* my_mtx_parser(int argc, char* argv[], const char* str_outtype, int verbose=0);

#endif // MATRIX_PARSER_HPP

