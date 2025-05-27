#include <string>
#ifndef IMPORT_SPARSE_MATRIX
#define IMPORT_SPARSE_MATRIX

void to_csr(float** dense_matrix, int M, int N, int nz, int*& row_offsets, int*& column_indices, float*& values);
void csr_to_file(const char* file, int M, int N, int nz, int*& row_offsets, int*& column_indices, float*& values);
void to_coo(float** dense_matrix, int M, int N, int nz, int*& row_indices, int*& column_indices, float*& values);
void coo_to_file(const char* file, int M, int N, int nz, int*& row_indices, int*& column_indices, float*& values);
void convert_mtx_to_file(const char* file);

void coo_from_file(std::string file, int &rows, int &cols, int &nnz, int*& row_indices, int*& col_indices, float*& values);
void csr_from_file(std::string file, int &rows, int &cols, int &nnz, int*& row_offsets, int*& col_indices, float*& values);

void transposed_csr_to_file(std::string file, int M, int N, int nz, int*& row_offsets, int*& column_indices, float*& values);
void transposed_coo_to_file(std::string file, int M, int N, int nz, int*& row_indices, int*& column_indices, float*& values);

#endif