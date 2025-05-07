#include <iostream>
#include "matrix_generation.h"
using namespace std;

int* generate_random_matrix(int size){
	int* matrix = (int*) malloc(size * size * sizeof(int));

	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			matrix[i * size + j] = rand() % 100;  // random number between 0 and 99
		}
	}

	return matrix;
}

int* generate_continous_matrix(int size){
	int* matrix = (int*) malloc(size * size * sizeof(int));

	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			matrix[i * size + j] = i * size + j;  // fill with numbers counting up from zero
		}
	}

	return matrix;
}

