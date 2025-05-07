CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64 -L/opt/shares/cuda/software/CUDA/12.1.1/lib
OPT = -std=c++14 -O2

NVCC = nvcc
NVCC_FLAGS = -lcublas
NVCC_BIN_FLAGS = -lcudart -lcuda
NVCC_LIBS = 
TARGETDIR := bin
SRCDIR := ./
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/matrix_generation.h
INCLUDE_CPP := $(INCDIR)/matrix_generation.cpp

LABII_FOLDER := ../Laboratory_II/ex5
LABII_OBJ_FOLDER := $(LABII_FOLDER)/obj
LABII_SRC_FOLDER := $(LABII_FOLDER)/src

MAIN_NAME=matrixTranspose
SPARSE_MAIN_NAME=SpMatrixTransposition

OBJ_FOLDER := obj

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o
DENSE_OBJECTS = $(TARGETDIR)/matrix_generation.o
SPARSE_OBJECTS = $(TARGETDIR)/mmio.o $(TARGETDIR)/import_sparse.o

all:  dense sparse
dense: $(TARGETDIR)/$(MAIN_NAME)
sparse: $(TARGETDIR)/$(SPARSE_MAIN_NAME)

$(OBJ_FOLDER)/my_time_lib.o: $(LABII_SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(LABII_SRC_FOLDER)/my_time_lib.c -o $@ -lm

$(TARGETDIR)/matrix_generation.o: $(INCLUDE_CPP) $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) -o $@ -c $< $(LIBS) $(OPT)

$(TARGETDIR)/$(MAIN_NAME): $(SRCDIR)/$(MAIN_NAME).cu $(DENSE_OBJECTS) $(OBJECTS)
	$(NVCC) $(SRCDIR)/$(MAIN_NAME).cu -o $@ $(DENSE_OBJECTS) $(OBJECTS) $(NVCC_BIN_FLAGS) -lcublas

$(TARGETDIR)/mmio.o: $(INCDIR)/mmio.c $(INCDIR)/mmio.h
	mkdir -p $(@D)
	$(CC) -o $@ -c $< $(LIBS) $(OPT)

$(TARGETDIR)/import_sparse.o: $(INCDIR)/import_sparse_matrix.cpp $(INCDIR)/import_sparse_matrix.h
	mkdir -p $(@D)
	$(CC) -o $@ -c $< $(LIBS) $(OPT)

$(TARGETDIR)/$(SPARSE_MAIN_NAME): $(SRCDIR)/$(SPARSE_MAIN_NAME).cu $(SPARSE_OBJECTS) $(OBJECTS)
	$(NVCC) $(SRCDIR)/$(SPARSE_MAIN_NAME).cu -o $@ $(SPARSE_OBJECTS) $(OBJECTS) $(NVCC_BIN_FLAGS) -lcusparse
	$(NVCC) $(SRCDIR)/$(SPARSE_MAIN_NAME).cu -o $@_kA $(SPARSE_OBJECTS) $(OBJECTS) $(NVCC_BIN_FLAGS) -lcusparse -DKERNEL=1
	$(NVCC) $(SRCDIR)/$(SPARSE_MAIN_NAME).cu -o $@_kB $(SPARSE_OBJECTS) $(OBJECTS) $(NVCC_BIN_FLAGS) -lcusparse -DKERNEL=2
	$(NVCC) $(SRCDIR)/$(SPARSE_MAIN_NAME).cu -o $@_kC $(SPARSE_OBJECTS) $(OBJECTS) $(NVCC_BIN_FLAGS) -lcusparse -DKERNEL=3

clean:
	rm -rf $(TARGETDIR)/ $(OBJ_FOLDER)/
