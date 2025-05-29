CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64 -L/opt/shares/cuda/software/CUDA/12.1.1/lib
OPT = -std=c++14 -O2

NVCC = nvcc
NVCC_FLAGS = -lcublas
NVCC_BIN_FLAGS = -lcudart -lcuda
NVCC_LIBS = 
TARGETDIR := bin
OBJECTDIR := obj
INCDIR := include
SRCDIR := ./

LABII_FOLDER := ../laboratories/Laboratory_II/ex5
LABII_OBJ_FOLDER := $(LABII_FOLDER)/obj
LABII_SRC_FOLDER := $(LABII_FOLDER)/src

TIME_OBJECTS = $(OBJECTDIR)/my_time_lib.o
DENSE_OBJECTS = $(TARGETDIR)/matrix_generation.o
PARSER_OBJECTS = $(OBJECTDIR)/matrix_parser.o $(OBJECTDIR)/mmio.o

all: asign parser
asign: $(TARGETDIR)/example_gpuassignment
parser: $(TARGETDIR)/example_parser $(PARSER_OBJECTS)

$(OBJECTDIR)/my_time_lib.o: $(LABII_SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJECTDIR) $(BATCH_OUT_FOLDER)
	$(CC) -c $(LABII_SRC_FOLDER)/my_time_lib.c -o $@ -lm

$(OBJECTDIR)/mmio.o: $(INCDIR)/mmio.c $(INCDIR)/mmio.h
	mkdir -p $(@D)
	$(CC) -o $@ -c $< $(LIBS) $(OPT)

$(OBJECTDIR)/matrix_parser.o: $(INCDIR)/matrix_parser.cpp $(INCDIR)/matrix_parser.hpp $(OBJECTDIR)/mmio.o
	mkdir -p $(@D)
	$(CC) -o $@ -c $< $(LIBS) $(OPT)

$(TARGETDIR)/example_parser: example_parser.cpp $(OBJECTDIR)/matrix_parser.o $(OBJECTDIR)/mmio.o
	mkdir -p $(@D)
	$(CC) $^ -o $@

$(TARGETDIR)/example_gpuassignment: example_gpuassignment.cu
	mkdir -p $(@D)
	$(NVCC) $< -o $@ -lnvidia-ml -lmpi

clean:
	rm -rf $(TARGETDIR)/ $(OBJECTDIR)/
