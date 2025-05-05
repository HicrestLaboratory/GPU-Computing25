CC=gcc

LIBS=-L/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0/lib 
INCLUDES=-I/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0/include -Iinclude
LIB_FLAGS=-lm -lopenblas

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs

LABII_FOLDER := ../Laboratory_II/ex5
LABII_OBJ_FOLDER := $(LABII_FOLDER)/obj
LABII_SRC_FOLDER := $(LABII_FOLDER)/src

MAIN_NAME=gemm
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).c

BLOCK_NAME=block_gemm
BLOCK_BIN=$(MAIN_NAME)
BLOCK_SRC=$(SRC_FOLDER)/$(BLOCK_NAME).c

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(BIN_FOLDER)/$(MAIN_BIN) $(BIN_FOLDER)/$(BLOCK_NAME)

$(OBJ_FOLDER)/my_time_lib.o: $(LABII_SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(LABII_SRC_FOLDER)/my_time_lib.c -o $@ $(LIB_FLAGS)

$(BIN_FOLDER)/$(MAIN_BIN): $(LABII_FOLDER)/$(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)
	$(CC) $^ -o $(MAIN_BIN)_O0 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O0
	$(CC) $^ -o $(MAIN_BIN)_O1 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O1
	$(CC) $^ -o $(MAIN_BIN)_O2 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O2
	$(CC) $^ -o $(MAIN_BIN)_O3 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O3

$(BIN_FOLDER)/$(BLOCK_NAME): $(BLOCK_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)
	$(CC) $^ -o $(BLOCK_NAME)_O0 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O0
	$(CC) $^ -o $(BLOCK_NAME)_O1 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O1
	$(CC) $^ -o $(BLOCK_NAME)_O2 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O2
	$(CC) $^ -o $(BLOCK_NAME)_O3 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -O3

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
	rm -rf $(MAIN_NAME)_* $(BLOCK_NAME)_*
