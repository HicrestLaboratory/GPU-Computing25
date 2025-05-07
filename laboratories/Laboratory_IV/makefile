CC=nvcc

LIBS=
INCLUDES=-I../../
LIB_FLAGS=-lm


LABII_FOLDER := ../Laboratory_II/ex5
LABII_OBJ_FOLDER := $(LABII_FOLDER)/obj
LABII_SRC_FOLDER := $(LABII_FOLDER)/src

OBJ_FOLDER := obj

MAIN_NAME=sequentialSpMV
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).c

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(MAIN_BIN)

$(OBJ_FOLDER)/my_time_lib.o: $(LABII_SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(LABII_SRC_FOLDER)/my_time_lib.c -o $@ $(LIB_FLAGS)

$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	$(CC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

clean:
	rm -rf $(MAIN_BIN) $(OBJ_FOLDER)
