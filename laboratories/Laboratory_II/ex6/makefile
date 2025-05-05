CC=gcc

LIB_FLAGS=-lm

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs

MAIN_NAME=dense2coo
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).c

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(BIN_FOLDER)/$(MAIN_BIN)

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/my_time_lib.c -o $@ $(LIB_FLAGS)

$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(LIB_FLAGS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
