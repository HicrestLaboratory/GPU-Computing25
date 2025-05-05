CC=nvcc

LIBS=
INCLUDES=-I../../
LIB_FLAGS=-lm

MAIN_NAME=elementWiseAddition
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).cu

all: $(MAIN_BIN)

$(MAIN_BIN): $(MAIN_SRC)
	$(CC) $^ -o $(MAIN_BIN)_layout2 $(LIBS) $(INCLUDES) $(LIB_FLAGS)
	$(CC) $^ -o $(MAIN_BIN)_layout1 $(LIBS) $(INCLUDES) $(LIB_FLAGS) -DLAYOUT1

clean:
	rm -rf $(MAIN_BIN)_*
