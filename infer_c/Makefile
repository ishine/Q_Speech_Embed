SHELL := cmd.exe
.DEFAULT_GOAL := all

CC := gcc
CFLAGS  := -O2 -std=c11 -Wall -I.
LDFLAGS := -lm

SRC = \
    activation.c \
    atten_blocks.c \
    bit_gate_net.c \
    conv2d.c \
    dense.c \
    main.c \
    qparams.c \
    pooling.c \
    softmax.c \
    tensor.c \
    upsampling.c \
    utils.c \
    weights_meta.c \
    weights.c

OBJ = $(SRC:.c=.o)

TARGET = infer.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-@del /Q *.o *.exe
	@echo Cleaned.

run: $(TARGET)
	@echo === Running inference ===
	@$(TARGET)
