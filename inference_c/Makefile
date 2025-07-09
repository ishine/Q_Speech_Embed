# ────────────────────────────────────────────────────────────
#  TinySpeech – Windows / MinGW Makefile  (fixed)
# ────────────────────────────────────────────────────────────
SHELL  := cmd.exe
.DEFAULT_GOAL := help

CC      := gcc
CFLAGS  := -O2 -std=c11 -Wall -I.
LDFLAGS := -lm

CORE_SRC := tensor.c utils.c weights.c conv2d.c pooling.c \
            upsampling.c dense.c modules.c softmax.c tinyspeech.c

LIB := libtinyspeech.a

# ───────── build static library ─────────────────────────────
$(LIB): $(CORE_SRC:%.c=%.o)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ───────── run full network (main.c) ────────────────────────
run: $(LIB) main.c
	$(CC) $(CFLAGS) $^ -o main.exe $(LDFLAGS)
	@echo === running ===
	@main.exe

# ───────── VERIFY: single layer  ────────────────────────────
# Usage:  mingw32-make verify LAYER=conv2d
verify:
	@if not defined LAYER ( \
		echo Specify layer: mingw32-make verify LAYER=conv2d & exit /b 1 ) \
	else ( \
		$(MAKE) layer_test L=%LAYER% )

layer_test : $(LIB)
	@echo --- compiling $(L)
	$(CC) $(CFLAGS) $(L)/test.c $(LIB) -o $(L)/test.exe $(LDFLAGS)
	@echo --- running $(L)/test.exe
	@$(L)\test.exe

# ───────── VERIFY ALL layers that have a test.c ─────────────
verify_all: $(LIB)
	@for /D %%d in (*) do ( \
		if exist "%%d\\test.c" ( \
			echo --- compiling %%d & \
			$(CC) $(CFLAGS) "%%d/test.c" $(LIB) -o "%%d/test.exe" $(LDFLAGS) & \
			echo --- running %%d/test.exe & \
			"%%d\\test.exe" \
		) \
	)

# ───────── housekeeping ────────────────────────────────────
clean:
	-@del /Q *.o *.exe $(LIB) 2>nul
	-@for /r %%f in (test.exe) do del /Q "%%f" 2>nul
	@echo cleaned

