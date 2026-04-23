CC := cc
CFLAGS := -O3 -march=native -Wall -Wextra -std=c11
LDFLAGS := -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
OPENMP_CFLAGS := -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
OPENMP_LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
else
OPENMP_CFLAGS := -fopenmp
OPENMP_LDFLAGS := -fopenmp
endif

BIN := bin/spmv_bench
SRC := src/spmv_bench.c

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	mkdir -p bin
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -o $(BIN) $(SRC) $(LDFLAGS) $(OPENMP_LDFLAGS)

run: $(BIN)
	mkdir -p results
	./$(BIN) --output results/benchmark_results.csv

clean:
	rm -rf bin results/*.csv results/*.png
