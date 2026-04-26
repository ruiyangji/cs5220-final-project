# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build the benchmark binary
make all

# Run full benchmark sweep (5 warm-up, 20 timed, threads 1..32)
./scripts/run_benchmarks.sh

# Override benchmark parameters
WARMUP=3 TIMED=10 MAX_THREADS=8 ./scripts/run_benchmarks.sh
USE_AVX2=0 ./scripts/run_benchmarks.sh  # disable AVX2 path

# Run binary directly with custom flags
./bin/spmv_bench --warmup 2 --timed 5 --min-threads 1 --max-threads 4 --output results/benchmark_results.csv

# Generate plots and narrative from CSV
python3 scripts/plot_results.py

# Override plot I/O paths
IN_CSV=results/my.csv OUT_ROOFLINE=results/r.png OUT_SCALING=results/s.png OUT_SUMMARY=results/n.md python3 scripts/plot_results.py

# Clean build artifacts and result CSVs/PNGs
make clean
```

There is no test suite; correctness is verified inline at the start of each benchmark run (`dense_reference` vs each format, tolerance 1e-6 to 1e-8).

## Architecture

**Single-source C benchmark** (`src/spmv_bench.c`) compiled to `bin/spmv_bench`. All matrix formats, generators, kernel implementations, timing, and CSV output are in this one file.

### Sparse formats implemented
- **CSR** (baseline): `CSRMatrix` with `row_ptr`, `col_idx`, `values`.
- **ELLPACK**: `ELLMatrix` with column-major layout (`col-major[k][row]`); padded to `max_row_nnz` with sentinel `-1` columns.
- **BCSR 2×2 and 4×4**: `BCSRMatrix` with dense block storage; block values stored as `nb × br × bc` flat array.

All formats are converted from CSR at runtime — no external matrix files are loaded.

### Kernel paths
- `csr_spmv`: OpenMP static-scheduled row loop.
- `ell_spmv`: Optional AVX2 4-wide unroll on x86; scalar fallback otherwise.
- `bcsr_spmv`: Optional AVX2 path for 4×4 blocks on x86; scalar fallback for 2×2 and non-x86.
- AVX2 is enabled/disabled by `Config.use_avx2` and gated at compile time by `HAVE_X86_SIMD`.

### Matrix families (locked, synthetic)
12 matrices across 4 families (3 sizes each), generated with fixed RNG seed (`srand(7)`):
- `irregular_graph` — power-law degree distribution
- `fem_mesh` — 2D structured mesh (9-point stencil on `m×m` grid)
- `block_structured` — dense block diagonal-shifted pattern
- `banded_pde` — banded tridiagonal-like stencil

Matrix parameters are hardcoded in `main()`. The selection is locked before benchmarking to prevent cherry-picking (see `docs/matrix_selection.md`).

### Benchmark loop
For each matrix, for each thread count in `[min_threads, max_threads]`:
1. Warm up each format `warmup_runs` times.
2. Time `timed_runs` runs and record the median.
3. Write one CSV row per (matrix, format, thread-count) with: time, GFLOP/s, bandwidth GB/s, arithmetic intensity, and STREAM peak.

STREAM peak bandwidth is estimated once at startup (`estimate_stream_bw_gbs`) using a triad kernel over 10M doubles.

### Output
- `results/benchmark_results.csv` — raw timing data.
- `results/roofline.png` — roofline scatter at max thread count.
- `results/strong_scaling.png` — speedup vs threads averaged across matrices.
- `results/results_narrative.md` — auto-generated per-family summary.

### Platform notes
- **macOS**: OpenMP linked via Homebrew libomp (`-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include`).
- **Linux**: Standard `-fopenmp`.
- AVX2 (`immintrin.h`) is only compiled on x86-64; ARM builds use the scalar fallback.
