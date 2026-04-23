#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_X86_SIMD 1
#else
#define HAVE_X86_SIMD 0
#endif

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *values;
    char name[64];
    char family[32];
} CSRMatrix;

typedef struct {
    int nrows;
    int ncols;
    int max_row_nnz;
    int *col_idx;
    double *values;
} ELLMatrix;

typedef struct {
    int nrows;
    int ncols;
    int br;
    int bc;
    int n_block_rows;
    int nb;
    int *row_ptr;
    int *col_idx;
    double *values;
} BCSRMatrix;

typedef struct {
    int warmup_runs;
    int timed_runs;
    int min_threads;
    int max_threads;
    int use_avx2;
    const char *output_csv;
} Config;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median(double *vals, int n) {
    qsort(vals, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 0) return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
    return vals[n / 2];
}

static double rand_unit(void) {
    return (double)rand() / (double)RAND_MAX;
}

static void free_csr(CSRMatrix *A) {
    free(A->row_ptr);
    free(A->col_idx);
    free(A->values);
}

static void free_ell(ELLMatrix *E) {
    free(E->col_idx);
    free(E->values);
}

static void free_bcsr(BCSRMatrix *B) {
    free(B->row_ptr);
    free(B->col_idx);
    free(B->values);
}

static void csr_spmv(const CSRMatrix *A, const double *x, double *y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A->nrows; i++) {
        double sum = 0.0;
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            sum += A->values[p] * x[A->col_idx[p]];
        }
        y[i] = sum;
    }
}

static ELLMatrix csr_to_ell(const CSRMatrix *A) {
    ELLMatrix E;
    E.nrows = A->nrows;
    E.ncols = A->ncols;
    E.max_row_nnz = 0;
    for (int i = 0; i < A->nrows; i++) {
        int rnnz = A->row_ptr[i + 1] - A->row_ptr[i];
        if (rnnz > E.max_row_nnz) E.max_row_nnz = rnnz;
    }
    size_t total = (size_t)E.nrows * (size_t)E.max_row_nnz;
    E.col_idx = (int *)malloc(total * sizeof(int));
    E.values = (double *)malloc(total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        E.col_idx[i] = -1;
        E.values[i] = 0.0;
    }
    for (int r = 0; r < E.nrows; r++) {
        int start = A->row_ptr[r];
        int end = A->row_ptr[r + 1];
        int k = 0;
        for (int p = start; p < end; p++, k++) {
            size_t idx = (size_t)k * (size_t)E.nrows + (size_t)r;
            E.col_idx[idx] = A->col_idx[p];
            E.values[idx] = A->values[p];
        }
    }
    return E;
}

static void ell_spmv(const ELLMatrix *E, const double *x, double *y, int use_avx2) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < E->nrows; r++) {
        double sum = 0.0;
        int k = 0;
        if (use_avx2 && HAVE_X86_SIMD) {
            for (; k + 3 < E->max_row_nnz; k += 4) {
#if HAVE_X86_SIMD
                double vals[4];
                int cols[4];
                for (int t = 0; t < 4; t++) {
                    size_t idx = (size_t)(k + t) * (size_t)E->nrows + (size_t)r;
                    vals[t] = E->values[idx];
                    cols[t] = E->col_idx[idx];
                }
                __m256d v = _mm256_loadu_pd(vals);
                __m256d xv = _mm256_set_pd(
                    cols[3] >= 0 ? x[cols[3]] : 0.0,
                    cols[2] >= 0 ? x[cols[2]] : 0.0,
                    cols[1] >= 0 ? x[cols[1]] : 0.0,
                    cols[0] >= 0 ? x[cols[0]] : 0.0
                );
                __m256d prod = _mm256_mul_pd(v, xv);
                double tmp[4];
                _mm256_storeu_pd(tmp, prod);
                sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
            }
        }
        for (; k < E->max_row_nnz; k++) {
            size_t idx = (size_t)k * (size_t)E->nrows + (size_t)r;
            int c = E->col_idx[idx];
            if (c >= 0) sum += E->values[idx] * x[c];
        }
        y[r] = sum;
    }
}

static BCSRMatrix csr_to_bcsr(const CSRMatrix *A, int br, int bc) {
    BCSRMatrix B;
    B.nrows = A->nrows;
    B.ncols = A->ncols;
    B.br = br;
    B.bc = bc;
    B.n_block_rows = (A->nrows + br - 1) / br;
    B.row_ptr = (int *)calloc((size_t)B.n_block_rows + 1, sizeof(int));

    int cap = 1024;
    B.col_idx = (int *)malloc((size_t)cap * sizeof(int));
    B.values = (double *)malloc((size_t)cap * (size_t)br * (size_t)bc * sizeof(double));
    B.nb = 0;

    for (int rb = 0; rb < B.n_block_rows; rb++) {
        B.row_ptr[rb] = B.nb;
        int r0 = rb * br;
        int row_count = (r0 + br <= A->nrows) ? br : (A->nrows - r0);

        int local_cap = 64;
        int local_n = 0;
        int *local_cols = (int *)malloc((size_t)local_cap * sizeof(int));
        double *local_blocks = (double *)malloc((size_t)local_cap * (size_t)br * (size_t)bc * sizeof(double));

        for (int rr = 0; rr < row_count; rr++) {
            int r = r0 + rr;
            for (int p = A->row_ptr[r]; p < A->row_ptr[r + 1]; p++) {
                int cb = A->col_idx[p] / bc;
                int found = -1;
                for (int k = 0; k < local_n; k++) {
                    if (local_cols[k] == cb) {
                        found = k;
                        break;
                    }
                }
                if (found < 0) {
                    if (local_n == local_cap) {
                        local_cap *= 2;
                        local_cols = (int *)realloc(local_cols, (size_t)local_cap * sizeof(int));
                        local_blocks = (double *)realloc(local_blocks, (size_t)local_cap * (size_t)br * (size_t)bc * sizeof(double));
                    }
                    found = local_n++;
                    local_cols[found] = cb;
                    for (int z = 0; z < br * bc; z++) local_blocks[(size_t)found * (size_t)br * (size_t)bc + (size_t)z] = 0.0;
                }
                int cc = A->col_idx[p] % bc;
                local_blocks[(size_t)found * (size_t)br * (size_t)bc + (size_t)rr * (size_t)bc + (size_t)cc] = A->values[p];
            }
        }

        for (int i = 0; i < local_n - 1; i++) {
            for (int j = i + 1; j < local_n; j++) {
                if (local_cols[j] < local_cols[i]) {
                    int tc = local_cols[i];
                    local_cols[i] = local_cols[j];
                    local_cols[j] = tc;
                    for (int z = 0; z < br * bc; z++) {
                        double tv = local_blocks[(size_t)i * (size_t)br * (size_t)bc + (size_t)z];
                        local_blocks[(size_t)i * (size_t)br * (size_t)bc + (size_t)z] = local_blocks[(size_t)j * (size_t)br * (size_t)bc + (size_t)z];
                        local_blocks[(size_t)j * (size_t)br * (size_t)bc + (size_t)z] = tv;
                    }
                }
            }
        }

        while (B.nb + local_n > cap) {
            cap *= 2;
            B.col_idx = (int *)realloc(B.col_idx, (size_t)cap * sizeof(int));
            B.values = (double *)realloc(B.values, (size_t)cap * (size_t)br * (size_t)bc * sizeof(double));
        }

        for (int k = 0; k < local_n; k++) {
            B.col_idx[B.nb] = local_cols[k];
            for (int z = 0; z < br * bc; z++) {
                B.values[(size_t)B.nb * (size_t)br * (size_t)bc + (size_t)z] =
                    local_blocks[(size_t)k * (size_t)br * (size_t)bc + (size_t)z];
            }
            B.nb++;
        }

        free(local_cols);
        free(local_blocks);
    }
    B.row_ptr[B.n_block_rows] = B.nb;
    return B;
}

static void bcsr_spmv(const BCSRMatrix *B, const double *x, double *y, int use_avx2) {
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < B->n_block_rows; rb++) {
        int r0 = rb * B->br;
        int row_count = (r0 + B->br <= B->nrows) ? B->br : (B->nrows - r0);
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (int p = B->row_ptr[rb]; p < B->row_ptr[rb + 1]; p++) {
            int c0 = B->col_idx[p] * B->bc;
            const double *blk = &B->values[(size_t)p * (size_t)B->br * (size_t)B->bc];
            if (use_avx2 && HAVE_X86_SIMD && B->bc == 4) {
#if HAVE_X86_SIMD
                __m256d xv = _mm256_set_pd(
                    (c0 + 3 < B->ncols) ? x[c0 + 3] : 0.0,
                    (c0 + 2 < B->ncols) ? x[c0 + 2] : 0.0,
                    (c0 + 1 < B->ncols) ? x[c0 + 1] : 0.0,
                    (c0 < B->ncols) ? x[c0] : 0.0
                );
                for (int rr = 0; rr < row_count; rr++) {
                    __m256d av = _mm256_loadu_pd(&blk[(size_t)rr * (size_t)B->bc]);
                    __m256d pv = _mm256_mul_pd(av, xv);
                    double tmp[4];
                    _mm256_storeu_pd(tmp, pv);
                    acc[rr] += tmp[0] + tmp[1] + tmp[2] + tmp[3];
                }
#endif
            } else {
                for (int rr = 0; rr < row_count; rr++) {
                    for (int cc = 0; cc < B->bc; cc++) {
                        int c = c0 + cc;
                        if (c < B->ncols) acc[rr] += blk[(size_t)rr * (size_t)B->bc + (size_t)cc] * x[c];
                    }
                }
            }
        }
        for (int rr = 0; rr < row_count; rr++) y[r0 + rr] = acc[rr];
    }
}

static void dense_reference(const CSRMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->nrows; i++) y[i] = 0.0;
    for (int r = 0; r < A->nrows; r++) {
        for (int p = A->row_ptr[r]; p < A->row_ptr[r + 1]; p++) {
            y[r] += A->values[p] * x[A->col_idx[p]];
        }
    }
}

static double max_abs_diff(const double *a, const double *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static CSRMatrix make_from_triplets(
    int nrows, int ncols, int nnz,
    int *rows, int *cols, double *vals,
    const char *name, const char *family
) {
    CSRMatrix A;
    A.nrows = nrows;
    A.ncols = ncols;
    A.nnz = nnz;
    snprintf(A.name, sizeof(A.name), "%s", name);
    snprintf(A.family, sizeof(A.family), "%s", family);

    A.row_ptr = (int *)calloc((size_t)nrows + 1, sizeof(int));
    A.col_idx = (int *)malloc((size_t)nnz * sizeof(int));
    A.values = (double *)malloc((size_t)nnz * sizeof(double));

    for (int i = 0; i < nnz; i++) A.row_ptr[rows[i] + 1]++;
    for (int i = 1; i <= nrows; i++) A.row_ptr[i] += A.row_ptr[i - 1];

    int *next = (int *)malloc((size_t)nrows * sizeof(int));
    memcpy(next, A.row_ptr, (size_t)nrows * sizeof(int));
    for (int i = 0; i < nnz; i++) {
        int r = rows[i];
        int p = next[r]++;
        A.col_idx[p] = cols[i];
        A.values[p] = vals[i];
    }
    free(next);

    for (int r = 0; r < nrows; r++) {
        int s = A.row_ptr[r];
        int e = A.row_ptr[r + 1];
        for (int i = s + 1; i < e; i++) {
            int c = A.col_idx[i];
            double v = A.values[i];
            int j = i - 1;
            while (j >= s && A.col_idx[j] > c) {
                A.col_idx[j + 1] = A.col_idx[j];
                A.values[j + 1] = A.values[j];
                j--;
            }
            A.col_idx[j + 1] = c;
            A.values[j + 1] = v;
        }
    }

    int *new_row_ptr = (int *)calloc((size_t)nrows + 1, sizeof(int));
    int *new_col_idx = (int *)malloc((size_t)nnz * sizeof(int));
    double *new_vals = (double *)malloc((size_t)nnz * sizeof(double));
    int out = 0;
    for (int r = 0; r < nrows; r++) {
        new_row_ptr[r] = out;
        int s = A.row_ptr[r];
        int e = A.row_ptr[r + 1];
        int p = s;
        while (p < e) {
            int c = A.col_idx[p];
            double acc = 0.0;
            while (p < e && A.col_idx[p] == c) {
                acc += A.values[p];
                p++;
            }
            new_col_idx[out] = c;
            new_vals[out] = acc;
            out++;
        }
    }
    new_row_ptr[nrows] = out;

    free(A.row_ptr);
    free(A.col_idx);
    free(A.values);
    A.row_ptr = new_row_ptr;
    A.col_idx = new_col_idx;
    A.values = new_vals;
    A.nnz = out;
    return A;
}

static CSRMatrix generate_irregular_graph(int n, int avg_deg, const char *name) {
    int cap = n * avg_deg * 2;
    int *rows = (int *)malloc((size_t)cap * sizeof(int));
    int *cols = (int *)malloc((size_t)cap * sizeof(int));
    double *vals = (double *)malloc((size_t)cap * sizeof(double));
    int nnz = 0;
    for (int r = 0; r < n; r++) {
        int degree = 1 + (int)(pow(rand_unit(), 2.0) * (double)(2 * avg_deg));
        for (int k = 0; k < degree; k++) {
            if (nnz == cap) {
                cap *= 2;
                rows = (int *)realloc(rows, (size_t)cap * sizeof(int));
                cols = (int *)realloc(cols, (size_t)cap * sizeof(int));
                vals = (double *)realloc(vals, (size_t)cap * sizeof(double));
            }
            rows[nnz] = r;
            cols[nnz] = rand() % n;
            vals[nnz] = 0.1 + rand_unit();
            nnz++;
        }
    }
    CSRMatrix A = make_from_triplets(n, n, nnz, rows, cols, vals, name, "irregular_graph");
    free(rows); free(cols); free(vals);
    return A;
}

static CSRMatrix generate_banded_pde(int n, int bw, const char *name) {
    int max_nnz = n * (2 * bw + 1);
    int *rows = (int *)malloc((size_t)max_nnz * sizeof(int));
    int *cols = (int *)malloc((size_t)max_nnz * sizeof(int));
    double *vals = (double *)malloc((size_t)max_nnz * sizeof(double));
    int nnz = 0;
    for (int r = 0; r < n; r++) {
        for (int d = -bw; d <= bw; d++) {
            int c = r + d;
            if (c >= 0 && c < n) {
                rows[nnz] = r;
                cols[nnz] = c;
                vals[nnz] = (d == 0) ? 2.0 : -1.0 / (double)(abs(d) + 1);
                nnz++;
            }
        }
    }
    CSRMatrix A = make_from_triplets(n, n, nnz, rows, cols, vals, name, "banded_pde");
    free(rows); free(cols); free(vals);
    return A;
}

static CSRMatrix generate_block_structured(int n, int block, int blocks_per_row, const char *name) {
    int nb = n / block;
    int cap = n * block * blocks_per_row * 2;
    int *rows = (int *)malloc((size_t)cap * sizeof(int));
    int *cols = (int *)malloc((size_t)cap * sizeof(int));
    double *vals = (double *)malloc((size_t)cap * sizeof(double));
    int nnz = 0;
    for (int br = 0; br < nb; br++) {
        for (int b = 0; b < blocks_per_row; b++) {
            int bc = (br + b) % nb;
            for (int rr = 0; rr < block; rr++) {
                for (int cc = 0; cc < block; cc++) {
                    if (nnz == cap) {
                        cap *= 2;
                        rows = (int *)realloc(rows, (size_t)cap * sizeof(int));
                        cols = (int *)realloc(cols, (size_t)cap * sizeof(int));
                        vals = (double *)realloc(vals, (size_t)cap * sizeof(double));
                    }
                    rows[nnz] = br * block + rr;
                    cols[nnz] = bc * block + cc;
                    vals[nnz] = 0.2 + rand_unit();
                    nnz++;
                }
            }
        }
    }
    CSRMatrix A = make_from_triplets(n, n, nnz, rows, cols, vals, name, "block_structured");
    free(rows); free(cols); free(vals);
    return A;
}

static CSRMatrix generate_fem_like(int n, const char *name) {
    int m = (int)sqrt((double)n);
    if (m < 2) m = 2;
    int N = m * m;
    int max_nnz = N * 9;
    int *rows = (int *)malloc((size_t)max_nnz * sizeof(int));
    int *cols = (int *)malloc((size_t)max_nnz * sizeof(int));
    double *vals = (double *)malloc((size_t)max_nnz * sizeof(double));
    int nnz = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            int r = i * m + j;
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < m && nj >= 0 && nj < m) {
                        int c = ni * m + nj;
                        rows[nnz] = r;
                        cols[nnz] = c;
                        vals[nnz] = (di == 0 && dj == 0) ? 4.0 : -0.5;
                        nnz++;
                    }
                }
            }
        }
    }
    CSRMatrix A = make_from_triplets(N, N, nnz, rows, cols, vals, name, "fem_mesh");
    free(rows); free(cols); free(vals);
    return A;
}

static double estimate_stream_bw_gbs(int n) {
    double *a = (double *)malloc((size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *c = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        a[i] = 1.0; b[i] = 2.0; c[i] = 0.0;
    }
    double best = 1e30;
    for (int rep = 0; rep < 8; rep++) {
        double t0 = now_sec();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) c[i] = a[i] + 3.0 * b[i];
        double t1 = now_sec();
        double dt = t1 - t0;
        if (rep > 1 && dt < best) best = dt;
    }
    double bytes = 3.0 * (double)n * sizeof(double);
    free(a); free(b); free(c);
    return (bytes / best) / 1e9;
}

static double csr_bytes_per_spmv(const CSRMatrix *A) {
    return (double)A->nnz * (sizeof(double) + sizeof(int) + sizeof(double)) +
           (double)A->nrows * sizeof(double);
}

static double ell_bytes_per_spmv(const ELLMatrix *E) {
    double elems = (double)E->nrows * (double)E->max_row_nnz;
    return elems * (sizeof(double) + sizeof(int) + sizeof(double)) + (double)E->nrows * sizeof(double);
}

static double bcsr_bytes_per_spmv(const BCSRMatrix *B) {
    double block_vals = (double)B->nb * (double)(B->br * B->bc);
    return block_vals * sizeof(double) +
           (double)B->nb * sizeof(int) +
           block_vals * sizeof(double) +
           (double)B->nrows * sizeof(double);
}

static void run_for_matrix(FILE *fout, const CSRMatrix *A, const Config *cfg, double stream_bw) {
    double *x = (double *)malloc((size_t)A->ncols * sizeof(double));
    double *y = (double *)malloc((size_t)A->nrows * sizeof(double));
    double *y_ref = (double *)malloc((size_t)A->nrows * sizeof(double));
    double *y_tmp = (double *)malloc((size_t)A->nrows * sizeof(double));

    for (int i = 0; i < A->ncols; i++) x[i] = rand_unit();
    dense_reference(A, x, y_ref);

    csr_spmv(A, x, y_tmp);
    if (max_abs_diff(y_ref, y_tmp, A->nrows) > 1e-8) {
        fprintf(stderr, "CSR correctness failed for %s\n", A->name);
        exit(2);
    }

    ELLMatrix E = csr_to_ell(A);
    ell_spmv(&E, x, y_tmp, cfg->use_avx2);
    if (max_abs_diff(y_ref, y_tmp, A->nrows) > 1e-7) {
        fprintf(stderr, "ELLPACK correctness failed for %s\n", A->name);
        exit(2);
    }

    BCSRMatrix B2 = csr_to_bcsr(A, 2, 2);
    bcsr_spmv(&B2, x, y_tmp, cfg->use_avx2);
    if (max_abs_diff(y_ref, y_tmp, A->nrows) > 1e-6) {
        fprintf(stderr, "BCSR2 correctness failed for %s\n", A->name);
        exit(2);
    }

    BCSRMatrix B4 = csr_to_bcsr(A, 4, 4);
    bcsr_spmv(&B4, x, y_tmp, cfg->use_avx2);
    if (max_abs_diff(y_ref, y_tmp, A->nrows) > 1e-6) {
        fprintf(stderr, "BCSR4 correctness failed for %s\n", A->name);
        exit(2);
    }

    for (int th = cfg->min_threads; th <= cfg->max_threads; th++) {
        omp_set_num_threads(th);
        double runs[128];

        for (int w = 0; w < cfg->warmup_runs; w++) csr_spmv(A, x, y);
        for (int r = 0; r < cfg->timed_runs; r++) {
            double t0 = now_sec();
            csr_spmv(A, x, y);
            double t1 = now_sec();
            runs[r] = t1 - t0;
        }
        double t = median(runs, cfg->timed_runs);
        double flops = 2.0 * (double)A->nnz;
        double gflops = flops / t / 1e9;
        double bw = csr_bytes_per_spmv(A) / t / 1e9;
        fprintf(fout, "%s,%s,%d,%d,%d,CSR,%d,%.9f,%.6f,%.6f,%.6f,%.6f\n",
            A->name, A->family, A->nrows, A->ncols, A->nnz, th, t, gflops, bw, flops / csr_bytes_per_spmv(A), stream_bw);

        for (int w = 0; w < cfg->warmup_runs; w++) ell_spmv(&E, x, y, cfg->use_avx2);
        for (int r = 0; r < cfg->timed_runs; r++) {
            double t0 = now_sec();
            ell_spmv(&E, x, y, cfg->use_avx2);
            double t1 = now_sec();
            runs[r] = t1 - t0;
        }
        t = median(runs, cfg->timed_runs);
        gflops = flops / t / 1e9;
        bw = ell_bytes_per_spmv(&E) / t / 1e9;
        fprintf(fout, "%s,%s,%d,%d,%d,ELLPACK,%d,%.9f,%.6f,%.6f,%.6f,%.6f\n",
            A->name, A->family, A->nrows, A->ncols, A->nnz, th, t, gflops, bw, flops / ell_bytes_per_spmv(&E), stream_bw);

        for (int w = 0; w < cfg->warmup_runs; w++) bcsr_spmv(&B2, x, y, cfg->use_avx2);
        for (int r = 0; r < cfg->timed_runs; r++) {
            double t0 = now_sec();
            bcsr_spmv(&B2, x, y, cfg->use_avx2);
            double t1 = now_sec();
            runs[r] = t1 - t0;
        }
        t = median(runs, cfg->timed_runs);
        gflops = flops / t / 1e9;
        bw = bcsr_bytes_per_spmv(&B2) / t / 1e9;
        fprintf(fout, "%s,%s,%d,%d,%d,BCSR2x2,%d,%.9f,%.6f,%.6f,%.6f,%.6f\n",
            A->name, A->family, A->nrows, A->ncols, A->nnz, th, t, gflops, bw, flops / bcsr_bytes_per_spmv(&B2), stream_bw);

        for (int w = 0; w < cfg->warmup_runs; w++) bcsr_spmv(&B4, x, y, cfg->use_avx2);
        for (int r = 0; r < cfg->timed_runs; r++) {
            double t0 = now_sec();
            bcsr_spmv(&B4, x, y, cfg->use_avx2);
            double t1 = now_sec();
            runs[r] = t1 - t0;
        }
        t = median(runs, cfg->timed_runs);
        gflops = flops / t / 1e9;
        bw = bcsr_bytes_per_spmv(&B4) / t / 1e9;
        fprintf(fout, "%s,%s,%d,%d,%d,BCSR4x4,%d,%.9f,%.6f,%.6f,%.6f,%.6f\n",
            A->name, A->family, A->nrows, A->ncols, A->nnz, th, t, gflops, bw, flops / bcsr_bytes_per_spmv(&B4), stream_bw);
    }

    free_bcsr(&B4);
    free_bcsr(&B2);
    free_ell(&E);
    free(x); free(y); free(y_ref); free(y_tmp);
}

int main(int argc, char **argv) {
    Config cfg;
    cfg.warmup_runs = 5;
    cfg.timed_runs = 20;
    cfg.min_threads = 1;
    cfg.max_threads = 32;
    cfg.use_avx2 = 1;
    cfg.output_csv = "results/benchmark_results.csv";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) cfg.warmup_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--timed") == 0 && i + 1 < argc) cfg.timed_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-threads") == 0 && i + 1 < argc) cfg.min_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-threads") == 0 && i + 1 < argc) cfg.max_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-avx2") == 0) cfg.use_avx2 = 0;
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) cfg.output_csv = argv[++i];
    }

    int hw_threads = omp_get_max_threads();
    if (cfg.max_threads > hw_threads) {
        fprintf(stderr, "Requested max threads=%d exceeds available=%d on this node; capping.\n", cfg.max_threads, hw_threads);
        cfg.max_threads = hw_threads;
    }

    srand(7);
    FILE *fout = fopen(cfg.output_csv, "w");
    if (!fout) {
        fprintf(stderr, "Failed to open output CSV: %s\n", cfg.output_csv);
        return 1;
    }
    fprintf(fout, "matrix,family,nrows,ncols,nnz,format,threads,time_sec,gflops,bandwidth_gbs,arith_intensity,stream_peak_gbs\n");

    double stream_bw = estimate_stream_bw_gbs(10 * 1000 * 1000);
    fprintf(stderr, "Estimated STREAM peak bandwidth: %.3f GB/s\n", stream_bw);

    CSRMatrix mats[12];
    mats[0] = generate_irregular_graph(3000, 12, "graph_small");
    mats[1] = generate_irregular_graph(4500, 14, "graph_medium");
    mats[2] = generate_irregular_graph(6000, 16, "graph_large");
    mats[3] = generate_fem_like(55 * 55, "fem_small");
    mats[4] = generate_fem_like(70 * 70, "fem_medium");
    mats[5] = generate_fem_like(85 * 85, "fem_large");
    mats[6] = generate_block_structured(2048, 4, 3, "block_small");
    mats[7] = generate_block_structured(3072, 4, 3, "block_medium");
    mats[8] = generate_block_structured(4096, 4, 4, "block_large");
    mats[9] = generate_banded_pde(4000, 3, "band_small");
    mats[10] = generate_banded_pde(6000, 5, "band_medium");
    mats[11] = generate_banded_pde(8000, 7, "band_large");

    for (int i = 0; i < 12; i++) {
        fprintf(stderr, "Running matrix %d/12: %s (%s), nnz=%d\n", i + 1, mats[i].name, mats[i].family, mats[i].nnz);
        run_for_matrix(fout, &mats[i], &cfg, stream_bw);
    }
    fclose(fout);

    for (int i = 0; i < 12; i++) free_csr(&mats[i]);
    fprintf(stderr, "Benchmark complete. Output: %s\n", cfg.output_csv);
    return 0;
}
