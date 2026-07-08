/* SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * morie_c_api.h -- the stable C ABI for the morie numeric core.
 *
 * This is the "prominently C" face of morie. The numeric algorithms live
 * once, in the header-only C++ core (libmorie/morie_core.hpp, namespace
 * morie::core). This façade wraps them in `extern "C"` with plain C types
 * (double, size_t, raw pointers) so that ANY language that can call a C
 * function can call the exact same compiled arithmetic:
 *
 *     C / C++            -- #include this header, link libmorie_core
 *     Go                 -- cgo
 *     Rust               -- bindgen / FFI  (rmorie-cli)
 *     Python (ctypes)    -- in addition to the nanobind path
 *     R (.C/.Call)       -- in addition to the Rcpp path
 *     WASM               -- emscripten, for browser demos
 *
 * One core, one set of results. The Python<->R (and now any-language)
 * parity-bug class is eliminated by construction: everyone links the
 * same object code, nobody reimplements the math.
 *
 * ABI stability: these signatures are the contract. Add new functions;
 * do not change existing signatures. Types are C-only here (no C++ types
 * cross the boundary).
 */
#ifndef MORIE_C_API_H
#define MORIE_C_API_H

#include <stddef.h> /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* --- summary statistics (return NaN on degenerate input) ----------------- */

/* Arithmetic mean of a[0..n). NaN if n == 0. */
double morie_mean(const double *a, size_t n);

/* Variance with delta degrees of freedom `ddof` (0 = population,
 * 1 = sample). NaN if n - ddof <= 0. */
double morie_variance(const double *a, size_t n, int ddof);

/* Standard deviation = sqrt(variance). */
double morie_stddev(const double *a, size_t n, int ddof);

/* Pearson correlation of x[0..n) and y[0..n). NaN if n < 2 or a vector
 * is constant. */
double morie_cor_pearson(const double *x, const double *y, size_t n);

/* Euclidean distance between a[0..n) and b[0..n). */
double morie_euclid_dist(const double *a, const double *b, size_t n);

/* --- array-valued kernels (write n values into caller-owned `out`) -------- */

/* Normal PDF of each x[i] under N(mean, sd^2) -> out[i]. */
void morie_normal_pdf(const double *x, size_t n, double mean, double sd,
                      double *out);

/* Natural log of the normal PDF -> out[i]. */
void morie_normal_logpdf(const double *x, size_t n, double mean, double sd,
                         double *out);

/* --- special functions --------------------------------------------------- */

/* Regularized lower incomplete gamma P(a, x) (the gamma CDF). */
double morie_gamma_cdf_regularized(double a, double x);

/* --- metadata ------------------------------------------------------------ */

/* Version string of the C ABI / core it wraps. Static storage; do not
 * free. */
const char *morie_core_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MORIE_C_API_H */
