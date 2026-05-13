// morie_fast.cpp — Rcpp-accelerated hot kernels for the R side of morie.
//
// This file is the v0.6.0 R-side counterpart of src/morie/_jit.py (Python
// Numba kernels).  Each function here mirrors a Python `morie.fast.*` kernel
// so users get the same numerical results across both languages without a
// Numba-or-Python branch in their own code.
//
// Compile via R's standard mechanism (R CMD INSTALL).  Activation: add
// `LinkingTo: Rcpp` to DESCRIPTION; users get the fast path automatically
// when the package is built with a C++ toolchain present (which is the
// CRAN-builder default).  Without a C++ toolchain at install time, R falls
// back to the pure-R implementations in R/_fast.R.
//
// Currently shipped:
//   - morie_normal_pdf_cpp(x, mean, sd)
//   - morie_mean_cpp(x)
//   - morie_var_cpp(x, ddof)
//   - morie_cor_pearson_cpp(x, y)
//
// All four are numerically identical to base R (and to the Python Numba
// kernels) but skip R's lazy-list overhead on hot paths.

#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

static const double kInvSqrt2Pi = 0.3989422804014327;     // 1 / sqrt(2*pi)
static const double kLogSqrt2Pi = 0.9189385332046727;     // 0.5 * log(2*pi)

// [[Rcpp::export]]
NumericVector morie_normal_pdf_cpp(NumericVector x, double mean, double sd) {
    if (sd <= 0.0) {
        Rcpp::stop("sd must be positive");
    }
    const double inv_sigma = 1.0 / sd;
    const R_xlen_t n = x.size();
    NumericVector out(n);
    for (R_xlen_t i = 0; i < n; ++i) {
        const double z = (x[i] - mean) * inv_sigma;
        out[i] = inv_sigma * kInvSqrt2Pi * std::exp(-0.5 * z * z);
    }
    return out;
}

// [[Rcpp::export]]
double morie_mean_cpp(NumericVector x) {
    const R_xlen_t n = x.size();
    if (n == 0) return NA_REAL;
    double s = 0.0;
    for (R_xlen_t i = 0; i < n; ++i) s += x[i];
    return s / static_cast<double>(n);
}

// [[Rcpp::export]]
double morie_var_cpp(NumericVector x, int ddof = 1) {
    const R_xlen_t n = x.size();
    if (n - ddof <= 0) return NA_REAL;
    const double m = morie_mean_cpp(x);
    double sq = 0.0;
    for (R_xlen_t i = 0; i < n; ++i) {
        const double d = x[i] - m;
        sq += d * d;
    }
    return sq / static_cast<double>(n - ddof);
}

// [[Rcpp::export]]
double morie_cor_pearson_cpp(NumericVector x, NumericVector y) {
    const R_xlen_t n = x.size();
    if (n != y.size() || n < 2) return NA_REAL;
    double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (R_xlen_t i = 0; i < n; ++i) {
        const double a = x[i], b = y[i];
        sx  += a;
        sy  += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    const double num = static_cast<double>(n) * sxy - sx * sy;
    const double den_sq = (static_cast<double>(n) * sxx - sx * sx)
                        * (static_cast<double>(n) * syy - sy * sy);
    if (den_sq <= 0.0) return NA_REAL;
    return num / std::sqrt(den_sq);
}
