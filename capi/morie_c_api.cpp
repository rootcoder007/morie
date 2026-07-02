// SPDX-License-Identifier: AGPL-3.0-or-later
//
// morie_c_api.cpp -- implementation of the C ABI façade.
//
// Every function here is a thin, allocation-free forward to the header-only
// C++ core. No algorithm logic lives in this file: it only translates C
// types to the core's calls so the SAME compiled arithmetic backs every
// language binding. If you find yourself writing math here, it belongs in
// libmorie/morie_core.hpp instead.

#include "morie_c_api.h"

#include "../libmorie/morie_core.hpp"

extern "C" {

double morie_mean(const double *a, size_t n) {
    return morie::core::mean(a, n);
}

double morie_variance(const double *a, size_t n, int ddof) {
    return morie::core::variance(a, n, ddof);
}

double morie_stddev(const double *a, size_t n, int ddof) {
    return morie::core::stddev(a, n, ddof);
}

double morie_cor_pearson(const double *x, const double *y, size_t n) {
    return morie::core::cor_pearson(x, y, n);
}

double morie_euclid_dist(const double *a, const double *b, size_t n) {
    return morie::core::euclid_dist(a, b, n);
}

void morie_normal_pdf(const double *x, size_t n, double mean, double sd,
                      double *out) {
    morie::core::normal_pdf(x, n, mean, sd, out);
}

void morie_normal_logpdf(const double *x, size_t n, double mean, double sd,
                         double *out) {
    morie::core::normal_logpdf(x, n, mean, sd, out);
}

double morie_gamma_cdf_regularized(double a, double x) {
    return morie::core::gamma_cdf_regularized(a, x);
}

const char *morie_core_version(void) {
    return "morie-core C ABI 0.1.0 (wraps libmorie/morie_core.hpp)";
}

} // extern "C"
