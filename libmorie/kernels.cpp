// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- nanobind (Python) bindings for the numeric kernels.
//
// Thin adapter layer: extracts raw pointers from numpy arrays and
// calls the binding-agnostic core in morie_core.hpp. The same core
// functions are bound for R via Rcpp -- one numeric source of truth.

#include "kernels.h"
#include "morie_core.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace core = morie::core;

namespace {

// A read-only, contiguous 1-D float64 array (the Python shim coerces
// every input to this layout before calling in).
using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
// Results go back as owned float64 bytes the Python shim wraps (morie.fn's
// frombuffer): a numpy-typed ndarray could not be exported on an install
// without numpy, which the plain wheel is.
using OutArray = nb::bytes;

static OutArray to_bytes(const std::vector<double> &v) {
    return nb::bytes(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(double));
}

OutArray normal_pdf(Vec x, double mean, double sd) {
    const std::size_t n = x.shape(0);
    std::vector<double> out(n);
    core::normal_pdf(x.data(), n, mean, sd, out.data());
    return to_bytes(out);
}

OutArray normal_logpdf(Vec x, double mean, double sd) {
    const std::size_t n = x.shape(0);
    std::vector<double> out(n);
    core::normal_logpdf(x.data(), n, mean, sd, out.data());
    return to_bytes(out);
}

double mean_jit(Vec a) { return core::mean(a.data(), a.shape(0)); }

double var_jit(Vec a, int ddof) {
    return core::variance(a.data(), a.shape(0), ddof);
}

double std_jit(Vec a, int ddof) {
    return core::stddev(a.data(), a.shape(0), ddof);
}

double cor_pearson_jit(Vec x, Vec y) {
    if (x.shape(0) != y.shape(0)) return std::nan("");
    return core::cor_pearson(x.data(), y.data(), x.shape(0));
}

double euclid_dist_jit(Vec a, Vec b) {
    if (a.shape(0) != b.shape(0)) return std::nan("");
    return core::euclid_dist(a.data(), b.data(), a.shape(0));
}

OutArray trimmed_ipw_weights_jit(Vec treat, Vec propensity, double trim_lo,
                                 double trim_hi) {
    const std::size_t n = treat.shape(0);
    std::vector<double> out(n);
    core::trimmed_ipw_weights(treat.data(), propensity.data(), n, trim_lo,
                              trim_hi, out.data());
    return to_bytes(out);
}

OutArray bootstrap_mean_jit(Vec a, long long B, long long seed) {
    const std::size_t nB = static_cast<std::size_t>(B);
    std::vector<double> out(nB);
    core::bootstrap_mean(a.data(), a.shape(0), nB,
                         static_cast<unsigned long long>(seed), out.data());
    return to_bytes(out);
}

}  // namespace

void register_kernels(nb::module_ &m) {
    m.def("normal_pdf", &normal_pdf, "x"_a, "mean"_a, "sd"_a,
          "Normal PDF over a 1-D float64 array.");
    m.def("normal_logpdf", &normal_logpdf, "x"_a, "mean"_a, "sd"_a,
          "Normal log-density over a 1-D float64 array.");
    m.def("mean_jit", &mean_jit, "arr"_a, "Arithmetic mean of a 1-D array.");
    m.def("var_jit", &var_jit, "arr"_a, "ddof"_a = 1,
          "Sample variance with ddof (two-pass).");
    m.def("std_jit", &std_jit, "arr"_a, "ddof"_a = 1,
          "Sample standard deviation with ddof.");
    m.def("cor_pearson_jit", &cor_pearson_jit, "x"_a, "y"_a,
          "Pearson correlation coefficient.");
    m.def("euclid_dist_jit", &euclid_dist_jit, "a"_a, "b"_a,
          "Euclidean (L2) distance between two equal-length vectors.");
    m.def("trimmed_ipw_weights_jit", &trimmed_ipw_weights_jit, "treat"_a,
          "propensity"_a, "trim_lo"_a = 0.01, "trim_hi"_a = 0.99,
          "IPW weights with propensity-score clipping.");
    m.def("bootstrap_mean_jit", &bootstrap_mean_jit, "a"_a, "B"_a, "seed"_a,
          "B bootstrap-replicate means of `a` (std::mt19937_64, "
          "reproducible per seed).");
}
