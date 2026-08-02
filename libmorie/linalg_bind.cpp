// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- nanobind bindings for the linalg/elementwise/FFT
// kernels (linalg_core.hpp).
//
// Interface contract (nanobind docs, "ndarray" chapter): the ndarray
// argument type accepts ANY object exporting the Python buffer
// protocol or DLPack -- including stdlib array.array('d'), which is
// what morie's pure-Python _array_core marshals through now that
// numpy is removed. Zero-copy in, owned-capsule out (the returned
// bytes objects are plain contiguous doubles the Python shim wraps
// back into marr).
//
// All kernels return through caller-visible buffers or Python bytes;
// nothing here writes to stdout/stderr or terminates the process.

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include "linalg_core.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace core = morie::core;

namespace {

using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;

// Output buffers surface as raw little-endian float64 bytes; the
// Python shim wraps them in stdlib array.array("d") (one memcpy, no
// numpy anywhere -- the nb::numpy ndarray return framework would
// import numpy at export time, which morie no longer ships).
nb::bytes make_out(const std::vector<double> &v) {
    return nb::bytes(reinterpret_cast<const char *>(v.data()),
                     v.size() * sizeof(double));
}

nb::bytes ew2(
    const Vec &a, const Vec &b, int op) {
    const std::size_t n = a.shape(0);
    if (b.shape(0) != n) throw std::invalid_argument("length mismatch");
    std::vector<double> out(n);
    switch (op) {
        case 0: core::ew_add(a.data(), b.data(), out.data(), n); break;
        case 1: core::ew_sub(a.data(), b.data(), out.data(), n); break;
        case 2: core::ew_mul(a.data(), b.data(), out.data(), n); break;
        default: core::ew_div(a.data(), b.data(), out.data(), n); break;
    }
    return make_out(out);
}

nb::bytes ew_scale_py(
    const Vec &a, double s, double c) {
    std::vector<double> out(a.shape(0));
    core::ew_scale(a.data(), s, c, out.data(), a.shape(0));
    return make_out(out);
}

double ksum_py(const Vec &a) { return core::ksum(a.data(), a.shape(0)); }

double dot_py(const Vec &a, const Vec &b) {
    if (b.shape(0) != a.shape(0))
        throw std::invalid_argument("length mismatch");
    return core::dot(a.data(), b.data(), a.shape(0));
}

std::pair<double, double> welford_py(const Vec &a, int ddof) {
    double m = 0.0, v = 0.0;
    core::welford(a.data(), a.shape(0), ddof, &m, &v);
    return {m, v};
}

nb::bytes matmul_py(
    const Vec &A, const Vec &B, std::size_t n, std::size_t m,
    std::size_t p) {
    if (A.shape(0) != n * m || B.shape(0) != m * p)
        throw std::invalid_argument("matmul: shape/buffer mismatch");
    std::vector<double> out(n * p);
    core::matmul(A.data(), B.data(), out.data(), n, m, p);
    return make_out(out);
}

nb::bytes solve_py(
    const Vec &A, const Vec &b, std::size_t n, std::size_t k) {
    if (A.shape(0) != n * n || b.shape(0) != n * k)
        throw std::invalid_argument("solve: shape/buffer mismatch");
    std::vector<double> out(n * k);
    if (!core::solve_gauss(A.data(), b.data(), out.data(), n, k))
        throw std::domain_error("singular matrix");
    return make_out(out);
}

std::pair<nb::bytes, nb::bytes> fft_py(const Vec &re, const Vec &im,
                                       bool invert) {
    const std::size_t n = re.shape(0);
    if (im.shape(0) != n) throw std::invalid_argument("length mismatch");
    std::vector<double> ro(n), io(n);
    core::fft(re.data(), im.data(), ro.data(), io.data(), n, invert);
    return {make_out(ro), make_out(io)};
}

}  // namespace

void register_linalg(nb::module_ &m) {
    m.def("ew2", &ew2, "a"_a, "b"_a, "op"_a,
          "Elementwise op on equal-length float64 buffers "
          "(0 add, 1 sub, 2 mul, 3 div).");
    m.def("ew_scale", &ew_scale_py, "a"_a, "s"_a, "c"_a,
          "Fused s*a + c over a float64 buffer.");
    m.def("ksum", &ksum_py, "a"_a,
          "Kahan-compensated sum of a float64 buffer.");
    m.def("dot", &dot_py, "a"_a, "b"_a,
          "Compensated dot product of two float64 buffers.");
    m.def("welford", &welford_py, "a"_a, "ddof"_a,
          "Single-pass (mean, variance) via Welford (1962).");
    m.def("matmul", &matmul_py, "A"_a, "B"_a, "n"_a, "m"_a, "p"_a,
          "Row-major (n x m) @ (m x p) on flat float64 buffers.");
    m.def("solve", &solve_py, "A"_a, "b"_a, "n"_a, "k"_a,
          "Gauss partial-pivot solve of A(n x n) X = b(n x k); raises "
          "on singularity.");
    m.def("fft", &fft_py, "re"_a, "im"_a, "invert"_a,
          "DFT (Cooley-Tukey pow2 / Bluestein otherwise) of a complex "
          "signal given as separate re/im buffers; returns (re, im).");
}
