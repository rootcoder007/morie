// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- dense linear algebra, elementwise, and FFT kernels
// (binding-agnostic; companion to morie_core.hpp and held to the same
// contract: raw pointers + sizes, C++ standard library only, no I/O,
// no exit(), no C rand() -- CRAN-compliant per Writing R Extensions
// sections 1.6.4 ("Portable C and C++ code") and 6 ("The R API"):
// compiled code called from R must not write to stdout/stderr nor
// terminate the process, and these kernels do neither.
//
// Matrices are ROW-MAJOR throughout, matching morie's Python
// _array_core layout; the Rcpp binding transposes at the boundary
// (R is column-major, WRE 5.2). Every routine writes into
// caller-provided buffers so ownership stays with the binding layer.
//
// References
// ----------
// Golub GH, Van Loan CF (2013). Matrix Computations, 4th ed., Johns
//   Hopkins UP. LU with partial pivoting: Algorithm 3.4.1; the
//   growth-factor argument for partial pivoting: Sec 3.4.6.
// Cooley JW, Tukey JW (1965). "An algorithm for the machine
//   calculation of complex Fourier series." Math. Comp. 19:297-301.
//   (iterative radix-2 decimation-in-time form below).
// Bluestein L (1970). "A linear filtering approach to the computation
//   of discrete Fourier transform." IEEE Trans. Audio
//   Electroacoust. 18(4):451-455. (arbitrary-n via chirp-z).
// Welford BP (1962). "Note on a method for calculating corrected sums
//   of squares and products." Technometrics 4(3):419-420.
//
// CANONICAL COPY: libmorie/linalg_core.hpp. The R package vendors a
// copy at r-package/morie/src/linalg_core.h -- keep the two in sync.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace morie {
namespace core {

// --- elementwise (out may alias a) ------------------------------------------

inline void ew_add(const double *a, const double *b, double *out,
                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

inline void ew_sub(const double *a, const double *b, double *out,
                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

inline void ew_mul(const double *a, const double *b, double *out,
                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

inline void ew_div(const double *a, const double *b, double *out,
                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
}

inline void ew_scale(const double *a, double s, double c, double *out,
                     std::size_t n) {
    // out = s * a + c  (fused scale-shift; covers +, -, *, / by scalar)
    for (std::size_t i = 0; i < n; ++i) out[i] = s * a[i] + c;
}

inline void ew_exp(const double *a, double *out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = std::exp(a[i]);
}

inline void ew_log(const double *a, double *out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = std::log(a[i]);
}

inline void ew_sqrt(const double *a, double *out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = std::sqrt(a[i]);
}

// --- reductions --------------------------------------------------------------

// Kahan-compensated sum: bounds the error independent of n, which the
// pure-Python arm gets from math.fsum; the compiled arm must not be
// LESS accurate than the reference arm.
inline double ksum(const double *a, std::size_t n) {
    double s = 0.0, c = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double y = a[i] - c;
        const double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

inline double dot(const double *a, const double *b, std::size_t n) {
    double s = 0.0, c = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double y = a[i] * b[i] - c;
        const double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

// Welford (1962) single-pass mean/variance -- numerically stable for
// the long, similar-magnitude series morie feeds it.
inline void welford(const double *a, std::size_t n, int ddof,
                    double *mean_out, double *var_out) {
    double m = 0.0, m2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d1 = a[i] - m;
        m += d1 / static_cast<double>(i + 1);
        m2 += d1 * (a[i] - m);
    }
    *mean_out = m;
    *var_out = (n > static_cast<std::size_t>(ddof))
                   ? m2 / static_cast<double>(n - ddof)
                   : 0.0;
}

// --- matmul (row-major) -------------------------------------------------------

// C(n x p) = A(n x m) @ B(m x p). ikj loop order keeps the inner loop
// contiguous over both B and C rows (Golub & Van Loan Sec 1.1 on
// blocked/ordered gaxpy forms); no aliasing permitted between out and
// the inputs.
inline void matmul(const double *A, const double *B, double *out,
                   std::size_t n, std::size_t m, std::size_t p) {
    for (std::size_t i = 0; i < n * p; ++i) out[i] = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < m; ++k) {
            const double aik = A[i * m + k];
            if (aik == 0.0) continue;
            const double *brow = B + k * p;
            double *crow = out + i * p;
            for (std::size_t j = 0; j < p; ++j) crow[j] += aik * brow[j];
        }
    }
}

// --- solve / inverse ---------------------------------------------------------

// Gaussian elimination with partial pivoting (Golub & Van Loan
// Alg. 3.4.1) on the augmented system; A is n x n row-major, b is
// n x k right-hand sides. Returns false on (numerical) singularity --
// the binding raises LinAlgError; the kernel itself never aborts
// (CRAN: compiled code must not terminate the process).
inline bool solve_gauss(const double *A, const double *b, double *out,
                        std::size_t n, std::size_t k) {
    std::vector<double> aug(n * (n + k));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) aug[i * (n + k) + j] = A[i * n + j];
        for (std::size_t j = 0; j < k; ++j)
            aug[i * (n + k) + n + j] = b[i * k + j];
    }
    const std::size_t w = n + k;
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        double best = std::fabs(aug[col * w + col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double v = std::fabs(aug[r * w + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return false;
        if (piv != col) {
            for (std::size_t j = 0; j < w; ++j)
                std::swap(aug[col * w + j], aug[piv * w + j]);
        }
        const double d = aug[col * w + col];
        for (std::size_t j = col; j < w; ++j) aug[col * w + j] /= d;
        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = aug[r * w + col];
            if (f == 0.0) continue;
            for (std::size_t j = col; j < w; ++j)
                aug[r * w + j] -= f * aug[col * w + j];
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < k; ++j) out[i * k + j] = aug[i * w + n + j];
    return true;
}

// --- FFT ---------------------------------------------------------------------

namespace detail {

// Iterative radix-2 decimation-in-time Cooley-Tukey (1965); n must be
// a power of two. In-place on interleaved std::complex<double>.
inline void fft_pow2(std::complex<double> *a, std::size_t n, bool invert) {
    // bit-reversal permutation
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double kTwoPi = 6.28318530717958647692;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = (invert ? kTwoPi : -kTwoPi) /
                           static_cast<double>(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

}  // namespace detail

// DFT of arbitrary length via Bluestein (1970) chirp-z when n is not
// a power of two; direct Cooley-Tukey otherwise. re/im are length-n
// input and output buffers (out-of-place).
inline void fft(const double *re_in, const double *im_in, double *re_out,
                double *im_out, std::size_t n, bool invert) {
    if (n == 0) return;
    const bool pow2 = (n & (n - 1)) == 0;
    if (pow2) {
        std::vector<std::complex<double>> a(n);
        for (std::size_t i = 0; i < n; ++i)
            a[i] = std::complex<double>(re_in[i], im_in[i]);
        detail::fft_pow2(a.data(), n, invert);
        for (std::size_t i = 0; i < n; ++i) {
            re_out[i] = a[i].real();
            im_out[i] = a[i].imag();
        }
        return;
    }
    // Bluestein: x_k * w_k convolved with conj chirp, via pow2 FFTs.
    const double kPiL = 3.14159265358979323846;
    const double sign = invert ? 1.0 : -1.0;
    std::vector<std::complex<double>> w(n);
    for (std::size_t k = 0; k < n; ++k) {
        // k*k mod 2n keeps the chirp argument bounded (exact in
        // double for the sizes morie uses).
        const std::size_t kk = (k * k) % (2 * n);
        const double ang = sign * kPiL * static_cast<double>(kk) /
                           static_cast<double>(n);
        w[k] = std::complex<double>(std::cos(ang), std::sin(ang));
    }
    std::size_t m = 1;
    while (m < 2 * n - 1) m <<= 1;
    std::vector<std::complex<double>> fa(m), fb(m);
    for (std::size_t k = 0; k < n; ++k)
        fa[k] = std::complex<double>(re_in[k], im_in[k]) * w[k];
    fb[0] = std::conj(w[0]);
    for (std::size_t k = 1; k < n; ++k)
        fb[k] = fb[m - k] = std::conj(w[k]);
    detail::fft_pow2(fa.data(), m, false);
    detail::fft_pow2(fb.data(), m, false);
    for (std::size_t i = 0; i < m; ++i) fa[i] *= fb[i];
    detail::fft_pow2(fa.data(), m, true);
    const double inv_m = 1.0 / static_cast<double>(m);
    for (std::size_t k = 0; k < n; ++k) {
        const std::complex<double> v = fa[k] * inv_m * w[k];
        re_out[k] = v.real();
        im_out[k] = v.imag();
    }
}

}  // namespace core
}  // namespace morie
