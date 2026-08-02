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

#include <algorithm>
#include <cmath>
#include <limits>
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


// ---------------------------------------------------------------------
// One-sided Jacobi SVD (Golub & Van Loan, 4th ed., Sec. 8.6.3; Demmel &
// Veselic 1992, "Jacobi's method is more accurate than QR"). Computes
// A = U S V^T for a row-major m x n matrix with m >= n to high RELATIVE
// accuracy in every singular value -- unlike the eig(A^T A) route,
// which loses the values below sqrt(eps)*s_max.
//
// a    : in/out, m*n row-major; overwritten with U*diag(S) columns.
// v    : out, n*n row-major; receives V (not V^T).
// sv   : out, n singular values, descending.
// Returns the number of sweeps used (<= max_sweeps; convergence is
// |a_p . a_q| <= tol * ||a_p|| * ||a_q|| for every column pair).
inline int jacobi_svd(double *a, std::size_t m, std::size_t n,
                      double *v, double *sv,
                      double tol = 1e-15, int max_sweeps = 60) {
    // v = I
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            v[i * n + j] = (i == j) ? 1.0 : 0.0;

    int sweep = 0;
    for (; sweep < max_sweeps; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                double app = 0.0, aqq = 0.0, apq = 0.0;
                for (std::size_t r = 0; r < m; ++r) {
                    const double x = a[r * n + p], y = a[r * n + q];
                    app += x * x;
                    aqq += y * y;
                    apq += x * y;
                }
                if (app == 0.0 || aqq == 0.0) continue;
                if (std::fabs(apq) <= tol * std::sqrt(app) * std::sqrt(aqq))
                    continue;
                rotated = true;
                const double zeta = (aqq - app) / (2.0 * apq);
                const double t =
                    (zeta >= 0.0 ? 1.0 : -1.0) /
                    (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;
                for (std::size_t r = 0; r < m; ++r) {
                    const double x = a[r * n + p], y = a[r * n + q];
                    a[r * n + p] = c * x - s * y;
                    a[r * n + q] = s * x + c * y;
                }
                for (std::size_t r = 0; r < n; ++r) {
                    const double x = v[r * n + p], y = v[r * n + q];
                    v[r * n + p] = c * x - s * y;
                    v[r * n + q] = s * x + c * y;
                }
            }
        }
        if (!rotated) break;
    }

    // Singular values = column norms; sort descending and permute the
    // columns of both a (=> U*S) and v to match.
    std::vector<std::size_t> ord(n);
    for (std::size_t i = 0; i < n; ++i) {
        double nrm = 0.0;
        for (std::size_t r = 0; r < m; ++r) {
            const double x = a[r * n + i];
            nrm += x * x;
        }
        sv[i] = std::sqrt(nrm);
        ord[i] = i;
    }
    std::sort(ord.begin(), ord.end(),
              [&](std::size_t i, std::size_t j) { return sv[i] > sv[j]; });
    std::vector<double> tmp_s(sv, sv + n);
    std::vector<double> tmp_a(a, a + m * n), tmp_v(v, v + n * n);
    for (std::size_t k = 0; k < n; ++k) {
        sv[k] = tmp_s[ord[k]];
        for (std::size_t r = 0; r < m; ++r)
            a[r * n + k] = tmp_a[r * n + ord[k]];
        for (std::size_t r = 0; r < n; ++r)
            v[r * n + k] = tmp_v[r * n + ord[k]];
    }
    return sweep;
}


// ---------------------------------------------------------------------
// Eigenvalues of a general real matrix: Householder reduction to upper
// Hessenberg form followed by the shifted QR iteration on the
// Hessenberg matrix (Golub & Van Loan, 4th ed., Alg. 7.4.2 + Sec. 7.5;
// the QR stage follows the classic EISPACK hqr scheme, Martin,
// Peters & Wilkinson 1970, Numer. Math. 14).
// a: in/out n*n row-major (destroyed). wr/wi: out, n eigenvalue parts.
// Returns true on convergence (30*n iteration budget, as in EISPACK).
inline bool eig_general(double *a, std::size_t n, double *wr, double *wi) {
    const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(n);
    auto A = [&](std::ptrdiff_t r, std::ptrdiff_t c) -> double & {
        return a[r * N + c];
    };

    // -- Householder reduction to upper Hessenberg (in place) --
    for (std::ptrdiff_t k = 1; k + 1 < N; ++k) {
        double scale = 0.0;
        for (std::ptrdiff_t i = k; i < N; ++i)
            scale += std::fabs(A(i, k - 1));
        if (scale == 0.0) continue;
        double h = 0.0;
        std::vector<double> u(static_cast<std::size_t>(N), 0.0);
        for (std::ptrdiff_t i = N - 1; i >= k; --i) {
            u[static_cast<std::size_t>(i)] = A(i, k - 1) / scale;
            h += u[static_cast<std::size_t>(i)] *
                 u[static_cast<std::size_t>(i)];
        }
        double g = std::sqrt(h);
        if (u[static_cast<std::size_t>(k)] > 0.0) g = -g;
        h -= u[static_cast<std::size_t>(k)] * g;
        u[static_cast<std::size_t>(k)] -= g;
        // apply P = I - u u^T / h from both sides
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            double f = 0.0;
            for (std::ptrdiff_t i = k; i < N; ++i)
                f += u[static_cast<std::size_t>(i)] * A(i, j);
            f /= h;
            for (std::ptrdiff_t i = k; i < N; ++i)
                A(i, j) -= f * u[static_cast<std::size_t>(i)];
        }
        for (std::ptrdiff_t i = 0; i < N; ++i) {
            double f = 0.0;
            for (std::ptrdiff_t j = k; j < N; ++j)
                f += u[static_cast<std::size_t>(j)] * A(i, j);
            f /= h;
            for (std::ptrdiff_t j = k; j < N; ++j)
                A(i, j) -= f * u[static_cast<std::size_t>(j)];
        }
        A(k, k - 1) = scale * g;
        for (std::ptrdiff_t i = k + 1; i < N; ++i) A(i, k - 1) = 0.0;
    }

    // -- Shifted QR on the Hessenberg matrix (EISPACK hqr) --
    double anorm = 0.0;
    for (std::ptrdiff_t i = 0; i < N; ++i)
        for (std::ptrdiff_t j = (i > 0 ? i - 1 : 0); j < N; ++j)
            anorm += std::fabs(A(i, j));
    std::ptrdiff_t nn = N - 1;
    double t = 0.0;
    long total_its = 0;
    const long budget = 30L * static_cast<long>(N) + 100L;
    while (nn >= 0) {
        long its = 0;
        std::ptrdiff_t l = 0;
        do {
            for (l = nn; l >= 1; --l) {
                const double s =
                    std::fabs(A(l - 1, l - 1)) + std::fabs(A(l, l));
                const double s2 = (s == 0.0) ? anorm : s;
                if (std::fabs(A(l, l - 1)) + s2 == s2) {
                    A(l, l - 1) = 0.0;
                    break;
                }
            }
            double x = A(nn, nn);
            if (l == nn) {                       // one real root found
                wr[nn] = x + t;
                wi[nn] = 0.0;
                --nn;
            } else {
                double y = A(nn - 1, nn - 1);
                double w = A(nn, nn - 1) * A(nn - 1, nn);
                if (l == nn - 1) {               // a 2x2 block
                    double p = 0.5 * (y - x);
                    const double q = p * p + w;
                    double z = std::sqrt(std::fabs(q));
                    x += t;
                    if (q >= 0.0) {              // real pair
                        z = p + (p >= 0.0 ? z : -z);
                        wr[nn - 1] = wr[nn] = x + z;
                        if (z != 0.0) wr[nn] = x - w / z;
                        wi[nn - 1] = wi[nn] = 0.0;
                    } else {                     // complex pair
                        wr[nn - 1] = wr[nn] = x + p;
                        wi[nn - 1] = -(wi[nn] = z);
                    }
                    nn -= 2;
                } else {                         // no root yet: QR step
                    double p = 0.0, q = 0.0, r = 0.0, z = 0.0, s = 0.0;
                    if (its == 10 || its == 20) {   // exceptional shift
                        t += x;
                        for (std::ptrdiff_t i = 0; i <= nn; ++i)
                            A(i, i) -= x;
                        s = std::fabs(A(nn, nn - 1)) +
                            std::fabs(A(nn - 1, nn - 2));
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    ++total_its;
                    if (total_its > budget) return false;
                    std::ptrdiff_t m = 0;
                    for (m = nn - 2; m >= l; --m) {
                        z = A(m, m);
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / A(m + 1, m) + A(m, m + 1);
                        q = A(m + 1, m + 1) - z - r - s;
                        r = A(m + 2, m + 1);
                        s = std::fabs(p) + std::fabs(q) + std::fabs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l) break;
                        const double u2 = std::fabs(A(m, m - 1)) *
                                          (std::fabs(q) + std::fabs(r));
                        const double v2 =
                            std::fabs(p) *
                            (std::fabs(A(m - 1, m - 1)) + std::fabs(z) +
                             std::fabs(A(m + 1, m + 1)));
                        if (u2 + v2 == v2) break;
                    }
                    for (std::ptrdiff_t i = m + 2; i <= nn; ++i) {
                        A(i, i - 2) = 0.0;
                        if (i != m + 2) A(i, i - 3) = 0.0;
                    }
                    for (std::ptrdiff_t k = m; k <= nn - 1; ++k) {
                        if (k != m) {
                            p = A(k, k - 1);
                            q = A(k + 1, k - 1);
                            r = (k != nn - 1) ? A(k + 2, k - 1) : 0.0;
                            x = std::fabs(p) + std::fabs(q) + std::fabs(r);
                            if (x != 0.0) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        s = std::sqrt(p * p + q * q + r * r);
                        if (p < 0.0) s = -s;
                        if (s == 0.0) continue;
                        if (k == m) {
                            if (l != m) A(k, k - 1) = -A(k, k - 1);
                        } else {
                            A(k, k - 1) = -s * x;
                        }
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;
                        for (std::ptrdiff_t j = k; j <= nn; ++j) {
                            p = A(k, j) + q * A(k + 1, j);
                            if (k != nn - 1) {
                                p += r * A(k + 2, j);
                                A(k + 2, j) -= p * z;
                            }
                            A(k + 1, j) -= p * y;
                            A(k, j) -= p * x;
                        }
                        const std::ptrdiff_t mmin =
                            (nn < k + 3) ? nn : k + 3;
                        for (std::ptrdiff_t i = l; i <= mmin; ++i) {
                            p = x * A(i, k) + y * A(i, k + 1);
                            if (k != nn - 1) {
                                p += z * A(i, k + 2);
                                A(i, k + 2) -= p * r;
                            }
                            A(i, k + 1) -= p * q;
                            A(i, k) -= p;
                        }
                    }
                }
            }
        } while (l < nn - 1 && nn >= 0);
    }
    return true;
}


// ---------------------------------------------------------------------
// Spatio-temporal Hawkes log-likelihood (exponential temporal decay,
// isotropic Gaussian spatial kernel):
//
//   lambda(t, x, y) = mu + sum_{t_j < t} alpha*beta*exp(-beta*(t-t_j))
//                          * (1 / (2*pi*sigma^2))
//                          * exp(-((x-x_j)^2 + (y-y_j)^2) / (2*sigma^2))
//
//   loglik = sum_i log lambda(t_i, x_i, y_i)
//            - mu*T*area - sum_j alpha*(1 - exp(-beta*(T - t_j)))
//
// (Reinhart 2018, "A review of self-exciting spatio-temporal point
// processes", Statist. Sci. 33(3), eq. 2.4; the compensator assumes the
// spatial density integrates to 1 over the region.)
// Events must be sorted by time. Returns the log-likelihood; the caller
// checks parameter feasibility.
inline double hawkes_st_loglik(const double *t, const double *x,
                               const double *y, std::size_t n,
                               double mu, double alpha, double beta,
                               double sigma, double T, double area) {
    const double kPi = 3.14159265358979323846;
    const double s2 = sigma * sigma;
    const double spatial_norm = 1.0 / (2.0 * kPi * s2);
    const double inv_2s2 = 1.0 / (2.0 * s2);
    double loglam = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double lam = mu;
        for (std::size_t j = 0; j < i; ++j) {
            const double dt = t[i] - t[j];
            const double dx = x[i] - x[j];
            const double dy = y[i] - y[j];
            const double d2 = dx * dx + dy * dy;
            lam += alpha * beta * std::exp(-beta * dt) * spatial_norm *
                   std::exp(-d2 * inv_2s2);
        }
        if (lam <= 0.0) return -std::numeric_limits<double>::infinity();
        loglam += std::log(lam);
    }
    double comp = mu * T * area;
    for (std::size_t j = 0; j < n; ++j)
        comp += alpha * (1.0 - std::exp(-beta * (T - t[j])));
    return loglam - comp;
}

}  // namespace core
}  // namespace morie
