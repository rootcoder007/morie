// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- nanobind (Python) binding for the Hawkes likelihood.
//
// Thin adapter: unwraps the numpy array and calls the binding-agnostic
// core in morie_core.hpp. The same core function is bound for R via
// Rcpp.

#include "hawkes.h"
#include "morie_core.hpp"

#include <cstdint>
#include <complex>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <tuple>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using CVec =
    nb::ndarray<const std::complex<double>, nb::ndim<1>, nb::c_contig>;

// negative log-likelihood and its analytic gradient for any (baseline, kernel,
// method); see morie::core::hawkes_nll_grad. Returns (nll, [gradient]).
std::pair<double, std::vector<double>> hawkes_nll_grad(Vec t, double T, int bkind, std::vector<double> a, double eta,
                                                       int kind, std::vector<double> psi, int method, double eps,
                                                       double soe_R, double soe_delta, bool want_grad) {
    const int nb = morie::core::hawkes_baseline_n(bkind);
    const int np = kind == 0 ? 1 : 2;
    if (static_cast<int>(a.size()) != nb || static_cast<int>(psi.size()) != np)
        throw std::invalid_argument("hawkes_nll_grad: wrong number of baseline or kernel parameters");
    std::vector<double> g(want_grad ? nb + 1 + np : 0);
    const double nll = morie::core::hawkes_nll_grad(t.data(), t.shape(0), T, bkind, a.data(), eta, kind, psi.data(),
                                                    method, eps, soe_R, soe_delta, want_grad ? g.data() : nullptr);
    return {nll, g};
}

// time-rescaling residuals of a fitted process (morie::core::hawkes_rescaled)
std::vector<double> hawkes_rescaled(Vec t, double T, int bkind, std::vector<double> a, double eta, int kind,
                                    std::vector<double> psi) {
    if (static_cast<int>(a.size()) != morie::core::hawkes_baseline_n(bkind) ||
        static_cast<int>(psi.size()) != (kind == 0 ? 1 : 2))
        throw std::invalid_argument("hawkes_rescaled: wrong number of baseline or kernel parameters");
    std::vector<double> U(t.shape(0));
    morie::core::hawkes_rescaled(t.data(), t.shape(0), T, bkind, a.data(), eta, kind, psi.data(), U.data());
    return U;
}

// the intensity at each event (the E-step of EM)
std::vector<double> hawkes_intensity(Vec t, double T, int bkind, std::vector<double> a, double eta, int kind,
                                     std::vector<double> psi) {
    if (static_cast<int>(a.size()) != morie::core::hawkes_baseline_n(bkind) ||
        static_cast<int>(psi.size()) != (kind == 0 ? 1 : 2))
        throw std::invalid_argument("hawkes_intensity: wrong number of baseline or kernel parameters");
    std::vector<double> lam(t.shape(0));
    morie::core::hawkes_intensity(t.data(), t.shape(0), T, bkind, a.data(), eta, kind, psi.data(), lam.data());
    return lam;
}

// the kernel part of the EM objective for psi_new with the old weights: (Q, P, dQ/dpsi)
std::tuple<double, double, std::vector<double>> hawkes_em_pass(Vec t, Vec lam_old, double eta_old, int kind,
                                                              std::vector<double> psi_old,
                                                              std::vector<double> psi_new) {
    const std::size_t np = kind == 0 ? 1 : 2;
    if (psi_old.size() != np || psi_new.size() != np || lam_old.shape(0) != t.shape(0))
        throw std::invalid_argument("hawkes_em_pass: wrong argument sizes");
    double P = 0.0, dQ[2] = {0.0, 0.0};
    const double Q = morie::core::hawkes_em_pass(t.data(), t.shape(0), lam_old.data(), eta_old, kind, psi_old.data(),
                                                 psi_new.data(), &P, dQ);
    return {Q, P, std::vector<double>(dQ, dQ + np)};
}

// sum_j G(T - t_j; psi) and its gradient (morie::core::hawkes_cdf_sum)
std::pair<double, std::vector<double>> hawkes_cdf_sum(Vec t, double T, int kind, std::vector<double> psi) {
    const std::size_t np = kind == 0 ? 1 : 2;
    if (psi.size() != np) throw std::invalid_argument("hawkes_cdf_sum: wrong number of kernel parameters");
    double g[2] = {0.0, 0.0};
    const double S = morie::core::hawkes_cdf_sum(t.data(), t.shape(0), T, kind, psi.data(), g);
    return {S, std::vector<double>(g, g + np)};
}

// the whole fit: projected BFGS from x0 within [lo, hi] (morie::core::hawkes_fit_pbfgs)
std::tuple<std::vector<double>, double, int> hawkes_fit_pbfgs(Vec t, double T, int bkind, int kind, int method,
                                                              double eps, double soe_R, double soe_delta,
                                                              std::vector<double> lo, std::vector<double> hi,
                                                              std::vector<double> x0, int maxiter, double gtol) {
    const std::size_t d = static_cast<std::size_t>(morie::core::hawkes_baseline_n(bkind) + 1 + (kind == 0 ? 1 : 2));
    if (lo.size() != d || hi.size() != d || x0.size() != d)
        throw std::invalid_argument("hawkes_fit_pbfgs: wrong number of parameters");
    int it = 0;
    const double f = morie::core::hawkes_fit_pbfgs(t.data(), t.shape(0), T, bkind, kind, method, eps, soe_R,
                                                   soe_delta, lo.data(), hi.data(), x0.data(), maxiter, gtol, &it);
    return {x0, f, it};
}

double hawkes_ll_exp_const(Vec t, double T, double a0, double eta,
                           double beta) {
    return morie::core::hawkes_ll_exp_const(t.data(), t.shape(0), T, a0,
                                            eta, beta);
}

double hawkes_ll_weibull_const(Vec t, double T, double a0, double eta,
                               double alpha, double lam) {
    return morie::core::hawkes_ll_weibull_const(t.data(), t.shape(0), T, a0,
                                                eta, alpha, lam);
}

double hawkes_ll_weibull_const_trunc(Vec t, double T, double a0, double eta,
                                     double alpha, double lam) {
    return morie::core::hawkes_ll_weibull_const_trunc(
        t.data(), t.shape(0), T, a0, eta, alpha, lam);
}

double hawkes_ll_lomax_const(Vec t, double T, double a0, double eta,
                             double alpha, double c) {
    return morie::core::hawkes_ll_lomax_const(t.data(), t.shape(0), T, a0,
                                              eta, alpha, c);
}

double hawkes_ll_gamma_const(Vec t, double T, double a0, double eta,
                             double alpha, double beta) {
    return morie::core::hawkes_ll_gamma_const(t.data(), t.shape(0), T, a0,
                                              eta, alpha, beta);
}

double hawkes_ll_gamma_const_trunc(Vec t, double T, double a0, double eta,
                                   double alpha, double beta) {
    return morie::core::hawkes_ll_gamma_const_trunc(
        t.data(), t.shape(0), T, a0, eta, alpha, beta);
}

double hawkes_ll_exp_sin(Vec t, double T, double a0, double a1, double a2,
                         double a3, double eta, double beta, Vec grid,
                         Vec grid_vals) {
    return morie::core::hawkes_ll_exp_sin(
        t.data(), t.shape(0), T, a0, a1, a2, a3, eta, beta, grid.data(),
        grid_vals.data(), grid.shape(0));
}

double hawkes_ll_weibull_sin(Vec t, double T, double a0, double a1, double a2,
                             double a3, double eta, double alpha, double lam,
                             Vec grid, Vec grid_vals) {
    return morie::core::hawkes_ll_weibull_sin(
        t.data(), t.shape(0), T, a0, a1, a2, a3, eta, alpha, lam,
        grid.data(), grid_vals.data(), grid.shape(0));
}

double hawkes_ll_lomax_sin(Vec t, double T, double a0, double a1, double a2,
                           double a3, double eta, double alpha, double c,
                           Vec grid, Vec grid_vals) {
    return morie::core::hawkes_ll_lomax_sin(
        t.data(), t.shape(0), T, a0, a1, a2, a3, eta, alpha, c,
        grid.data(), grid_vals.data(), grid.shape(0));
}

// User-callback bridge: g_addr / G_addr are native function-pointer
// addresses (e.g. from numba @cfunc) for the triggering kernel and its
// integral. They are cast back to plain function pointers and called
// inside the C++ O(n^2) loop, GIL-free.
double hawkes_ll_custom(Vec t, double T, double nu, double eta,
                        std::uintptr_t g_addr, std::uintptr_t G_addr) {
    auto g = reinterpret_cast<morie::core::HawkesKernelFn>(g_addr);
    auto G = reinterpret_cast<morie::core::HawkesKernelFn>(G_addr);
    return morie::core::hawkes_ll_custom(t.data(), t.shape(0), T, nu, eta,
                                         g, G);
}

double hawkes_ll_soe(Vec t, double T, double nu, double eta, Vec w,
                     Vec beta) {
    return morie::core::hawkes_ll_soe(t.data(), t.shape(0), T, nu, eta,
                                      w.data(), beta.data(), w.shape(0));
}

double hawkes_ll_soe_cplx(Vec t, double T, double nu, double eta, CVec w,
                          CVec beta) {
    return morie::core::hawkes_ll_soe_cplx(t.data(), t.shape(0), T, nu, eta,
                                           w.data(), beta.data(),
                                           w.shape(0));
}

double hawkes_ll_gamma_hybrid(Vec t, double T, double a0, double eta,
                              double alpha, double beta, double u_split,
                              CVec w_soe, CVec beta_soe) {
    return morie::core::hawkes_ll_gamma_hybrid(
        t.data(), t.shape(0), T, a0, eta, alpha, beta, u_split,
        w_soe.data(), beta_soe.data(), w_soe.shape(0));
}

// Split real/imag variants: callers without a complex128 buffer type
// (the native array core is float64-only) pass four double arrays.
// Semantics identical to the CVec forms above.
double hawkes_ll_soe_cplx_ri(Vec t, double T, double nu, double eta,
                             Vec w_re, Vec w_im, Vec b_re, Vec b_im) {
    const std::size_t m = w_re.shape(0);
    std::vector<std::complex<double>> w(m), b(m);
    for (std::size_t i = 0; i < m; ++i) {
        w[i] = {w_re.data()[i], w_im.data()[i]};
        b[i] = {b_re.data()[i], b_im.data()[i]};
    }
    return morie::core::hawkes_ll_soe_cplx(t.data(), t.shape(0), T, nu, eta,
                                           w.data(), b.data(), m);
}

double hawkes_ll_gamma_hybrid_ri(Vec t, double T, double a0, double eta,
                                 double alpha, double beta, double u_split,
                                 Vec w_re, Vec w_im, Vec b_re, Vec b_im) {
    const std::size_t m = w_re.shape(0);
    std::vector<std::complex<double>> w(m), b(m);
    for (std::size_t i = 0; i < m; ++i) {
        w[i] = {w_re.data()[i], w_im.data()[i]};
        b[i] = {b_re.data()[i], b_im.data()[i]};
    }
    return morie::core::hawkes_ll_gamma_hybrid(
        t.data(), t.shape(0), T, a0, eta, alpha, beta, u_split,
        w.data(), b.data(), m);
}

}  // namespace

void register_hawkes(nb::module_ &m) {
    m.def("hawkes_intensity", &hawkes_intensity, "t"_a, "T"_a, "bkind"_a, "a"_a, "eta"_a, "kind"_a, "psi"_a,
          "Hawkes intensity at each event (morie::core::hawkes_intensity).");
    m.def("hawkes_em_pass", &hawkes_em_pass, "t"_a, "lam_old"_a, "eta_old"_a, "kind"_a, "psi_old"_a, "psi_new"_a,
          "Kernel part of the Hawkes EM objective (morie::core::hawkes_em_pass): (Q, P, dQ/dpsi).");
    m.def("hawkes_cdf_sum", &hawkes_cdf_sum, "t"_a, "T"_a, "kind"_a, "psi"_a,
          "sum_j G(T - t_j; psi) and its gradient (morie::core::hawkes_cdf_sum).");
    m.def("ks_pkolmogorov_exact", &morie::core::ks_pkolmogorov_exact, "n"_a, "d"_a,
          "Exact P(D_n < d), one-sample Kolmogorov-Smirnov (Marsaglia, Tsang & Wang 2003).");
    m.def("hawkes_fit_pbfgs", &hawkes_fit_pbfgs, "t"_a, "T"_a, "bkind"_a, "kind"_a, "method"_a, "eps"_a,
          "soe_R"_a, "soe_delta"_a, "lo"_a, "hi"_a, "x0"_a, "maxiter"_a = 2000, "gtol"_a = 1e-6,
          "Hawkes MLE by projected BFGS in C++ (morie::core::hawkes_fit_pbfgs): (theta, nll, iterations).");
    m.def("hawkes_rescaled", &hawkes_rescaled, "t"_a, "T"_a, "bkind"_a, "a"_a, "eta"_a, "kind"_a, "psi"_a,
          "Time-rescaling residuals of a fitted Hawkes process (morie::core::hawkes_rescaled).");
    m.def("hawkes_nll_grad", &hawkes_nll_grad, "t"_a, "T"_a, "bkind"_a, "a"_a, "eta"_a, "kind"_a, "psi"_a,
          "method"_a = 0, "eps"_a = 1e-9, "soe_R"_a = 0.0, "soe_delta"_a = 0.0, "want_grad"_a = true,
          "Hawkes negative log-likelihood and its analytic gradient (morie::core::hawkes_nll_grad).");
    m.def("hawkes_ll_exp_const", &hawkes_ll_exp_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "beta"_a,
          "Hawkes negative log-likelihood -- exponential triggering "
          "kernel, constant baseline. Returns 1e12 for an infeasible "
          "parameter vector.");
    m.def("hawkes_ll_weibull_const", &hawkes_ll_weibull_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "alpha"_a, "lam"_a,
          "Hawkes negative log-likelihood -- Weibull triggering kernel, "
          "constant baseline. Exact O(n^2). Returns 1e12 for an "
          "infeasible parameter vector.");
    m.def("hawkes_ll_weibull_const_trunc", &hawkes_ll_weibull_const_trunc,
          "t"_a, "T"_a, "a0"_a, "eta"_a, "alpha"_a, "lam"_a,
          "Sliding-window O(n*w) form of hawkes_ll_weibull_const -- "
          "bit-identical to the O(n^2) version (the truncated terms "
          "underflow to exactly zero).");
    m.def("hawkes_ll_lomax_const", &hawkes_ll_lomax_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "alpha"_a, "c"_a,
          "Hawkes negative log-likelihood -- Lomax (power-law) triggering "
          "kernel, constant baseline. The caller enforces alpha > 1 and "
          "c > 0.");
    m.def("hawkes_ll_gamma_const", &hawkes_ll_gamma_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "alpha"_a, "beta"_a,
          "Hawkes negative log-likelihood -- gamma triggering kernel, "
          "constant baseline. Returns 1e12 for an infeasible parameter "
          "vector.");
    m.def("hawkes_ll_gamma_const_trunc", &hawkes_ll_gamma_const_trunc,
          "t"_a, "T"_a, "a0"_a, "eta"_a, "alpha"_a, "beta"_a,
          "Sliding-window O(n*w) form of hawkes_ll_gamma_const -- "
          "bit-identical to the O(n^2) version (the truncated terms "
          "underflow to exactly zero).");
    m.def("hawkes_ll_exp_sin", &hawkes_ll_exp_sin, "t"_a, "T"_a, "a0"_a,
          "a1"_a, "a2"_a, "a3"_a, "eta"_a, "beta"_a, "grid"_a, "grid_vals"_a,
          "Hawkes negative log-likelihood -- exponential triggering "
          "kernel, sinusoidal baseline (trapezoid grid). Returns 1e12 "
          "for an infeasible parameter vector.");
    m.def("hawkes_ll_weibull_sin", &hawkes_ll_weibull_sin, "t"_a, "T"_a,
          "a0"_a, "a1"_a, "a2"_a, "a3"_a, "eta"_a, "alpha"_a, "lam"_a,
          "grid"_a, "grid_vals"_a,
          "Hawkes negative log-likelihood -- Weibull triggering kernel, "
          "sinusoidal baseline (trapezoid grid). Returns 1e12 for an "
          "infeasible parameter vector.");
    m.def("hawkes_ll_lomax_sin", &hawkes_ll_lomax_sin, "t"_a, "T"_a,
          "a0"_a, "a1"_a, "a2"_a, "a3"_a, "eta"_a, "alpha"_a, "c"_a,
          "grid"_a, "grid_vals"_a,
          "Hawkes negative log-likelihood -- Lomax (power-law) triggering "
          "kernel, sinusoidal baseline (trapezoid grid). Returns 1e12 for "
          "an infeasible parameter vector.");
    m.def("hawkes_ll_custom", &hawkes_ll_custom, "t"_a, "T"_a, "nu"_a,
          "eta"_a, "g_addr"_a, "G_addr"_a,
          "Hawkes negative log-likelihood with a user-supplied triggering "
          "kernel. g_addr / G_addr are native function-pointer addresses "
          "(from numba @cfunc) for the kernel g(dt) and its integral "
          "G(u) = integral_0^u g.");
    m.def("hawkes_ll_soe", &hawkes_ll_soe, "t"_a, "T"_a, "nu"_a, "eta"_a,
          "w"_a, "beta"_a,
          "Hawkes negative log-likelihood with a sum-of-exponentials "
          "triggering kernel g(u) = sum_m w[m]*exp(-beta[m]*u). O(M*n) "
          "via M parallel exponential recursions.");
    m.def("hawkes_ll_soe_cplx", &hawkes_ll_soe_cplx, "t"_a, "T"_a, "nu"_a,
          "eta"_a, "w"_a, "beta"_a,
          "Complex-pole form of hawkes_ll_soe: w and beta are complex "
          "(complex128). Carries the conjugate-pole pairs from a "
          "matrix-pencil fit; conjugate poles must be passed in pairs "
          "so the likelihood is real. Identical to hawkes_ll_soe for "
          "purely real poles.");
    m.def("hawkes_ll_soe_cplx_ri", &hawkes_ll_soe_cplx_ri, "t"_a, "T"_a,
          "nu"_a, "eta"_a, "w_re"_a, "w_im"_a, "b_re"_a, "b_im"_a,
          "hawkes_ll_soe_cplx with w and beta split into real/imag "
          "double arrays (for callers without complex128 buffers).");
    m.def("hawkes_ll_gamma_hybrid_ri", &hawkes_ll_gamma_hybrid_ri, "t"_a,
          "T"_a, "a0"_a, "eta"_a, "alpha"_a, "beta"_a, "u_split"_a,
          "w_re"_a, "w_im"_a, "b_re"_a, "b_im"_a,
          "hawkes_ll_gamma_hybrid with the SoE modes split into "
          "real/imag double arrays.");
    m.def("hawkes_ll_gamma_hybrid", &hawkes_ll_gamma_hybrid, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "alpha"_a, "beta"_a, "u_split"_a, "w_soe"_a,
          "beta_soe"_a,
          "Hybrid gamma-kernel Hawkes negative log-likelihood: exact "
          "kernel on lags [0, u_split], complex SoE (from "
          "soe_fit_gamma_tail) beyond. O(n*w + M*n).");
}
