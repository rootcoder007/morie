// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- Hawkes-process likelihood kernels.
//
// Phase 2 of the v0.9.1 backend port. hawkes_ll_exp_const is the
// canonical self-exciting point-process negative log-likelihood with
// an exponential triggering kernel and a constant baseline.
//
// The O(n) recursion  A_i = exp(-beta*dt_i) * (A_{i-1} + beta)  is
// genuinely sequential -- it cannot be numpy-vectorised -- which is
// exactly where a compiled loop pays off over the Python equivalent.
//
// Ported from morie/tps_hawkes_jit.py::_ll_exp_const; the arithmetic
// is identical operation for operation.

#include "hawkes.h"

#include <cmath>
#include <cstddef>

#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;

// Sentinel returned for an infeasible parameter vector -- the
// optimiser reads it as "very bad", same as the Python version.
const double kBig = 1e12;

double hawkes_ll_exp_const(Vec t, double T, double a0, double eta,
                           double beta) {
    // Box constraints (mirror _neg_loglik_general): beta is bounded
    // away from infinity to avoid the well-known Hawkes spike-train
    // degeneracy where beta -> inf drives the log-likelihood to +inf.
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < beta && beta < 30.0)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const std::size_t n = t.shape(0);
    const double nu = std::exp(a0);
    if (!std::isfinite(nu) || nu <= 0.0) return kBig;

    // Sum of log-intensities, via the O(n) recursive update
    //   A_1 = 0;  A_i = exp(-beta*dt_i) * (A_{i-1} + beta)
    double log_sum = 0.0;
    double A = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double lam_i;
        if (i == 0) {
            lam_i = nu;
        } else {
            const double dt = t(i) - t(i - 1);
            A = std::exp(-beta * dt) * (A + beta);
            lam_i = nu + eta * A;
        }
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    // Compensator integral: int_0^T nu dt + eta * sum(1 - exp(-beta(T-t_i)))
    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        integral += eta * (1.0 - std::exp(-beta * (T - t(i))));
    }
    if (!std::isfinite(log_sum) || !std::isfinite(integral)) return kBig;

    return -(log_sum - integral);
}

}  // namespace

void register_hawkes(nb::module_ &m) {
    m.def("hawkes_ll_exp_const", &hawkes_ll_exp_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "beta"_a,
          "Hawkes negative log-likelihood -- exponential triggering "
          "kernel, constant baseline. Returns 1e12 for an infeasible "
          "parameter vector.");
}
