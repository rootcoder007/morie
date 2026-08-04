# morie.fn -- function file (rootcoder007/morie)
"""Root-n rate for the index coefficients, and its empirical check.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.2 (page 11), Section 2.4 (page 18) and
Theorem 2.2, equation (2.26) (page 21).

Semiparametric estimators of beta in a single-index model attain the
parametric rate:  (b_n - beta) = O_p(n^{-1/2}), and

    n^{1/2} (btilde_n - betatilde) -> N(0, Sigma).             (2.26)

This is the reason the plug-in estimator of G in Section 2.4 has the
same asymptotics as the infeasible one that knows beta.

Given a sequence of estimates of beta computed on nested subsamples,
this function fits log||b_n - beta|| on log n by least squares and
returns the fitted exponent; -1/2 is the value the theory predicts.
The subsample sizes come from a fixed geometric ladder, so repeated
calls on the same data give the same answer.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["simbrate", "horowitz_rate_beta_estimation"]


def simbrate(errors, sizes, target=-0.5):
    """Fit the empirical convergence exponent of an estimator of beta.

    Parameters
    ----------
    errors : array-like, (k,)
        ||b_n - beta|| at each sample size, in the same order as
        `sizes`.  Strictly positive.
    sizes : array-like, (k,)
        The sample sizes.  At least three, strictly increasing.
    target : float, default -0.5
        The exponent the theory predicts; reported alongside the fit
        as `gap`.

    Returns
    -------
    RichResult
        payload keys: exponent, se, intercept, gap, rsq, target, k,
        n, method.
    """
    e = np.asarray(errors, dtype=float).ravel()
    ns = np.asarray(sizes, dtype=float).ravel()
    if e.size != ns.size or e.size < 3:
        raise ValueError("need at least three matching errors and sizes.")
    if bool(np.any(e <= 0)) or bool(np.any(ns <= 0)):
        raise ValueError("errors and sizes must be strictly positive.")
    k = int(e.size)
    ly = np.log(e)
    lx = np.log(ns)
    mx = float(np.mean(lx))
    my = float(np.mean(ly))
    sxx = float(np.sum((lx - mx) ** 2))
    if sxx <= 0:
        raise ValueError("sizes must not all be equal.")
    slope = float(np.sum((lx - mx) * (ly - my)) / sxx)
    inter = my - slope * mx
    fit = inter + slope * lx
    sse = float(np.sum((ly - fit) ** 2))
    sst = float(np.sum((ly - my) ** 2))
    se = float(np.sqrt(sse / max(k - 2, 1) / sxx))
    return RichResult(
        title="Empirical convergence exponent for beta",
        payload={"exponent": slope, "se": se, "intercept": float(inter),
                 "gap": float(slope - float(target)),
                 "rsq": float(1.0 - sse / sst) if sst > 0 else float("nan"),
                 "target": float(target), "k": k, "n": int(ns[-1]),
                 "method": "Horowitz (2009) eq. (2.26), root-n rate for beta"},
    )


horowitz_rate_beta_estimation = simbrate


def cheatsheet():
    return "hrzrob: empirical check that beta converges at the root-n rate (eq. 2.26)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    ns = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    err = 3.0 * ns ** -0.5           # exactly the theoretical rate
    r = simbrate(err, ns)
    assert abs(r["exponent"] + 0.5) < 1e-10, r["exponent"]
    assert abs(r["gap"]) < 1e-10
    assert r["rsq"] > 0.999999
    print("ok", r["exponent"])
