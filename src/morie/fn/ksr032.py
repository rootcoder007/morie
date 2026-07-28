# morie.fn -- function file (rootcoder007/morie)
"""Weak convergence characterisation."""

import numpy as np

from ._richresult import RichResult
from .ksr031 import kosorok_ch2_weak_convergence_tightness

__all__ = ["kosorok_ch2_weak_convergence_iff"]


def kosorok_ch2_weak_convergence_iff(X_n, X=None, T=None, eps=0.1):
    r"""Characterisation of weak convergence in
    :math:`\ell^\infty(T)`:

    :math:`X_n \Rightarrow X` (tight) **iff** (i) all
    finite-dimensional distributions converge, and (ii) :math:`X_n`
    is asymptotically tight.

    Both halves are checked, because either alone is insufficient:
    fidi convergence without tightness is the standard counterexample
    (a spike of shrinking width converges pointwise to 0 but not
    weakly), and this returns them separately rather than collapsing
    to one verdict.

    Fidi convergence is assessed by comparing marginal means and
    variances at the grid points against the reference ``X`` sample;
    tightness reuses :mod:`morie.fn.ksr031`.

    Parameters
    ----------
    X_n : array-like, shape (n_rep, n_points)
        Replicated paths of the approximating process.
    X : array-like, shape (m_rep, n_points), optional
        Reference limit paths. Fidi comparison is skipped when absent.
    T : array-like, optional
        Index grid.
    eps : float, default 0.1
        Oscillation threshold for the tightness half.

    Returns
    -------
    RichResult
        keys: ``fidi_converged``, ``asymptotically_tight``,
        ``weak_convergence``, ``mean_gap``, ``var_gap``, ``mean_tol``,
        ``var_tol`` (Monte-Carlo-scaled, not fixed constants),
        ``tightness_probabilities``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (weak convergence in l^infty(T)).
    """
    A = np.atleast_2d(np.asarray(X_n, dtype=float))
    tight = kosorok_ch2_weak_convergence_tightness(A, rho=T, eps=eps)
    fidi, mgap, vgap, mtol, vtol = None, None, None, None, None
    if X is not None:
        B = np.atleast_2d(np.asarray(X, dtype=float))
        if B.shape[1] != A.shape[1]:
            raise ValueError("X and X_n must share the index grid.")
        mgap = float(np.max(np.abs(A.mean(axis=0) - B.mean(axis=0))))
        vgap = float(np.max(np.abs(A.var(axis=0) - B.var(axis=0))))
        # Tolerances scale with the replication count, not a fixed
        # constant: the max over k grid points of a mean difference has
        # sd ~ sigma sqrt(2/n_rep), and a variance difference
        # ~ sigma^2 sqrt(2/n_rep). A fixed threshold either passes
        # everything at small n_rep or fails identical laws at large k
        # -- at 400 reps over 40 points, two samples from the SAME law
        # already differ by 0.21 in mean and 0.25 in variance.
        na, nb = A.shape[0], B.shape[0]
        k = A.shape[1]
        pooled_sd = float(np.sqrt(0.5 * (A.var() + B.var())))
        z = 4.0  # multiplicity allowance over the k grid points
        mtol = z * pooled_sd * np.sqrt(1.0 / na + 1.0 / nb) * np.sqrt(np.log(max(k, 2)))
        vtol = z * pooled_sd**2 * np.sqrt(2.0 / na + 2.0 / nb) * np.sqrt(
            np.log(max(k, 2))
        )
        fidi = bool(mgap < mtol and vgap < vtol)
    return RichResult(
        payload={"fidi_converged": fidi,
                 "asymptotically_tight": tight["decreasing"],
                 "weak_convergence": None if fidi is None else bool(
                     fidi and tight["decreasing"]),
                 "mean_gap": mgap, "var_gap": vgap,
                 "mean_tol": mtol, "var_tol": vtol,
                 "tightness_probabilities": tight["probabilities"],
                 "method": "fidi convergence AND tightness, reported separately"}
    )


def cheatsheet():
    return "ksr032: fidi alone is NOT enough; tightness is the other half"
