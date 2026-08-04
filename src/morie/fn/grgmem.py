# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One EM step for a Gaussian mixture."""

from . import _array_core as np

from ._richresult import RichResult
from .grgmll import geron_gmm_log_likelihood

__all__ = ["geron_gmm_em_step"]

_METHOD = "Gaussian mixture EM step (E + M)"


def geron_gmm_em_step(X, pi, means, covars, reg=1e-6):
    r"""E-step responsibilities, then the M-step updates.

    E-step:

    .. math::
        r_{ik} = \frac{\pi_k \mathcal N(x_i \mid \mu_k, \Sigma_k)}
                      {\sum_j \pi_j \mathcal N(x_i \mid \mu_j, \Sigma_j)}

    M-step, with :math:`N_k = \sum_i r_{ik}`:

    .. math::
        \pi_k = \frac{N_k}{m},\qquad
        \mu_k = \frac{1}{N_k}\sum_i r_{ik} x_i,\qquad
        \Sigma_k = \frac{1}{N_k}\sum_i r_{ik}
        (x_i-\mu_k)(x_i-\mu_k)^{\mathsf T}

    The covariance is computed against the *new* mean, not the old one
    -- that is what makes the step a genuine maximisation of the
    expected complete-data log-likelihood.

    EM never decreases the log-likelihood.  Both the before and after
    values are computed by
    :func:`morie.fn.grgmll.geron_gmm_log_likelihood` and the guarantee
    is checked: a decrease beyond floating-point slack raises.

    Parameters
    ----------
    X : array-like, shape (m, d)
    pi : array-like, shape (K,)
    means : array-like, shape (K, d)
    covars : array-like, shape (K, d, d)
    reg : float, optional
        Ridge added to each new covariance diagonal, default ``1e-6``.
        Without it a component that collapses onto a single point
        produces a singular covariance and an infinite likelihood.

    Returns
    -------
    RichResult
        Payload keys ``responsibilities``, ``pi_new``, ``means_new``,
        ``covars_new``, ``Nk``, ``log_likelihood_before``,
        ``log_likelihood_after``, ``improvement``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 8, Expectation-Maximization section.

    Examples
    --------
    Two well-separated points, two components already sitting on them:
    the responsibilities are essentially hard, and the update is a fixed
    point.

    >>> X = [[0.0], [10.0]]
    >>> r = geron_gmm_em_step(X, [0.5, 0.5], [[0.0], [10.0]], [[[1.0]], [[1.0]]])
    >>> [round(v, 6) for v in r["responsibilities"][0]]
    [1.0, 0.0]
    >>> [round(mu[0], 6) for mu in r["means_new"]]
    [0.0, 10.0]
    >>> r["improvement"] >= 0
    True

    Start one component in the wrong place and the step drags it toward
    the data it is responsible for, raising the likelihood:

    >>> r2 = geron_gmm_em_step(X, [0.5, 0.5], [[3.0], [10.0]], [[[1.0]], [[1.0]]])
    >>> round(r2["means_new"][0][0], 6)
    0.0
    >>> r2["improvement"] > 0
    True
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    before = geron_gmm_log_likelihood(A, pi, means, covars)   # validates everything
    logp = np.asarray(before["component_log_densities"], dtype=float)
    m, d = A.shape
    K = logp.shape[1]
    reg = float(reg)
    if reg < 0:
        raise ValueError(f"reg must be non-negative, got {reg}.")

    mx = logp.max(axis=1, keepdims=True)
    E = np.exp(logp - mx)
    R = E / E.sum(axis=1, keepdims=True)

    Nk = R.sum(axis=0)
    dead = np.flatnonzero(Nk <= 0)
    if dead.size:
        raise ValueError(
            f"components {dead.tolist()} have zero total responsibility; their "
            f"mean and covariance updates would divide by zero. Drop them or "
            f"re-seed the mixture."
        )
    pi_new = Nk / m
    mu_new = (R.T @ A) / Nk[:, None]
    S_new = np.empty((K, d, d))
    for k in range(K):
        diff = A - mu_new[k]
        S_new[k] = (diff * R[:, k][:, None]).T @ diff / Nk[k]
        S_new[k] += reg * np.eye(d)

    after = geron_gmm_log_likelihood(A, pi_new, mu_new, S_new)
    improvement = after["log_likelihood"] - before["log_likelihood"]
    if improvement < -1e-6 * max(1.0, abs(before["log_likelihood"])):
        raise ValueError(
            f"the EM step decreased the log-likelihood by {-improvement}; EM is "
            f"monotone, so this indicates a numerical problem (try a larger reg)."
        )

    return RichResult(
        title="GMM EM step",
        summary_lines=[("log L before", before["log_likelihood"]),
                       ("log L after", after["log_likelihood"]),
                       ("Improvement", improvement)],
        payload={
            "responsibilities": R.tolist(),
            "pi_new": pi_new.tolist(),
            "means_new": mu_new.tolist(),
            "covars_new": S_new.tolist(),
            "Nk": Nk.tolist(),
            "log_likelihood_before": before["log_likelihood"],
            "log_likelihood_after": after["log_likelihood"],
            "improvement": float(improvement),
            "estimate": mu_new.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgmem: E-step responsibilities then weighted pi/mu/Sigma updates; log L checked monotone"


# compact alias per ledger/NAMING.md
gerongmmemstep = geron_gmm_em_step
