# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap functional delta method."""

import numpy as np

from ._kosorok import hadamard_derivative
from ._richresult import RichResult

__all__ = ["kosorok_ch2_functional_delta_bootstrap"]


def kosorok_ch2_functional_delta_bootstrap(phi, X_n, X_hat_n, r_n, c=1.0, mu=None):
    r"""Bootstrap version of the functional delta method:

    .. math:: r_n c\,\big(\phi(\hat X_n) - \phi(X_n)\big)
              \Rightarrow_W \phi'_\mu(X) \quad
              \text{in probability, under the bootstrap weights } W.

    Note the centring: the bootstrap statistic is centred at
    :math:`\phi(X_n)`, the SAMPLE value, not at
    :math:`\phi(\mu)`. Centring at the truth instead is the classic
    bootstrap error -- it adds the original sampling error back in and
    inflates every interval.

    The constant c rescales for bootstrap schemes whose weights have
    variance other than 1 (c = 1 for the nonparametric bootstrap).

    Parameters
    ----------
    phi : callable
        The functional.
    X_n : array-like or float
        The original statistic.
    X_hat_n : sequence of array-like
        Bootstrap replicates of the statistic.
    r_n : float
        Scaling rate.
    c : float, default 1.0
        Scheme constant.
    mu : array-like, optional
        The truth, used only to report the (wrong) truth-centred
        version for comparison.

    Returns
    -------
    RichResult
        keys: ``scaled_replicates``, ``mean``, ``sd``,
        ``truth_centred_sd`` (if mu given -- larger, showing the
        error), ``derivative``, ``n_boot``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (the bootstrap delta method).
    """
    r_n = float(r_n)
    if r_n <= 0:
        raise ValueError(f"r_n must be positive, got {r_n}.")
    c = float(c)
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}.")
    Xn = np.asarray(X_n, dtype=float)
    reps = [np.asarray(x, dtype=float) for x in X_hat_n]
    if len(reps) < 2:
        raise ValueError("need at least 2 bootstrap replicates.")
    base = np.asarray(phi(Xn), dtype=float)
    scaled = np.array([r_n * c * (np.asarray(phi(r), dtype=float) - base)
                       for r in reps])
    der, _drift, _ok = hadamard_derivative(phi, Xn, np.ones_like(Xn))
    payload = {"scaled_replicates": scaled, "mean": float(np.mean(scaled)),
               "sd": float(np.std(scaled, ddof=1)), "derivative": der,
               "n_boot": len(reps),
               "method": "Centred at phi(X_n), the SAMPLE value, not at phi(mu)"}
    if mu is not None:
        truth = np.asarray(phi(np.asarray(mu, dtype=float)), dtype=float)
        wrong = np.array([r_n * c * (np.asarray(phi(r), dtype=float) - truth)
                          for r in reps])
        payload["truth_centred_sd"] = float(np.std(wrong, ddof=1))
        payload["truth_centred_mean"] = float(np.mean(wrong))
    return RichResult(payload=payload)


def cheatsheet():
    return "ksr045: centre at phi(X_n); centring at phi(mu) inflates every interval"
