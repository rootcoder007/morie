# morie.fn -- function file (rootcoder007/morie)
"""Frechet differentiability."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_frechet_differentiability"]


def kosorok_ch2_frechet_differentiability(phi, theta, h_n, derivative=None):
    r"""Frechet differentiability check:

    .. math:: \frac{\|\phi(\theta + h_n) - \phi(\theta)
              - \phi'_\theta(h_n)\|}{\|h_n\|} \to 0.

    Frechet is STRONGER than Hadamard: the ratio must vanish for every
    sequence :math:`h_n \to 0`, not merely for those converging in a
    direction. The quantile map is the standard example -- Hadamard
    differentiable (van der Vaart 1998, Lemma 21.3, with derivative
    :math:`\alpha \mapsto -\alpha(F^{-1})/f(F^{-1})`) but not
    Frechet. Reeds (1976) introduced Hadamard into this literature for
    exactly that reason, and it is why the delta method in this book
    is built on Hadamard rather than Frechet.

    Evaluated along the supplied sequence; a small ratio here is
    evidence, not proof, and the returned key is named
    ``ratio_shrinking`` accordingly.

    Parameters
    ----------
    phi : callable
        The functional.
    theta : array-like or float
        Base point.
    h_n : sequence of array-like
        A sequence of perturbations with decreasing norm.
    derivative : callable, optional
        The candidate derivative; a numerical one is used if omitted.

    Returns
    -------
    RichResult
        keys: ``ratios``, ``norms``, ``ratio_shrinking``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (differentiability of
    functionals).

    van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge
    University Press. Lemma 21.3 (Hadamard differentiability of the
    quantile map).

    Reeds, J. A. (1976). *On the Definition of von Mises Functionals*.
    PhD thesis, Harvard University.
    """
    th = np.asarray(theta, dtype=float)
    base = np.asarray(phi(th), dtype=float)
    seq = [np.asarray(h, dtype=float) for h in h_n]
    if len(seq) < 2:
        raise ValueError("h_n must contain at least 2 perturbations.")
    norms = np.array([float(np.linalg.norm(h)) for h in seq])
    if np.any(norms <= 0):
        raise ValueError("perturbations must be non-zero.")
    if derivative is None:
        # Frechet demands ONE linear map valid in every direction, so
        # the fallback builds the Jacobian once at theta by central
        # differences and applies it linearly. Recomputing a
        # directional derivative per h would make the check vacuous --
        # it would return the Hadamard derivative and report even a
        # kinked map like |.| as Frechet differentiable.
        t = 1e-6
        flat = np.atleast_1d(th).astype(float)
        basis = np.eye(flat.size)
        cols = []
        for e in basis:
            plus = np.asarray(phi((flat + t * e).reshape(np.shape(th))), dtype=float)
            minus = np.asarray(phi((flat - t * e).reshape(np.shape(th))), dtype=float)
            cols.append((plus - minus) / (2 * t))
        J = np.stack(cols, axis=-1)

        def derivative(h, J=J):
            return J @ np.atleast_1d(np.asarray(h, dtype=float)).ravel()

    ratios = []
    for h, nrm in zip(seq, norms):
        num = np.asarray(phi(th + h), dtype=float) - base - np.asarray(
            derivative(h), dtype=float
        )
        ratios.append(float(np.linalg.norm(num) / nrm))
    ratios = np.array(ratios)
    order = np.argsort(-norms)  # largest perturbation first
    r_sorted = ratios[order]
    return RichResult(
        payload={"ratios": ratios, "norms": norms,
                 "ratio_shrinking": bool(r_sorted[-1] <= r_sorted[0] + 1e-12),
                 "method": "||phi(th+h) - phi(th) - phi'(h)|| / ||h|| along h_n"}
    )


def cheatsheet():
    return "ksr050: Frechet is STRONGER than Hadamard; quantiles are Hadamard only"
