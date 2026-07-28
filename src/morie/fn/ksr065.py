# morie.fn -- function file (rootcoder007/morie)
"""Efficient influence function via the information operator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch3_efficient_influence_general"]


def kosorok_ch3_efficient_influence_general(A, psi_tilde, chi_tilde=None, eta=None):
    r"""Efficient influence function through the information operator
    (Kosorok Ch. 3):

    .. math:: A^*_\eta\, \tilde\psi_{P_\eta}
              = \tilde\chi_\eta \quad \text{in } \mathrm{lin}\,
              H_\eta.

    The efficient influence function is obtained by solving this
    operator equation, i.e. by projecting onto the orthocomplement of
    the nuisance tangent space. Solved here as a least-squares problem
    with the residual reported: an inconsistent system means the
    parameter is NOT pathwise differentiable in the model, which is a
    real finding rather than a numerical nuisance to be suppressed.

    Parameters
    ----------
    A : array-like, shape (m, k)
        Matrix representation of the adjoint information operator on
        the chosen basis.
    psi_tilde : array-like, shape (m,)
        The right-hand side (the parameter's gradient).
    chi_tilde : array-like, optional
        A candidate solution to check instead of solving.
    eta : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``chi``, ``residual_norm``, ``consistent`` (bool),
        ``rank``, ``efficient_variance`` (||chi||^2), ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 3 (efficient influence functions and information operators).
    """
    Amat = np.atleast_2d(np.asarray(A, dtype=float))
    rhs = np.asarray(psi_tilde, dtype=float).ravel()
    if Amat.shape[0] != rhs.size:
        raise ValueError(f"A has {Amat.shape[0]} rows but psi_tilde has {rhs.size}.")
    if chi_tilde is not None:
        chi = np.asarray(chi_tilde, dtype=float).ravel()
        if chi.size != Amat.shape[1]:
            raise ValueError("chi_tilde must match the columns of A.")
    else:
        chi, *_ = np.linalg.lstsq(Amat, rhs, rcond=None)
    resid = float(np.linalg.norm(Amat @ chi - rhs))
    scale = max(1.0, float(np.linalg.norm(rhs)))
    return RichResult(
        payload={"chi": chi, "residual_norm": resid,
                 "consistent": bool(resid < 1e-8 * scale),
                 "rank": int(np.linalg.matrix_rank(Amat)),
                 "efficient_variance": float(chi @ chi),
                 "method": "Solve A* psi = chi; a large residual means NOT pathwise diff."}
    )


def cheatsheet():
    return "ksr065: inconsistent system = parameter not pathwise differentiable"
