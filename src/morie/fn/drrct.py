# morie.fn -- function file (rootcoder007/morie)
"""Experimental Selection Correction: an RCT-validated control function.

Source opened: Athey, S., Chetty, R. and Imbens, G. W. (2020/2025).
The experimental selection correction estimator: using experiments to
remove biases in observational estimates.  arXiv:2006.09676, page 10,
equations (2.3) and (2.4).  A secondary outcome Y^S that the experiment
does identify supplies a control function: the coefficients tau^S and
gamma^S are estimated ON THE EXPERIMENTAL SAMPLE, where W is randomised,

    alpha_hat^S_i = Y^S_i - W_i tau_hat^S - X_i' gamma_hat^S      (2.3)

is then formed for the OBSERVATIONAL units, and the primary outcome is
regressed there on treatment, covariates and that residual,

    Y^P_i = W_i tau + X_i' gamma + delta alpha^S_i + eps^P_i.     (2.4)

The coefficient on W is the selection-corrected effect.  The identifying
content is latent unconfoundedness: W is exogenous once alpha^S enters
the conditioning set, though not given X alone.

Which sample the first stage is fitted on is the whole estimator.  Fit
both stages on one sample and alpha^S is by construction orthogonal to
(1, W, X), delta cannot move the coefficient on W, and the correction is
identically zero -- that exact degeneracy is what this module returns
when no group indicator is supplied, and it is the algebraic anchor the
implementation is checked against.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_rct_assisted_did"]


def dr_rct_assisted_did(y_obs, y_rct, D, X=None, G=None):
    """Selection-corrected treatment effect on the primary outcome.

    Parameters
    ----------
    y_obs : array-like
        Primary outcome Y^P, the one whose effect is wanted.
    y_rct : array-like
        Secondary outcome Y^S, observed in both samples.
    D : array-like
        Binary treatment W.
    X : 2-D array-like, optional
        Pre-treatment covariates.
    G : array-like, optional
        1 for the experimental rows (first stage), 0 for the
        observational rows (second stage).  ``None`` puts every row in
        both, which forces the correction to zero exactly.

    Returns
    -------
    result : dict
        Keys: estimate (tau_esc), tau_esc, tau_naive, correction, delta,
        tau_secondary, alpha_sd, n_exp, n_obs, n.

    References
    ----------
    Athey, Chetty & Imbens, arXiv:2006.09676, eqs. (2.3)-(2.4), p. 10.
    """
    yp = k.vec(y_obs)
    ys = k.vec(y_rct)
    dv = k.vec(D)
    n = len(yp)
    if n == 0:
        raise ValueError("empty input: y_obs has no observations")
    if len(ys) != n or len(dv) != n:
        raise ValueError("y_obs, y_rct and D must have the same length")
    Z = k.design(X, n)
    W = [[1.0, dv[i]] + list(Z[i][1:]) for i in range(n)]
    if G is None:
        ie = list(range(n))
        io = list(range(n))
    else:
        gv = k.vec(G)
        if len(gv) != n:
            raise ValueError("G must have the same length as y_obs")
        ie = [i for i in range(n) if gv[i] >= 0.5]
        io = [i for i in range(n) if gv[i] < 0.5]
        if not ie or not io:
            raise ValueError("G must mark both an experimental and an "
                             "observational subsample")
    for idx, nm in ((ie, "experimental"), (io, "observational")):
        s = sum(dv[i] for i in idx)
        if s <= 0.0 or s >= float(len(idx)):
            raise ValueError("the %s subsample must contain both arms" % nm)
    bs = k.lstsq([W[i] for i in ie], [ys[i] for i in ie])
    alpha = []
    for i in io:
        f = 0.0
        for j in range(len(bs)):
            f += bs[j] * W[i][j]
        alpha.append(ys[i] - f)
    bn = k.lstsq([W[i] for i in io], [yp[i] for i in io])
    Wa = [W[io[j]] + [alpha[j]] for j in range(len(io))]
    be = k.lstsq(Wa, [yp[i] for i in io])
    return RichResult(
        title="Experimental selection correction",
        summary_lines=[("delta", be[len(be) - 1])],
        payload={
            "estimate": be[1],
            "tau_esc": be[1],
            "tau_naive": bn[1],
            "correction": be[1] - bn[1],
            "delta": be[len(be) - 1],
            "tau_secondary": bs[1],
            "alpha_sd": k.sd(alpha) if len(alpha) > 1 else 0.0,
            "n_exp": float(len(ie)),
            "n_obs": float(len(io)),
            "n": n,
            "method": "DR-DiD with RCT side data",
        },
    )


def cheatsheet():
    return "drrct: DR-DiD with RCT side data"
