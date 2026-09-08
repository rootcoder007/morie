# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Residualized CATE estimator.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition) -- NOT LOCATED in the corpus copy
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["rlearn", "r_learner"]

_METHOD = "Residualized CATE estimator"


def rlearn(y, t, m, e, x=None):
    """Residualized CATE estimator.

    Residualized (Robinson-style) CATE estimator.

    NOT LOCATED IN THE EXTRACTED TEXT of the corpus copy of Molak,
    which covers the S-, T-, X- and DR-learners but has no R-learner
    section.  The estimator is therefore taken from the primary source
    and quoted from it:

    Robinson decomposition, eq. (1):
    "Y_i - m*(X_i) = {W_i - e*(X_i)} tau*(X_i) + eps_i"

    R-learner objective, eq. (4):
    "tau_hat(.) = argmin_tau [ L_hat_n{tau(.)} + Lambda_n{tau(.)} ]"
    with
    "L_hat_n{tau(.)} = (1/n) sum_i [ {Y_i - m_hat^(-q(i))(X_i)}
    - {W_i - e_hat^(-q(i))(X_i)} tau(X_i) ]^2"

    -- Nie, X. and Wager, S. (2021), "Quasi-Oracle Estimation of
    Heterogeneous Treatment Effects", Biometrika 108(2):299-319
    (arXiv:1712.04912), where "e*(x) = pr(W = 1 | X = x)" and
    "m*(x) = E(Y | X = x)".

    This routine computes eq. (4) with the regularizer Lambda_n set to
    zero and the CROSS-FITTED nuisance predictions m_hat^(-q(i)) and
    e_hat^(-q(i)) supplied BY THE CALLER, so the fold assignment q(.)
    -- the only random ingredient in the paper's construction -- lives
    outside this function and both language arms see identical numbers.
    ``x`` gives basis columns for a linear tau(X); omit it for a
    constant treatment effect.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._molak.rlearn``.
    t : as documented for the shelf core
        See ``morie.fn._molak.rlearn``.
    m : as documented for the shelf core
        See ``morie.fn._molak.rlearn``.
    e : as documented for the shelf core
        See ``morie.fn._molak.rlearn``.
    x : as documented for the shelf core
        See ``morie.fn._molak.rlearn``.

    Returns
    -------
    result : RichResult
        Payload keys: ate, loss, n.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition) -- NOT LOCATED in the corpus copy
    """
    res = _core.rlearn(y=y, t=t, m=m, e=e, x=x)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ate", res["ate"]), ("loss", res["loss"]), ("n", res["n"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
r_learner = rlearn


def cheatsheet():
    return "rlearn: Residualized CATE estimator"
