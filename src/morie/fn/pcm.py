"""Masters partial credit model probabilities (Masters 1982)."""

import math

from ._richresult import RichResult

__all__ = ["pcm", "partial_credit_model"]


def pcm(theta, steps, a=1.0, D=1.0):
    """
    Category response probabilities of the (generalized) partial
    credit model.

    plink Eq. 4 (Weeks 2010; Muraki 1992 GPCM), with the first step
    parameter dropped (b_1 is arbitrary and cancels):

        P(X = k | theta) = exp(sum_{v=1}^{k} D a (theta - b_v))
                           / sum_{h=1}^{K} exp(sum_{v=1}^{h} D a (theta - b_v)),

    where the sum over an empty index set (k = 1, the lowest
    category) is zero.  With the slope constrained to a = 1 this is
    the partial credit model of Masters (1982), as stated by Weeks
    (2010, Sec. 2.1): "the slope parameters for the GPCM can be
    constrained to be equal across all items.  When they equal one,
    this is known as the partial credit model (PCM; Masters 1982)."
    With a single step parameter the model reduces to the 2PL/Rasch
    dichotomous model.

    Sources
    -------
    Masters, G. N. (1982). A Rasch model for partial credit scoring.
    *Psychometrika*, 47, 149-174.
    Muraki, E. (1992). A generalized partial credit model:
    Application of an EM algorithm. *Applied Psychological
    Measurement*, 16, 159-176.
    Weeks, J. P. (2010). plink: An R package for linking mixed-format
    tests using IRT-based methods. *Journal of Statistical Software*,
    35(12), Eq. 4 and Sec. 2.1 (local copy
    fetched-wave3/weeks-2010-plink-JSS35.pdf).

    Parameters
    ----------
    theta : float
        Ability value.
    steps : sequence of float
        Step (intersection) parameters b_2, ..., b_K; K - 1 values
        for K response categories (b_1 excluded, plink convention).
    a : float
        Common discrimination (1.0 = Masters PCM).
    D : float
        Scaling constant (1.0 or 1.7).

    Returns
    -------
    RichResult
        Keys: probabilities (length K), expected_score (0-based),
        n_categories.
    """
    th = float(theta)
    b = [float(v) for v in steps]
    a = float(a)
    D = float(D)
    if not b:
        raise ValueError("need at least one step parameter")
    # cumulative numerator exponents; category 1 has empty sum = 0
    exps = [0.0]
    run = 0.0
    for bv in b:
        run += D * a * (th - bv)
        exps.append(run)
    mx = max(exps)
    ex = [math.exp(e - mx) for e in exps]
    den = sum(ex)
    p = [e / den for e in ex]
    esc = sum(k * pk for k, pk in enumerate(p))
    return RichResult(payload={
        "probabilities": p,
        "expected_score": esc,
        "n_categories": len(p),
        "theta": th, "a": a, "D": D,
        "method": "partial credit model (Masters 1982; plink Eq. 4)",
    })


# long descriptive alias (stub-era name)
partial_credit_model = pcm


def cheatsheet():
    return "pcm: P(k) = exp(sum_v Da(theta-b_v)) / sum_h exp(...), a=1 Masters"
