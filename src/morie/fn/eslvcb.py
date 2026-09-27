"""Vapnik-Chervonenkis bounds on the test error (ESL sec 7.9)."""

import math

from ._richresult import RichResult

__all__ = ["esl_vc_bound"]


def esl_vc_bound(err, h, N, eta=0.05, task="classification", a1=None, a2=None, c=1.0):
    r"""Upper bound on :math:`\mathrm{Err}_T` holding with probability :math:`1-\eta` over the whole class.

    ESL eq 7.46 (after Cherkassky & Mulier 2007): with
    :math:`\epsilon = a_1\{h[\log(a_2N/h) + 1] - \log(\eta/4)\}/N`,

    .. math::  \mathrm{Err}_T \le \mathrm{err} + \frac{\epsilon}{2}\Big(1 + \sqrt{1 + 4\,\mathrm{err}/\epsilon}\Big)
               \ \text{(classification)}, \qquad
               \mathrm{Err}_T \le \frac{\mathrm{err}}{(1 - c\sqrt\epsilon)_+}\ \text{(regression)}.

    Defaults follow the text: :math:`a_1 = 4, a_2 = 2` (worst case) for
    classification, :math:`a_1 = a_2 = 1` and c = 1 for regression. For
    regression the practical bound of eq 7.47,
    :math:`\mathrm{err}(1 - \sqrt{\rho - \rho\log\rho + \log N/(2N)})_+^{-1}` with
    :math:`\rho = h/N`, is returned alongside. An infinite bound means the
    denominator is not positive.

    Parameters
    ----------
    err : float
        Training error.
    h : float
        VC dimension, > 0.
    N : int
        Sample size.
    eta : float
        One minus the confidence of the bound.
    task : {"classification", "regression"}
    a1, a2, c : float, optional

    Returns
    -------
    RichResult
        ``bound``, ``epsilon``, and for regression ``practical_bound``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 7.9; Cherkassky, V. & Mulier,
    F. (2007). Learning from Data (2nd ed.), pp. 116-118.
    """
    if task not in ("classification", "regression"):
        raise ValueError("`task` must be 'classification' or 'regression'")
    if h <= 0 or N <= 0 or not 0 < eta < 1 or err < 0:
        raise ValueError("need h > 0, N > 0, 0 < eta < 1 and err >= 0")
    if a1 is None:
        a1 = 4.0 if task == "classification" else 1.0
    if a2 is None:
        a2 = 2.0 if task == "classification" else 1.0
    eps = a1 * (h * (math.log(a2 * N / h) + 1) - math.log(eta / 4)) / N
    out = {"epsilon": eps}
    if task == "classification":
        out["bound"] = err + eps / 2 * (1 + math.sqrt(1 + 4 * err / eps))
    else:
        den = 1 - c * math.sqrt(eps)
        out["bound"] = err / den if den > 0 else math.inf
        rho = h / N
        den2 = 1 - math.sqrt(rho - rho * math.log(rho) + math.log(N) / (2 * N))
        out["practical_bound"] = err / den2 if den2 > 0 else math.inf
    return RichResult(title=f"VC bound ({task})", summary_lines=[("bound", out["bound"])], payload=out)


def cheatsheet():
    return "eslvcb: eps = a1 (h (log(a2 N / h) + 1) - log(eta/4)) / N; ESL 7.46-7.47"
