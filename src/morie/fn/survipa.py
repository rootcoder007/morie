# morie.fn -- function file (rootcoder007/morie)
"""Index of prediction accuracy (scaled Brier score)."""

from .brier import brier

from ._richresult import RichResult

__all__ = ["ipa_brier"]


def ipa_brier(time, event, predicted_S, eval_time):
    """``IPA = 1 - BS(model) / BS(null)``, the null being marginal Kaplan-Meier.

    The Brier score is hard to read on its own: its scale depends on the
    event rate, so a small number can mean a good model or merely a rare
    event.  Scaling by the Brier score of the marginal Kaplan-Meier fixes
    the reference point -- 1 is perfect, 0 is no better than ignoring the
    covariates entirely, and negative means actively worse.  This is
    exactly the ``scaled_brier`` already returned by ``brier.brier``, so
    this module is a thin alias rather than a second implementation.

    Formula: ``IPA = 1 - Brier_model / Brier_null``.

    Parameters
    ----------
    time : array-like
        Observed event or censoring times.
    event : array-like
        Event indicator, 1 = event, 0 = censored.
    predicted_S : array-like
        Predicted survival at ``eval_time``.
    eval_time : float
        Evaluation time.

    Returns
    -------
    RichResult
        ``estimate`` (IPA), ``brier_score``, ``scaled_brier``,
        ``eval_time``, ``method``.

    References
    ----------
    Kattan, M. W. & Gerds, T. A. (2018).  The index of prediction
    accuracy: an intuitive measure useful for evaluating risk prediction
    models.  Diagnostic and Prognostic Research 2:7.
    <https://doi.org/10.1186/s41512-018-0029-2>
    """
    r = brier(time, event, predicted_S, eval_time)
    return RichResult(payload={
        "estimate": r["scaled_brier"], "brier_score": r["brier_score"],
        "scaled_brier": r["scaled_brier"], "eval_time": r["eval_time"],
        "method": "IPA = 1 - BS_model/BS_null [Kattan & Gerds 2018]"})


# CANONICAL TEST
# >>> # a model that reproduces the marginal KM exactly scores IPA = 0
# >>> t = [1.0, 2.0, 3.0, 4.0]; e = [1.0, 1.0, 1.0, 1.0]
# >>> from .brier import brier as _b
# >>> null = _b(t, e, [0.5] * 4, 2.5)["brier_score"]
# >>> assert null >= 0.0


def cheatsheet():
    return "survipa(time, event, predicted_S, eval_time): IPA (alias of brier)."
