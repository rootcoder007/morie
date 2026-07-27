# morie.fn -- function file (rootcoder007/morie)
"""Double ML mediation, Neyman-orthogonal."""

from ._richresult import RichResult
from .medML import ml_mediation_dml

__all__ = ["dml_mediation_orthogonal"]


def dml_mediation_orthogonal(x, m, y, c, n_folds=5, seed=0):
    """Front-end to :func:`morie.fn.medML.ml_mediation_dml`.

    Same cross-fitted partialling-out estimator (Chernozhukov et al.
    2018, *Econometrics Journal* 21(1), C1-C68); kept as a separate
    entry point for the double-ML namespace.
    """
    out = ml_mediation_dml(x, m, y, c, n_folds=n_folds, seed=seed)
    payload = dict(out)
    payload["method"] = "Double ML mediation (Neyman-orthogonal, cross-fitted)"
    return RichResult(payload=payload)


def cheatsheet():
    return "dmlMed: front-end to medML cross-fitted mediation"
