# morie.fn -- function file (rootcoder007/morie)
""".632 error estimator from apparent and out-of-bag errors."""

from ._richresult import RichResult

__all__ = ["boot_632_estimator"]


def boot_632_estimator(err_app, err_oob, gamma=None):
    """Efron and Tibshirani's (1997) .632 prediction-error estimator,
    `.368 err_app + .632 err_oob`, with the .632+ refinement when the
    no-information rate is supplied.

    One estimator across the library: the computation is
    :func:`morie.fn.eslo63.esl_oob_632`, which carries the full
    derivation -- why the second argument must be the OUT-OF-BAG
    (leave-one-out bootstrap) error and not the naive bootstrap
    error, the book-exact 1-NN counterexample, and the .632+ weight.
    This entry point exists for the bootstrap shelf's naming and adds
    nothing beyond the alias record.

    References
    ----------
    Efron, B. and Tibshirani, R. (1997), "Improvements on
    cross-validation: the .632+ bootstrap method", *JASA*
    92:548-560.
    """
    from .eslo63 import esl_oob_632

    out = esl_oob_632(err_app, err_oob, gamma=gamma)
    payload = dict(out)
    payload["alias_of"] = "morie.fn.eslo63.esl_oob_632"
    return RichResult(payload=payload)


def cheatsheet():
    return "bt632: alias of eslo63 -- one .632 implementation in the library"
