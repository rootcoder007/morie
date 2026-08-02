# morie.fn -- function file (rootcoder007/morie)
"""Algorithm-agnostic variable importance from average predictiveness."""

from . import _array_core as np

from ._richresult import RichResult
from ._vimp import MEASURES, vim

__all__ = ["tmle_average_predictiveness"]


def tmle_average_predictiveness(y, D, X, f=None, loss="r_squared",
                                n_folds=5, sample_split=True, alpha=0.05,
                                seed=0, **learner):
    r"""Importance of a variable group, measured as lost predictiveness.

    .. math::
       \psi_s = V(f_0, P_0) - V(f_{0,s}, P_0),

    the drop in population predictiveness when the best predictor is
    forbidden from using the columns in ``D``. It is algorithm-agnostic:
    the target is a property of the DISTRIBUTION, defined through the
    oracle prediction functions, not of whichever learner happened to be
    fitted -- so two analysts using different learners are estimating
    the same number.

    There is no fluctuation step, and its absence is the point.
    Plug-in estimators of smooth functionals normally carry first-order
    bias that has to be removed by targeting or a one-step correction.
    Here :math:`f_0` MAXIMISES :math:`V(\cdot, P_0)`, so the derivative
    of the predictiveness in the direction of the prediction function
    vanishes at the optimum and the first-order term is already zero.
    The difference of the two plug-ins is efficient as it stands.

    What does need care is inference, and two separate devices are
    involved. They are often conflated; they solve different problems.

    CROSS-FITTING trains on one fold and evaluates on another, which
    removes the Donsker condition that would otherwise limit how
    flexible the learner may be.

    SAMPLE-SPLITTING estimates :math:`V(f_0)` and :math:`V(f_{0,s})` on
    DISJOINT halves. Under the null :math:`\psi_s = 0` the two
    influence functions coincide, so their difference is identically
    zero and the estimator has no non-degenerate limit -- a Wald
    interval built on it has the wrong coverage at every sample size,
    and no amount of data fixes it. Splitting restores a non-degenerate
    limit, at the cost of power. ``sample_split=False`` buys that power
    back and is only safe when the importance is known to be non-null.

    Four predictiveness measures are available, each with its exact
    mean-zero gradient: ``'r_squared'`` for continuous outcomes, and
    ``'accuracy'``, ``'auc'`` and ``'deviance'`` for binary ones.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome. Binary for every measure except ``'r_squared'``.
    D : int or sequence of int
        Column indices of ``X`` forming the group whose importance is
        assessed. A group, not necessarily a single variable -- the
        importance of correlated columns is only interpretable jointly.
    X : array-like, shape (n, p)
        Covariates.
    f : callable, optional
        Learner-as-function: given a covariate block, returns
        predictions. Supply one to use a specific estimator. Omitted, a
        native gradient-boosted ensemble is fitted.
    loss : {'r_squared', 'accuracy', 'auc', 'deviance'}
        The predictiveness measure.
    n_folds : int
        Cross-fitting folds.
    sample_split : bool
        Split the sample between the two predictiveness estimates. Keep
        it on for testing; see above.
    alpha : float
        Two-sided level for ``ci``.
    seed : int
        Controls the fold and split assignment.
    **learner
        Passed to the default learner (``n_estimators``, ``max_depth``,
        ``learning_rate``).

    Returns
    -------
    RichResult
        ``estimate`` (:math:`\psi_s`), ``se``, ``ci``,
        ``ci_one_sided``, ``p_value``, ``v_full``, ``v_reduced``,
        ``measure``, ``sample_split``, ``null_inference_valid``.

    References
    ----------
    Williamson, Gilbert, Simon and Carone (2023), "A general framework
    for inference on algorithm-agnostic variable importance", *Journal
    of the American Statistical Association* 118:1645-1658.
    Preprint arXiv:2004.03683; the predictiveness measures of their
    section 2.3 and the gradients of their Appendix A are implemented
    here, with the sample-splitting scheme of their Algorithm 3.
    Williamson, Gilbert, Carone and Simon (2021), "Nonparametric
    variable importance assessment using machine learning techniques",
    *Biometrics* 77:9-22 -- the R-squared special case.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 3))
    >>> y = 2 * X[:, 0] + rng.normal(scale=0.5, size=600)
    >>> out = tmle_average_predictiveness(y, 0, X)
    >>> bool(out["estimate"] > 0.4)
    True
    """
    if loss not in MEASURES:
        raise ValueError(
            "loss must be one of %s, got %r." % (MEASURES, loss)
        )
    res = vim(
        y, X, D, measure=loss, f=f, n_folds=n_folds,
        sample_split=sample_split, alpha=alpha, seed=seed, **learner
    )
    return RichResult(
        payload={
            "estimate": res["estimate"],
            "se": res["se"],
            "ci": res["ci"],
            "ci_lower": res["ci"][0],
            "ci_upper": res["ci"][1],
            "ci_one_sided": res["ci_one_sided"],
            "ci_note": (
                "the null value 0 sits on the boundary of the parameter "
                "space, so the one-sided interval is the one to test with; "
                "the two-sided interval loses power for that purpose"
            ),
            "test_statistic": res["test_statistic"],
            "p_value": res["p_value"],
            "v_full": res["v_full"],
            "v_reduced": res["v_reduced"],
            "predictiveness_note": (
                "importance is the DROP from v_full to v_reduced; a small "
                "drop from a poor v_full says the group adds nothing to a "
                "model that already explains nothing"
            ),
            "eta_full": res["eta_full"],
            "eta_reduced": res["eta_reduced"],
            "measure": res["measure"],
            "variables": np.asarray(res["s"]),
            "n_folds": res["n_folds"],
            "sample_split": res["sample_split"],
            "null_inference_valid": res["sample_split"],
            "null_note": (
                None if res["sample_split"] else
                "without sample-splitting the estimator is degenerate under "
                "psi = 0, so this interval does NOT have its nominal "
                "coverage when the group is truly unimportant"
            ),
            "n_full": res["n_full"],
            "n_reduced": res["n_reduced"],
            "binary_outcome": res["binary_outcome"],
            "n": res["n"],
            "method": (
                "Algorithm-agnostic variable importance from %s "
                "predictiveness" % res["measure"]
            ),
        }
    )


def cheatsheet():
    return (
        "tmlavp: variable importance as lost predictiveness -- efficient "
        "without a correction, cross-fitted for flexibility, "
        "sample-split so the zero-importance null is testable"
    )
