# morie.fn -- function file (rootcoder007/morie)
""".632 and .632+ estimators of prediction error, ESL Sec. 7.11."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_oob_632"]


def esl_oob_632(err_train, err_loo_boot, gamma=None, y=None, y_pred=None,
                p1=None, q1=None):
    r"""The ".632 estimator", ESL Eq. (7.57):

    .. math:: \widehat{\mathrm{Err}}^{(.632)} = .368\,\overline{err}
              + .632\,\widehat{\mathrm{Err}}^{(1)} ,

    and the ".632+" refinement, Eq. (7.61):

    .. math:: \widehat{\mathrm{Err}}^{(.632+)} = (1-\hat w)\,
              \overline{err} + \hat w\,\widehat{\mathrm{Err}}^{(1)},
              \qquad \hat w = \frac{.632}{1 - .368\hat R}.

    **The second argument is the LEAVE-ONE-OUT bootstrap (7.56), not
    the naive (7.54).** The distinction is the whole point of the
    section. (7.54) trains and tests on overlapping samples and is
    biased downward; (7.56) removes the overlap but, because a
    bootstrap sample holds only about :math:`0.632N` distinct
    observations, behaves like twofold cross-validation and is biased
    upward. .632 pulls (7.56) back down toward the training error,
    and the constant :math:`.632` is exactly the inclusion
    probability of (7.55). Substituting (7.54) here corrects a bias
    that is not there, in a quantity that is already too small.

    .632 works well in "light fitting" situations and breaks down in
    overfit ones. The book's example: a 1-nearest-neighbour rule on
    two equal classes with targets independent of the labels gives
    :math:`\overline{err} = 0` and
    :math:`\widehat{\mathrm{Err}}^{(1)} = 0.5`, so
    :math:`\widehat{\mathrm{Err}}^{(.632)} = .632 \times 0.5 = 0.316`
    against a true rate of 0.5.

    .632+ fixes that by measuring how much overfitting there is.
    With :math:`\hat\gamma` the *no-information error rate* -- the
    error of the rule if inputs and labels were independent, Eq.
    (7.58) :math:`\hat\gamma = N^{-2}\sum_i\sum_{i'} L(y_i, \hat
    f(x_{i'}))`, or Eq. (7.59) :math:`\hat p_1(1-\hat q_1) + (1-\hat
    p_1)\hat q_1` in the dichotomous case -- the *relative
    overfitting rate* is Eq. (7.60)

    .. math:: \hat R = \frac{\widehat{\mathrm{Err}}^{(1)}
              - \overline{err}}{\hat\gamma - \overline{err}},

    zero when there is no overfitting and one when overfitting equals
    the no-information value. The weight then runs from ``.632`` at
    :math:`\hat R = 0` to ``1`` at :math:`\hat R = 1`, so
    :math:`\widehat{\mathrm{Err}}^{(.632+)}` runs from
    :math:`\widehat{\mathrm{Err}}^{(.632)}` to
    :math:`\widehat{\mathrm{Err}}^{(1)}`. On the 1-NN example
    :math:`\hat w = \hat R = 1` and it returns 0.5, the correct
    answer.

    Parameters
    ----------
    err_train : float
        The training error :math:`\overline{err}`.
    err_loo_boot : float
        The LEAVE-ONE-OUT bootstrap error (7.56), i.e.
        ``esl_bootstrap_err(...)["err_loo_boot"]``.
    gamma : float, optional
        The no-information error rate. Supply this, or ``y`` and
        ``y_pred`` for (7.58), or ``p1`` and ``q1`` for (7.59); .632+
        is omitted when none is available.
    y, y_pred : array-like, optional
        Response and fitted values, for the (7.58) double sum.
    p1, q1 : float, optional
        Observed proportion of responses equal to 1 and of
        predictions equal to 1, for (7.59).

    Returns
    -------
    RichResult
        keys: ``value`` (the .632 estimate), ``err_632``,
        ``err_632_plus`` (None when gamma is unavailable), ``weight``,
        ``relative_overfitting_rate``, ``gamma``, ``err_train``,
        ``err_loo_boot``, ``uses_leave_one_out``, ``method``.

    References
    ----------
    Hastie, Tibshirani and Friedman, *The Elements of Statistical
    Learning*, 2nd ed., Sec. 7.11, Eqs. (7.55)-(7.61). Read from the
    PDF. Efron (1983); Efron and Tibshirani (1997); Breiman et al.
    (1984) for the counterexample.
    """
    et = float(err_train)
    e1 = float(err_loo_boot)
    if not np.isfinite(et) or not np.isfinite(e1):
        raise ValueError("both error estimates must be finite.")
    err632 = 0.368 * et + 0.632 * e1

    if gamma is None and y is not None and y_pred is not None:
        yv = np.asarray(y, dtype=float).ravel()
        pv = np.asarray(y_pred, dtype=float).ravel()
        if yv.size != pv.size:
            raise ValueError(
                f"y has {yv.size} entries and y_pred has {pv.size}.")
        # (7.58): every response paired with every prediction
        gamma = float(np.mean((yv[:, None] - pv[None, :]) ** 2))
    if gamma is None and p1 is not None and q1 is not None:
        pp, qq = float(p1), float(q1)
        if not (0 <= pp <= 1 and 0 <= qq <= 1):
            raise ValueError("p1 and q1 are proportions and must lie in [0, 1].")
        gamma = pp * (1 - qq) + (1 - pp) * qq            # (7.59)

    w = None
    R = None
    err632p = None
    if gamma is not None:
        g = float(gamma)
        if g <= et:
            # gamma is the error with NO information; a rule that beats
            # it on its own training data leaves (7.60) undefined
            R = 0.0
        else:
            R = (e1 - et) / (g - et)                      # (7.60)
            R = float(min(max(R, 0.0), 1.0))
        w = 0.632 / (1.0 - 0.368 * R)                    # (7.61)
        err632p = (1.0 - w) * et + w * e1
    return RichResult(payload={
        "value": err632, "err_632": err632, "err_632_plus": err632p,
        "weight": w, "relative_overfitting_rate": R, "gamma": gamma,
        "err_train": et, "err_loo_boot": e1,
        "uses_leave_one_out": True,
        "second_argument_note": "(7.57) takes Err^(1) from (7.56), NOT "
                                "Err_boot from (7.54); the latter is biased "
                                "downward and needs no correction downward",
        "weight_range": "w runs from .632 at R = 0 to 1 at R = 1, so the "
                        ".632+ estimate runs from Err^(.632) to Err^(1)",
        "method": "ESL (7.57) .632 and (7.61) .632+ prediction-error estimators"})


def cheatsheet():
    return "eslo63: .632 takes Err^(1) (7.56), not Err_boot (7.54) -- and .632+ handles overfit rules"
