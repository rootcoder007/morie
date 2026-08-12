"""CAT precision stopping rule (Wainer & Mislevy; Magis & Raiche 2012)."""

import math

from ._richresult import RichResult

__all__ = ["catstop", "cat_stopping_rule"]


def _item_info_4pl(theta, a, b, c, d):
    # catR Eq. 1 (4PL) and Eq. 4: I_j = P'^2 / (P Q)
    e = math.exp(a * (theta - b))
    p = c + (d - c) * e / (1.0 + e)
    q = 1.0 - p
    dp = a * (d - c) * e / (1.0 + e) ** 2
    if p <= 0.0 or q <= 0.0:
        return 0.0, p
    return dp * dp / (p * q), p


def catstop(items, theta, se_target, estimator="ML", prior_var=1.0):
    """
    Precision-based stopping rule for computerized adaptive testing.

    Computes the provisional standard error of the ability estimate
    from the administered items and stops the CAT when it falls at or
    below the target: the precision criterion of Magis & Raiche
    (2012, Sec. 2-3), catR Eqs. 3-4 for the ML case,

        se(theta_ML) = 1 / sqrt(sum_j I_j(theta)),
        I_j = [P_j'(theta)]^2 / (P_j Q_j),

    with the four-parameter logistic item response function
    P_j = c_j + (d_j - c_j) exp(a_j (theta-b_j)) / (1 + exp(...))
    (catR Eq. 1; Barton & Lord 1981).  For the Bayes-modal estimator
    with a normal prior of variance sigma^2, catR Eq. 6:
    se = 1 / sqrt(1/sigma^2 + sum_j I_j).

    Sources
    -------
    Magis, D. & Raiche, G. (2012). Random generation of response
    patterns under computerized adaptive testing with the R package
    catR. *Journal of Statistical Software*, 48(8), Eqs. 1, 3, 4, 6
    and the stopping-rule discussion (local copy
    fetched-wave3/magis-raiche-2012-catR-JSS48.pdf).
    Wainer, H. & Mislevy, R. J. (2000). Item response theory,
    item calibration, and ability estimation. In H. Wainer (ed.),
    *Computerized Adaptive Testing: A Primer*, 2nd ed., Erlbaum
    (the SE-based stopping criterion, as cited by the stub).

    Parameters
    ----------
    items : sequence of (a, b, c, d) tuples
        Parameters of the items administered so far.  Pass
        (a, b, 0, 1) for 2PL, (1, b, 0, 1) for 1PL items.
    theta : float
        Current (provisional) ability estimate.
    se_target : float
        Precision target; the CAT stops when se <= se_target.
    estimator : str
        "ML" (default) or "BM" (normal prior, catR Eq. 6).
    prior_var : float
        Prior variance sigma^2 for the BM estimator.

    Returns
    -------
    RichResult
        Keys: stop (bool), se, information (test information),
        item_information, n_items.
    """
    th = float(theta)
    tgt = float(se_target)
    if tgt <= 0:
        raise ValueError("se_target must be positive")
    est = str(estimator).upper()
    if est not in ("ML", "BM"):
        raise ValueError("estimator must be 'ML' or 'BM'")
    infos = []
    for it in items:
        a, b, c, d = (float(v) for v in it)
        if not (0.0 <= c < d <= 1.0):
            raise ValueError("item parameters need 0 <= c < d <= 1")
        info, _ = _item_info_4pl(th, a, b, c, d)
        infos.append(info)
    total = sum(infos)
    if est == "BM":
        pv = float(prior_var)
        if pv <= 0:
            raise ValueError("prior_var must be positive")
        denom = 1.0 / pv + total
    else:
        denom = total
    se = float("inf") if denom <= 0 else 1.0 / math.sqrt(denom)
    return RichResult(payload={
        "stop": bool(se <= tgt),
        "se": se,
        "information": total,
        "item_information": infos,
        "n_items": len(infos),
        "estimator": est,
        "se_target": tgt,
        "method": "CAT precision stopping rule (catR Eqs. 3-4, 6)",
    })


# long descriptive alias (stub-era name)
cat_stopping_rule = catstop


def cheatsheet():
    return "catstop: stop CAT when se(theta) = 1/sqrt(sum I_j) <= target"
