"""Weighting-class nonresponse adjustment (Lohr 2010, Sec. 8.5.1)."""

from ._richresult import RichResult

__all__ = ["respwt", "response_weight_adjustment"]


def respwt(weights, responded, classes):
    """
    Weighting-class adjustment of survey weights for nonresponse.

    Lohr (2010), Sec. 8.5.1: within each weighting class c estimate
    the response probability by

        phi_hat_c = (sum of weights for respondents in c)
                    / (sum of weights for the selected sample in c),

    and multiply each respondent's weight by 1/phi_hat_c; the
    respondents in class c then carry the nonrespondents' share of
    the population as well as their own.  By construction the
    adjusted respondent weights in each class sum exactly to the
    class's original full-sample weight total -- an identity this
    implementation verifies.  Her Example 8.4/Table 8.2 (weight
    factor 1.622 for the 15-24 class, phi_hat = 0.6165) is the
    printed test case.

    Sources
    -------
    Lohr, S. L. (2010). *Sampling: Design and Analysis*, 2nd ed.,
    Brooks/Cole, Sec. 8.5.1, Example 8.4 and Table 8.2 (local copy
    fetched-wave3/Sampling_Design_and_Analysis.pdf).

    Parameters
    ----------
    weights : sequence of float
        Sampling weights w_i = 1/pi_i for the SELECTED sample.
    responded : sequence of bool
        Response indicator per selected unit.
    classes : sequence
        Weighting-class label per selected unit.

    Returns
    -------
    RichResult
        Keys: adjusted (per unit; None for nonrespondents),
        phi_hat ({class: response probability}), factors
        ({class: 1/phi_hat}), balance_error (max class imbalance).
    """
    w = [float(v) for v in weights]
    n = len(w)
    if len(responded) != n or len(classes) != n or n == 0:
        raise ValueError("weights, responded, classes must be paired")
    if any(v <= 0 for v in w):
        raise ValueError("weights must be positive")
    tot = {}
    resp = {}
    for i in range(n):
        c = classes[i]
        tot[c] = tot.get(c, 0.0) + w[i]
        if responded[i]:
            resp[c] = resp.get(c, 0.0) + w[i]
    phi = {}
    fac = {}
    for c in tot:
        if resp.get(c, 0.0) <= 0.0:
            raise ValueError("class %r has no respondents" % (c,))
        phi[c] = resp[c] / tot[c]
        fac[c] = 1.0 / phi[c]
    adjusted = [w[i] * fac[classes[i]] if responded[i] else None
                for i in range(n)]
    # exact balance identity per class
    bal = 0.0
    for c in tot:
        s = sum(adjusted[i] for i in range(n)
                if responded[i] and classes[i] == c)
        bal = max(bal, abs(s - tot[c]))
    return RichResult(payload={
        "adjusted": adjusted,
        "phi_hat": {str(k): v for k, v in phi.items()},
        "factors": {str(k): v for k, v in fac.items()},
        "balance_error": bal,
        "n": n,
        "method": "weighting-class adjustment (Lohr 2010, Sec. 8.5.1)",
    })


# long descriptive alias (stub-era name)
response_weight_adjustment = respwt


def cheatsheet():
    return "respwt: phi_c = sum w_resp / sum w_all; w_adj = w / phi_c"
