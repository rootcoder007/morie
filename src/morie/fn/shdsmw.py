# morie.fn -- function file (rootcoder007/morie)
r"""Marginal structural model with regularized propensity weights.

Setoguchi et al. (2008) and Westreich, Lessler & Funk (2010) both ask
the same question -- what happens to an IP-weighted estimate when the
propensity model is fitted by something other than plain logistic
regression -- and reach a shared practical conclusion: a flexible or
penalized propensity model can reduce bias when the true model is
non-additive, but it also produces more extreme weights, and the
variance can go the wrong way. Neither paper says "use the flexible
model"; both say "look at what it does to the weights".

So this module fits the propensity by *ridge-penalized* logistic
regression with penalty ``lam``, forms the Sec. 12.3 stabilized
weights, fits the MSM, and reports the diagnostics that make the
trade-off visible rather than a single number that hides it: the
effective sample size, the largest weight, and the estimate's
sensitivity across a path of penalties.

``lam=0`` is ordinary maximum likelihood and reproduces the unpenalized
MSM exactly. Increasing ``lam`` shrinks the propensity coefficients
toward zero, which pulls the fitted probabilities toward the marginal
prevalence, which pulls the weights toward 1 -- so a large enough
penalty converges on the *unadjusted* estimate. That limit is not a
defect; it is the bias-variance trade the two papers are about, and the
anchor checks that the path actually runs between those two endpoints
rather than wandering.

**Shrinkage is applied to the slopes only.** Penalizing the intercept
would shrink the fitted prevalence itself, which is not a
regularization of the confounding relationship but a distortion of the
marginal treatment rate, and it would break the ``lam -> infinity``
limit above.

References
----------
Setoguchi, S., Schneeweiss, S., Brookhart, M. A., Glynn, R. J. & Cook,
E. F. (2008) "Evaluating uses of data mining techniques in propensity
score estimation: a simulation study", *Pharmacoepidemiology and Drug
Safety* 17(6), 546-555, doi:10.1002/pds.1555.

Westreich, D., Lessler, J. & Funk, M. J. (2010) "Propensity score
estimation: neural networks, support vector machines, decision trees
(CART), and meta-classifiers as alternatives to logistic regression",
*Journal of Clinical Epidemiology* 63(8), 826-833,
doi:10.1016/j.jclinepi.2009.11.020.

Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*,
Chapman & Hall/CRC, Sec. 12.3 -- the stabilized weights and the
mean-1 diagnostic; Fine Point 12.2 on checking positivity.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["shrinkage_msm", "penalty_path"]


def shrinkage_msm(y, treatment_history, covariate_history, lam=0.0,
                  contrast="cumulative", path=None, trim=None):
    r"""IP-weighted MSM whose propensity model is ridge-penalized.

    Parameters
    ----------
    y : array-like
        Outcome.
    treatment_history : list of array-like
        Treatment at each time point; a single vector is one time point.
    covariate_history : list of array-like
        Covariates at each time point.
    lam : float
        Ridge penalty on the propensity model's slopes. 0 is ordinary
        maximum likelihood.
    path : list of float, optional
        Extra penalties to evaluate, so the estimate's sensitivity to
        the choice is reported rather than left to the reader's
        imagination. Defaults to a short log-spaced path.

    Returns
    -------
    RichResult
        ``estimate`` at the requested `lam`, with ``path`` giving the
        estimate, effective sample size and largest weight at each
        penalty on the path.

    Examples
    --------
    Compare the unpenalized fit against a heavily penalized one::

        r = shrinkage_msm(y, [A0, A1], [L0, L1], lam=1.0)
        r["estimate"], r["path"]
    """
    if float(lam) < 0.0:
        raise ValueError("shrinkage_msm: lam must be non-negative, got %r"
                         % (lam,))
    A_hist = _hist(treatment_history)
    L_hist = _hist(covariate_history, allow_none=True)
    if len(L_hist) != len(A_hist):
        raise ValueError("shrinkage_msm: %d treatment times but %d "
                         "covariate blocks" % (len(A_hist), len(L_hist)))
    yv = k.vec(y)
    n = len(yv)

    def fit_at(lm):
        w, per = k.ip_weights_history(A_hist, L_hist,
                                      penalty=float(lm), trim=trim)
        cum = [sum(k.vec(a)[i] for a in A_hist) for i in range(n)]
        if contrast == "cumulative":
            e = cum
        elif contrast == "final":
            e = list(k.vec(A_hist[-1]))
        elif contrast == "everexposed":
            e = [1.0 if v > 0.0 else 0.0 for v in cum]
        else:
            raise ValueError("shrinkage_msm: contrast must be "
                             "'cumulative', 'final' or 'everexposed', "
                             "got %r" % (contrast,))
        f = k.wls([[v] for v in e], yv, w)
        s1, s2 = sum(w), sum(v * v for v in w)
        return {"lam": float(lm), "estimate": f["coef"][1],
                "se": f["se"][1], "weights": w,
                "mean_weight": s1 / n, "max_weight": max(w),
                "effective_sample_size": (s1 * s1 / s2) if s2 else 0.0,
                "exposure": e}

    main = fit_at(lam)
    if path is None:
        path = [0.0, 1e-4, 1e-2, 0.1, 1.0, 10.0, 1e3]
    rows = []
    for lm in path:
        r = fit_at(lm)
        rows.append({"lam": r["lam"], "estimate": r["estimate"],
                     "se": r["se"], "max_weight": r["max_weight"],
                     "effective_sample_size":
                         r["effective_sample_size"]})
    unadj = k.wls([[v] for v in main["exposure"]], yv, [1.0] * n)

    out = dict(main)
    out.update({"path": rows, "unadjusted": unadj["coef"][1],
                "n": n, "n_times": len(A_hist), "contrast": contrast,
                "method": "MSM with ridge-penalized propensity weights, "
                          "Setoguchi et al. (2008) and Westreich, Lessler "
                          "& Funk (2010); weights per Hernan & Robins "
                          "(2020) Sec. 12.3"})
    return RichResult(payload=out)


def penalty_path(y, treatment_history, covariate_history, path=None,
                 contrast="cumulative"):
    """Just the path, for when the sensitivity is the whole question."""
    r = shrinkage_msm(y, treatment_history, covariate_history, lam=0.0,
                      contrast=contrast, path=path)
    return r["path"]


def _hist(obj, allow_none=False):
    if obj is None:
        return [None] if allow_none else []
    if isinstance(obj, (list, tuple)) and obj and (
            isinstance(obj[0], (list, tuple)) or obj[0] is None
            or hasattr(obj[0], "shape")):
        return list(obj)
    return [obj]


def cheatsheet():
    return ("shdsmw: MSM with a ridge-penalized propensity model "
            "(Setoguchi 2008; Westreich 2010). lam=0 is plain MLE; "
            "lam -> inf shrinks the weights to 1 and the estimate to "
            "the unadjusted one. Reports ESS and max weight along a "
            "penalty path.")


# compact alias per ledger/NAMING.md
shrinkagemsm = shrinkage_msm
