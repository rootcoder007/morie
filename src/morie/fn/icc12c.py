# morie.fn -- slice s05 (rootcoder007/morie)
"""The six Shrout-Fleiss intraclass correlations from one rating table.

Shrout, P. E. and Fleiss, J. L. (1979), "Intraclass correlations: uses
in assessing rater reliability", *Psychological Bulletin* 86(2),
420-428, doi:10.1037/0033-2909.86.2.420.  The paper was opened
directly (na-mic/moodle mirror of the published typescript) and every
formula below was read off a rendered page image, not the text layer.

There is no such thing as "the" ICC.  Six coefficients are defined on
the SAME n-targets-by-k-judges table and they are different numbers:
on the example of the paper itself (Table 2, p. 423) they run from .17 to
.91.  The choice is a statement about the design, not about the
arithmetic:

* Case 1 (one-way): each target is rated by a DIFFERENT set of k
  judges.  Judges are nested in targets, so no judge main effect can
  be estimated and rater bias is pooled into the within-target error.
* Case 2 (two-way random): the same k judges rate every target and
  are a RANDOM SAMPLE of judges.  Systematic judge differences are
  charged against reliability, because a future judge brings their
  own bias.  This is absolute agreement.
* Case 3 (two-way mixed): the same k judges rate every target and are
  the ONLY judges of interest.  Judge offsets cost nothing, so this
  is consistency, not agreement, and is always at least as large as
  Case 2.

The second index is the unit being reported: a single rating, or the
mean of all k.  ICC(3,k) is the Cronbach (1951) alpha (p. 426).

Estimators, all read from the rendered pages (p. 423 and p. 426):

    ICC(1,1) = (BMS - WMS) / (BMS + (k-1) WMS)          p. 423
    ICC(2,1) = (BMS - EMS) / (BMS + (k-1) EMS
                              + k (JMS - EMS) / n)      p. 423
    ICC(3,1) = (BMS - EMS) / (BMS + (k-1) EMS)          p. 423
    ICC(1,k) = (BMS - WMS) / BMS                        p. 426
    ICC(2,k) = (BMS - EMS) / (BMS + (JMS - EMS) / n)    p. 426
    ICC(3,k) = (BMS - EMS) / BMS                        p. 426

with BMS the between-targets, WMS the within-target, JMS the
between-judges and EMS the residual mean square of the targets-by-
judges ANOVA (Table 1, p. 422; the examples values are in Table 3,
p. 423).

All six are returned every time, so the gap between the case a study
assumed and the case it could defend is visible rather than hidden
behind a single reported number.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["icc_two_way"]

_FORMS = ("11", "1k", "21", "2k", "31", "3k")

_NAN = float("nan")


def _div(a, b):
    return a / b if b != 0.0 else _NAN


def _form(model, k):
    """Normalise a form label to one of the six keys.

    Anything spelling out a form is accepted -- ``"ICC(2,k)"``,
    ``"2-k"``, ``"2k"`` -- and, because the paper labels its own table
    with the concrete number of judges (``ICC(1,4)`` for k = 4), a
    trailing integer equal to k is read as the average-measure form.
    """
    s = "".join(c for c in str(model).lower() if c.isdigit() or c == "k")
    if len(s) == 2 and s[1].isdigit() and int(s[1]) == k:
        s = s[0] + "k"
    if s not in _FORMS:
        raise ValueError(
            "icc_two_way: model must name one of the six Shrout-Fleiss "
            "forms ICC(1,1), ICC(1,k), ICC(2,1), ICC(2,k), ICC(3,1), "
            "ICC(3,k); got %r" % (model,))
    return s


def icc_two_way(X, model="2,k"):
    """Intraclass correlations for a complete targets-by-judges table.

    Parameters
    ----------
    X : array-like
        n-by-k matrix; row i holds the k ratings of target i, one per
        judge, in a common judge order.  The design must be complete
        and crossed.
    model : str
        Which of the six forms ``estimate`` reports.  Default
        ``"2,k"``, the two-way random average-measure coefficient.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``model``, ``icc11``, ``icc1k``, ``icc21``,
        ``icc2k``, ``icc31``, ``icc3k``, ``BMS``, ``WMS``, ``JMS``,
        ``EMS``, ``n``, ``k``, ``method``.

    References
    ----------
    Shrout, P. E. and Fleiss, J. L. (1979), *Psychological Bulletin*
    86(2):420-428, doi:10.1037/0033-2909.86.2.420.
    """
    from ._psycho import anova_two_way

    M = core.mat(X)
    n = len(M)
    if n < 2:
        raise ValueError("icc_two_way: need at least two targets")
    k = len(M[0])
    for row in M:
        if len(row) != k:
            raise ValueError(
                "icc_two_way: the design must be complete and crossed; "
                "every target must be rated by the same k judges")
    if k < 2:
        raise ValueError("icc_two_way: need at least two judges")
    y, subject, rater = [], [], []
    for i in range(n):
        for j in range(k):
            y.append(M[i][j])
            subject.append(i)
            rater.append(j)
    a = anova_two_way(y, subject, rater)
    bms = float(a["MSR"])
    jms = float(a["MSC"])
    ems = float(a["MSE"])
    wms = float(a["MSW"])
    icc = {
        "11": _div(bms - wms, bms + (k - 1) * wms),
        "1k": _div(bms - wms, bms),
        "21": _div(bms - ems, bms + (k - 1) * ems + k * (jms - ems) / n),
        "2k": _div(bms - ems, bms + (jms - ems) / n),
        "31": _div(bms - ems, bms + (k - 1) * ems),
        "3k": _div(bms - ems, bms),
    }
    f = _form(model, k)
    return RichResult(payload={
        "estimate": icc[f], "model": f,
        "icc11": icc["11"], "icc1k": icc["1k"],
        "icc21": icc["21"], "icc2k": icc["2k"],
        "icc31": icc["31"], "icc3k": icc["3k"],
        "BMS": bms, "WMS": wms, "JMS": jms, "EMS": ems,
        "n": int(n), "k": int(k),
        "method": "Shrout-Fleiss (1979) ICC(%s,%s)" % (f[0], f[1])})


def cheatsheet():
    return ("icc12c: six ICCs, one table -- .17 to .91 on the Shrout-Fleiss "
            "own example; the case is a design claim, not arithmetic")


# compact alias per ledger/NAMING.md
icctwoway = icc_two_way
