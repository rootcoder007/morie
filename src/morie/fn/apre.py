# morie.fn -- function file (rootcoder007/morie)
"""Aggregate proportional reduction in error (APRE) for OC/NOMINATE fits."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["oc_apre"]


def oc_apre(votes, predictions):
    r"""APRE per the Armstrong Sec 5.3.5 footnote.

    .. math:: \mathrm{APRE} = \frac{\sum_{j=1}^{q}
              (\text{Minority Vote} - \text{Classification Errors})_j}
              {\sum_{j=1}^{q} \text{Minority Vote}_j},

    the roll-call-aggregated improvement over predicting every choice
    at the modal category. Per-roll-call PRE values are reported too;
    a roll call the model classifies no better than its margin scores
    zero.

    Parameters
    ----------
    votes : array-like, shape (n, q)
        Observed binary votes (1 = yea, 0 = nay, NaN = missing).
    predictions : array-like, shape (n, q)
        Model-predicted votes on the same coding.

    Returns
    -------
    RichResult
        keys: ``apre``, ``per_vote_pre`` (q,), ``minority_total``,
        ``errors_total``, ``n_choices``, ``method``.

    References
    ----------
    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Sec. 5.3.5 footnote, p. 143
    (formula verified against the PDF page).

    Poole, K. T. & Rosenthal, H. (1997). *Congress*. Oxford
    University Press.
    """
    V = np.asarray(votes, dtype=float)
    P = np.asarray(predictions, dtype=float)
    if V.shape != P.shape or V.ndim != 2:
        raise ValueError("votes and predictions must be 2-D arrays of the same shape.")

    minority_total = 0
    errors_total = 0
    n_choices = 0
    per = []
    for j in range(V.shape[1]):
        valid = ~np.isnan(V[:, j])
        y = V[valid, j]
        if y.size == 0:
            per.append(np.nan)
            continue
        pred = P[valid, j]
        yea = int(y.sum())
        minority = min(yea, y.size - yea)
        errors = int((pred != y).sum())
        minority_total += minority
        errors_total += errors
        n_choices += y.size
        per.append((minority - errors) / minority if minority > 0 else np.nan)

    if minority_total == 0:
        raise ValueError("every roll call is unanimous; APRE is undefined.")
    return RichResult(
        payload={
            "apre": float((minority_total - errors_total) / minority_total),
            "per_vote_pre": np.array(per),
            "minority_total": int(minority_total),
            "errors_total": int(errors_total),
            "n_choices": int(n_choices),
            "method": "APRE (Armstrong Sec 5.3.5 footnote, p. 143)",
        }
    )


def cheatsheet():
    return "apre: sum(minority - errors) / sum(minority) across roll calls"


# compact alias per ledger/NAMING.md
ocapre = oc_apre
