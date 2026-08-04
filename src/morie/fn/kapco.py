# morie.fn -- function file (rootcoder007/morie)
"""Cohen's Kappa coefficient of agreement for a two-class confusion table."""


from ._richresult import RichResult

__all__ = ['kappacoef', 'kappa_coefficient']


def kappacoef(tp, fp, fn, tn):
    """Cohen's Kappa coefficient of agreement for a two-class confusion table.

    Formula: kappa = (P0 - Pe) / (1 - Pe),  Pe = (tp+fn)/n*(tp+fp)/n + (fp+tn)/n*(fn+tn)/n

    Parameters
    ----------
    tp : float
        True positives.
    fp : float
        False positives.
    fn : float
        False negatives.
    tn : float
        True negatives.

    Returns
    -------
    RichResult
        ``kappa``, ``p0``, ``pe``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 4, Sect. 4.5.2, p. 134, which gives kappa = (P0 - Pe)/(1 - Pe) with P0 the proportion correctly classified and Pe as written above; the book attributes it to Cohen (1960).  Read from the chapter PDF, not recalled.
    """
    tp = float(tp); fp = float(fp); fn = float(fn); tn = float(tn)
    n = tp + fp + fn + tn
    if n <= 0.0:
        raise ValueError("the confusion table must have at least one observation")
    p0 = (tp + tn) / n
    pe = ((tp + fn) / n) * ((tp + fp) / n) + ((fp + tn) / n) * ((fn + tn) / n)
    if pe == 1.0:
        raise ValueError("kappa is undefined when the chance agreement is 1")
    return RichResult(payload={
        "kappa": (p0 - pe) / (1.0 - pe), "p0": p0, "pe": pe, "n": n,
        "method": "Cohen's kappa, MVSML Sect. 4.5.2"})


kappa_coefficient = kappacoef


def cheatsheet():
    return "kapco: Cohen's Kappa coefficient of agreement for a two-class confusion table."
