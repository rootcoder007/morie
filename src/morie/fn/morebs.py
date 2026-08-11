"""Assuncao-Reis empirical-Bayes-adjusted Moran I for rates."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["morebs", "empirical_bayes_moran"]


def morebs(cases, population, W):
    """
    Empirical-Bayes standardized Moran I for area rates (EBI).

    Raw rates p_i = O_i / n_i from small populations are unstable;
    Assuncao & Reis replace them with the EB-standardized deviates

        b = sum(O) / sum(n)
        s^2 = sum_i n_i (p_i - b)^2 / sum(n)
        a = max(0, s^2 - b / (sum(n)/m))          (moment estimator)
        v_i = a + b / n_i
        z_i = (p_i - b) / sqrt(v_i)

    and compute Moran I on z (centred), with W as given:

        EBI = (m / S0) * sum_i ztil_i (W ztil)_i / sum_i ztil_i^2,
        ztil = z - mean(z),  S0 = sum_ij w_ij.

    Sources
    -------
    Assuncao, R. M. & Reis, E. A. (1999). A new proposal to adjust Moran I
    for population density. *Statistics in Medicine*, 18(16), 2147-2162
    (EB standardization; the moment estimators are Marshall 1991 global
    moment estimators).
    Bivand, R. S., Pebesma, E. & Gomez-Rubio, V. (2013). *Applied Spatial
    Data Analysis with R*, 2nd ed., Springer, Sec. 9.3, p. 282
    (local PDF: WD_BLACK/library/pdf/bivand2013.pdf).
    Reference implementation: spdep::EBImoran.mc / spdep::EBest /
    spdep:::EBImoran (CRAN, read directly; subtract_mean_in_numerator
    convention adopted, the spdep default since Feb 2016).

    Parameters
    ----------
    cases : array-like, (m,)
        Event counts O_i.
    population : array-like, (m,)
        Population at risk n_i (strictly positive).
    W : array-like, (m, m)
        Spatial weights matrix.

    Returns
    -------
    RichResult
        Keys: statistic (EBI), z (EB deviates), rates (raw), eb_rates
        (marginal EB shrunk rates), a, b, S0.
    """
    O = np.asarray(cases, dtype=float).ravel()
    n = np.asarray(population, dtype=float).ravel()
    W = np.asarray(W, dtype=float)
    m = O.size
    if n.size != m:
        raise ValueError("`cases` and `population` must have equal length")
    if W.shape != (m, m):
        raise ValueError(f"W must be ({m}, {m}), got {W.shape}")
    if np.any(n <= 0):
        raise ValueError("population must be strictly positive")
    if np.any(O < 0):
        raise ValueError("cases must be non-negative")
    p = O / n
    b = float(np.sum(O) / np.sum(n))
    s2 = float(np.sum(n * (p - b) ** 2) / np.sum(n))
    a = s2 - b / (float(np.sum(n)) / m)
    if a < 0:
        a = 0.0
    v = a + b / n
    z = (p - b) / np.sqrt(v)
    eb_rates = b + (a * (p - b)) / (a + b / n)
    S0 = float(np.sum(W))
    zt = z - float(np.mean(z))
    lz = W @ zt
    ebi = (m / S0) * float(np.sum(zt * lz)) / float(np.sum(zt**2))
    return RichResult(payload={
        "statistic": float(ebi), "z": z, "rates": p, "eb_rates": eb_rates,
        "a": float(a), "b": b, "s2": s2, "S0": S0, "n": int(m),
        "method": "Assuncao-Reis EB-standardized Moran I",
    })


# long descriptive alias (stub-era name)
empirical_bayes_moran = morebs


def cheatsheet():
    return "morebs: Assuncao-Reis (1999) EB-adjusted Moran I for rates"
