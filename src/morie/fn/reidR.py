# morie.fn -- function file (rootcoder007/morie)
"""Re-identification risk from quasi-identifier equivalence classes."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["reidentification_risk"]


def reidentification_risk(quasi_identifiers, sample_fraction=1.0,
                          threshold=0.05, attack="prosecutor"):
    r"""Risk under the three standard attacker models.

    Records sharing a quasi-identifier combination form an equivalence
    class of size :math:`f_j`. The three risks El Emam defines are

    .. math::
       R_{prosecutor} = \max_j \frac{1}{f_j},\qquad
       R_{journalist} = \frac{1}{\min_j f_j},\qquad
       R_{marketer} = \frac{1}{N}\sum_j \frac{n_j}{f_j},

    differing only in what the attacker is assumed to know and want.
    The PROSECUTOR knows a specific person is in the data and targets
    them; the JOURNALIST wants any one re-identification and so is
    bounded by the rarest record; the MARKETER wants many and is
    measured on the average. They are not interchangeable, and quoting
    the marketer risk for a dataset facing a prosecutor threat
    understates the exposure by orders of magnitude.

    Equivalence classes of size 1 are UNIQUES, and they are the whole
    problem: a single unique record makes the prosecutor and journalist
    risks both 1.0 regardless of how large and safe the rest of the
    data is. ``n_unique`` counts them and ``k_anonymity`` reports the
    smallest class size, which is the k the data actually satisfies.

    When the released data is a SAMPLE, population class sizes exceed
    sample ones and the sample risk overstates. ``sample_fraction``
    applies the standard correction; leaving it at 1 treats the release
    as a census, which is conservative and usually wrong for survey
    data.

    Parameters
    ----------
    quasi_identifiers : array-like, shape (n, p)
        Columns are the quasi-identifiers; values may be any hashable.
    sample_fraction : float
        Share of the population the release represents.
    threshold : float
        Acceptable risk, conventionally 0.05 or 0.09.
    attack : {'prosecutor', 'journalist', 'marketer'}

    Returns
    -------
    RichResult
        ``risk``, ``prosecutor_risk``, ``journalist_risk``,
        ``marketer_risk``, ``k_anonymity``, ``n_unique``,
        ``acceptable``, ``class_sizes``.

    References
    ----------
    El Emam, Dankar, Vaillancourt, Roffey and Lysyk (2011), *Medical
    Decision Making* 31:e1-e12.
    Sweeney (2002) for k-anonymity.

    Examples
    --------
    >>> q = [["M", 30], ["M", 30], ["F", 40], ["F", 40]]
    >>> out = reidentification_risk(q)
    >>> float(out["prosecutor_risk"])
    0.5
    """
    Q = np.atleast_2d(np.asarray(quasi_identifiers, dtype=object))
    if Q.ndim != 2:
        raise ValueError("quasi_identifiers must be 2-dimensional.")
    n = Q.shape[0]
    if n < 1:
        raise ValueError("need at least one record.")
    if not 0 < sample_fraction <= 1:
        raise ValueError(
            "sample_fraction must lie in (0, 1], got %r." % sample_fraction
        )
    if attack not in ("prosecutor", "journalist", "marketer"):
        raise ValueError(
            "attack must be prosecutor, journalist or marketer, got %r."
            % attack
        )

    keys = [tuple(row) for row in Q]
    uniq = {}
    for k in keys:
        uniq[k] = uniq.get(k, 0) + 1
    sizes = np.array(sorted(uniq.values()))
    # population class sizes under simple random sampling
    pop = sizes / sample_fraction

    prosecutor = float(np.max(1.0 / pop))
    journalist = float(1.0 / np.min(pop))
    marketer = float(np.sum(sizes / pop) / n)
    risk = {"prosecutor": prosecutor, "journalist": journalist,
            "marketer": marketer}[attack]
    k_anon = int(sizes.min())
    n_uni = int(np.sum(sizes == 1))
    return RichResult(
        payload={
            "estimate": risk,
            "risk": risk,
            "prosecutor_risk": prosecutor,
            "journalist_risk": journalist,
            "marketer_risk": marketer,
            "attack": attack,
            "attack_note": (
                "the three differ in what the attacker knows and wants and "
                "are not interchangeable; quoting the marketer risk against "
                "a prosecutor threat understates exposure by orders of "
                "magnitude"
            ),
            "k_anonymity": k_anon,
            "n_unique": n_uni,
            "unique_note": (
                None if n_uni == 0 else
                "%d record(s) are unique on the quasi-identifiers; a single "
                "unique makes the prosecutor and journalist risks 1.0 "
                "however large and safe the rest of the data is" % n_uni
            ),
            "class_sizes": sizes,
            "n_classes": int(sizes.size),
            "mean_class_size": float(sizes.mean()),
            "acceptable": bool(risk <= threshold),
            "threshold": float(threshold),
            "sample_fraction": float(sample_fraction),
            "sampling_note": (
                "population classes are larger than sample ones, so a census "
                "assumption (fraction 1) overstates risk; that is the "
                "conservative direction and usually wrong for survey data"
            ),
            "n": int(n),
            "method": "Re-identification risk (%s model)" % attack,
        }
    )


def cheatsheet():
    return (
        "reidR: prosecutor, journalist and marketer risks with k-anonymity "
        "and the uniques that drive them"
    )
