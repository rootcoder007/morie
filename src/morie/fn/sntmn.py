"""Mandatory minimum sentence analysis."""

from __future__ import annotations

import math

from morie.fn._containers import DescriptiveResult


def _column(df, name):
    col = df[name]
    col = col.tolist() if hasattr(col, "tolist") else list(col)
    return col


def sentence_mandatory_min(
    df,
    *,
    offense_col: str = "offense",
    sentence_col: str = "sentence_days",
    min_col: str = "mandatory_min_days",
) -> DescriptiveResult:
    r"""Mandatory minimum sentence analysis.

    Compares each imposed sentence with the mandatory minimum that applied
    to it: the share of cases sentenced exactly at the minimum, below it
    (relief, e.g. safety-valve or substantial-assistance departures) and
    above it, and the mean excess ``sentence - minimum`` over the cases
    above -- overall and by offense (the measures of the U.S. Sentencing
    Commission's 2017 mandatory-minimum report).

    Parameters
    ----------
    df : mapping or data frame
        Columns ``offense_col``, ``sentence_col``, ``min_col``.
    offense_col, sentence_col, min_col : str
        Column names.

    Returns
    -------
    DescriptiveResult
        ``value`` is the share at the minimum; ``extra`` has
        ``pct_at_minimum``, ``pct_below_minimum``, ``pct_above_minimum``,
        ``mean_above_minimum``, ``n`` and ``by_offense``.

    References
    ----------
    United States Sentencing Commission (2017). *Mandatory Minimum Penalties in the Federal Criminal
    Justice System*. Washington, DC.

    Examples
    --------
    >>> d = {"offense": ["a", "a", "b", "b"], "sentence_days": [60, 90, 30, 40], "mandatory_min_days": [60, 60, 30, 60]}
    >>> r = sentence_mandatory_min(d)
    >>> r.value, r.extra["pct_below_minimum"], r.extra["mean_above_minimum"]
    (0.5, 0.25, 30.0)
    """
    off = _column(df, offense_col)
    s = [float(v) for v in _column(df, sentence_col)]
    m = [float(v) for v in _column(df, min_col)]
    if not (len(off) == len(s) == len(m)) or not s:
        raise ValueError("columns must be non-empty and of equal length")

    def summary(idx):
        d = [s[i] - m[i] for i in idx]
        k = len(d)
        above = [v for v in d if v > 0]
        return {
            "pct_at_minimum": sum(1 for v in d if v == 0) / k,
            "pct_below_minimum": sum(1 for v in d if v < 0) / k,
            "pct_above_minimum": len(above) / k,
            "mean_above_minimum": math.fsum(above) / len(above) if above else 0.0,
            "n": k,
        }

    overall = summary(range(len(s)))
    labels = []
    for o in off:
        if o not in labels:
            labels.append(o)
    overall["by_offense"] = {o: summary([i for i in range(len(s)) if off[i] == o]) for o in labels}
    return DescriptiveResult(name="sentence_mandatory_min", value=overall["pct_at_minimum"], extra=overall)


sntmn = sentence_mandatory_min


def cheatsheet() -> str:
    return "sentence_mandatory_min(df) -> shares at / below / above the mandatory minimum, mean excess"
