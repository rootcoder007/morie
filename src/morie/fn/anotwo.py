# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Two-way ANOVA using sum-of-squares decomposition (Type I)."""

from __future__ import annotations

from . import _frame_core as pd
from . import _stats_core as stats
from ._containers import TestResult
from ._helpers import _validate_df


def _rss(y, cols):
    """Residual sum of squares of y on the columns (list of column lists), by QR."""
    from .nnlsq import _lstsq

    if not cols:
        return sum(v * v for v in y)
    beta = _lstsq(cols, y)
    fit = [sum(c[i] * b for c, b in zip(cols, beta)) for i in range(len(y))]
    return sum((a - f) ** 2 for a, f in zip(y, fit))


def _dummies(levels_of, values):
    return [[1.0 if v == lv else 0.0 for v in values] for lv in levels_of[1:]]


def anova_twoway(
    data: pd.DataFrame, cdf=None, *, y: str = "y", a: str = "a", b: str = "b", interaction: bool = False
) -> TestResult:
    """Two-way ANOVA with sequential (Type I) sums of squares.

    Fits :math:`y_{iju} = \\mu + \\alpha_i + \\delta_j + \\epsilon_{iju}`
    (Hedderich, Sachs & Reynarowych 2023, eq 8.47) or, with
    ``interaction=True``, :math:`\\mu + \\alpha_i + \\delta_j + \\gamma_{ij}`
    (eq 8.48), and decomposes the model sum of squares sequentially,
    :math:`SS(A)`, :math:`SS(B \\mid A)` and :math:`SS(AB \\mid A, B)`, each
    as the drop in residual sum of squares when its dummy columns are added
    -- ``anova(lm(y ~ a + b))`` or ``anova(lm(y ~ a * b))``, also for
    unbalanced data.

    :param data: DataFrame with outcome and two factor columns.
    :param y: Name of the outcome column.
    :param a: Name of the first factor column.
    :param b: Name of the second factor column.
    :param interaction: Include the A x B interaction.
    :return: TestResult for factor A; factor B (and the interaction) in
        ``extra``.
    """
    _validate_df(data, y, a, b)
    df = data[[y, a, b]].dropna()
    yv = [float(v) for v in df[y]]
    av, bv = list(df[a]), list(df[b])
    n = len(yv)
    la = sorted(set(av), key=lambda v: (str(type(v)), v))
    lb = sorted(set(bv), key=lambda v: (str(type(v)), v))
    one = [[1.0] * n]
    da, db = _dummies(la, av), _dummies(lb, bv)
    dab = [[u * w for u, w in zip(ca, cb)] for ca in da for cb in db] if interaction else []
    r0 = _rss(yv, one)
    r1 = _rss(yv, one + da)
    r2 = _rss(yv, one + da + db)
    r3 = _rss(yv, one + da + db + dab) if interaction else r2
    df_a, df_b = len(la) - 1, len(lb) - 1
    df_ab = df_a * df_b if interaction else 0
    ss_a, ss_b, ss_ab = r0 - r1, r1 - r2, r2 - r3
    ss_resid = r3
    df_resid = n - 1 - df_a - df_b - df_ab
    if df_resid <= 0:
        raise ValueError("Not enough observations for two-way ANOVA")
    ms_resid = ss_resid / df_resid

    def ftest(ss, d):
        if d <= 0 or ms_resid <= 0:
            return 0.0, 1.0
        f = ss / d / ms_resid
        return f, float(stats.f.sf(f, d, df_resid))

    f_a, p_a = ftest(ss_a, df_a)
    f_b, p_b = ftest(ss_b, df_b)
    extra = {
        "ss_a": float(ss_a),
        "ss_b": float(ss_b),
        "ss_resid": float(ss_resid),
        "df_b": float(df_b),
        "df_resid": float(df_resid),
        "f_b": float(f_b),
        "p_b": float(p_b),
    }
    if interaction:
        f_ab, p_ab = ftest(ss_ab, df_ab)
        extra.update({"ss_ab": float(ss_ab), "df_ab": float(df_ab), "f_ab": float(f_ab), "p_ab": float(p_ab)})
    return TestResult(
        test_name="Two-way ANOVA (factor A)",
        statistic=float(f_a),
        p_value=float(p_a),
        df=float(df_a),
        n=n,
        method="Two-way ANOVA Type I" + (" with interaction" if interaction else ""),
        extra=extra,
    )


anotwo = anova_twoway


def cheatsheet() -> str:
    return "anova_twoway({}) -> Two-way ANOVA."


# compact alias per ledger/NAMING.md
anovatwoway = anova_twoway
