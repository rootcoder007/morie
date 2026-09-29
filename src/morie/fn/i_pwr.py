# morie.fn -- function file (rootcoder007/morie)
"""Calculate statistical power for an ANOVA F-test (interaction power)."""

from __future__ import annotations

from . import _powercore as pc


def calculate_interaction_power(sample_size: int, alpha: float = 0.05, effect_size: float = 0.2, df1: int = 1) -> float:
    r"""Calculate statistical power for an ANOVA F-test (interaction power).

    Power of the F test of an interaction (or any set of ``df1`` regression
    terms) with Cohen's ``f`` effect size in the fixed-effects linear model
    (Cohen 1988, ch. 9; G*Power "linear multiple regression: fixed model, R2
    increase", Faul et al. 2007): ``ncp = f^2 N``, ``df = (df1, N - df1 -
    1)`` and ``power = 1 - F'_{df, ncp}(F_{1 - alpha, df})``. The default
    ``df1 = 1`` is a single-degree-of-freedom interaction.

    :param sample_size: Total number of observations ``N``.
    :param alpha: Significance level.
    :param effect_size: Cohen's f of the tested terms.
    :param df1: Numerator degrees of freedom.
    :return: Power in [0, 1].

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Erlbaum, ch. 9.

    Faul, F., Erdfelder, E., Lang, A.-G. and Buchner, A. (2007). G*Power 3. *Behavior Research
    Methods*, 39(2), 175-191.

    Examples
    --------
    >>> round(calculate_interaction_power(200), 12)
    0.803647504421
    """
    return pc.anova_power(float(sample_size), 2, float(effect_size), float(alpha), df1=int(df1))


i_pwr = calculate_interaction_power


def cheatsheet() -> str:
    return "calculate_interaction_power(N, alpha, f, df1) -> noncentral-F power, ncp = f^2 N"
