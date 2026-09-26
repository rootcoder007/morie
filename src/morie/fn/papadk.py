"""Papadakis nearest-neighbour adjustment for field trials."""

from ._ols_small import ols
from ._richresult import RichResult

__all__ = ["papadakis_analysis"]


def papadakis_analysis(z, row, col, treatment, combined=False):
    r"""Neighbour-adjusted analysis of covariance for a field trial (Papadakis 1937).

    Schabenberger & Gotway (2005, Sec. 6.1.3.2, pp. 318-319): fit the
    treatment-only model :math:`Z(i, j)_{kl} = \mu + \tau_l + \epsilon_{kl}`,
    take its residuals :math:`\hat\epsilon(i, j) = Z(i, j) - \bar Z_l`, and
    replace the block effects of the design model (6.10) by covariates built
    from neighbouring residuals -- the East-West and North-South averages of
    Stroup, Baenziger & Mulitze (1994),

    .. math::

        x_1(i, j) = \tfrac12\{\hat\epsilon(i, j-1) + \hat\epsilon(i, j+1)\},\quad
        x_2(i, j) = \tfrac12\{\hat\epsilon(i-1, j) + \hat\epsilon(i+1, j)\},

    then fit :math:`Z = \beta_0 + \tau_l + \beta_1 x_1 + \beta_2 x_2 + \epsilon^*`
    by OLS. A plot on the edge averages the neighbours it has. With
    ``combined=True`` a single covariate averages all available
    neighbours. This is the non-iterative analysis; the iterated version
    the book mentions has no guaranteed fixed point (rebuilding the
    covariates from the fitted treatment effects, as in Bartlett 1978, can
    settle and then drift away), so it is not offered.

    Parameters
    ----------
    z : sequence of float
    row, col : sequence of int
        Lattice position of each plot; each (row, col) at most once.
    treatment : sequence
        Treatment label of each plot.
    combined : bool

    Returns
    -------
    RichResult
        ``levels``, ``treatment_effects`` (contrasts with the first level),
        ``se_treatment``, ``beta_neighbour``, ``se_neighbour``,
        ``intercept``, ``sigma2``, ``df``, ``adjusted_means`` (at the mean
        covariate values), ``covariates``.

    References
    ----------
    Papadakis, J. S. (1937). Methode statistique pour des experiences sur
    champ. Bulletin de l'Institut d'Amelioration des Plantes a Salonique
    23. Stroup, W. W., Baenziger, P. S. & Mulitze, D. K. (1994). Removing
    spatial variation from wheat yield trials. Crop Science 34, 62-66.
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, Sec. 6.1.3.2, pp. 318-319.
    Bartlett, M. S. (1978). Nearest neighbour models in the analysis of
    field experiments. JRSS B 40, 147-174.
    """
    z = [float(v) for v in z]
    n = len(z)
    if not (len(row) == len(col) == len(treatment) == n):
        raise ValueError("`z`, `row`, `col` and `treatment` must be the same length")
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}
    if len(pos) != n:
        raise ValueError("each (row, col) position may hold only one plot")
    levels = sorted(set(treatment), key=lambda v: (str(type(v)), v))
    if len(levels) < 2:
        raise ValueError("need at least two treatments")
    tmean = {
        lv: sum(z[i] for i in range(n) if treatment[i] == lv) / sum(1 for t in treatment if t == lv) for lv in levels
    }
    resid = [z[i] - tmean[treatment[i]] for i in range(n)]

    def neighbour_mean(i, offsets):
        r, c = int(row[i]), int(col[i])
        vals = [resid[pos[(r + dr, c + dc)]] for dr, dc in offsets if (r + dr, c + dc) in pos]
        return sum(vals) / len(vals) if vals else 0.0

    groups = [[(0, -1), (0, 1), (-1, 0), (1, 0)]] if combined else [[(0, -1), (0, 1)], [(-1, 0), (1, 0)]]
    cov = [[neighbour_mean(i, g) for i in range(n)] for g in groups]
    X = [[1.0] + [1.0 if treatment[i] == lv else 0.0 for lv in levels[1:]] + [c[i] for c in cov] for i in range(n)]
    beta, se, s2, df, _ = ols(X, z)
    t = len(levels) - 1
    cbar = [sum(c) / n for c in cov]
    nb = beta[1 + t :]
    adj = [beta[0] + (beta[k] if k > 0 else 0.0) + sum(b * m for b, m in zip(nb, cbar)) for k in range(t + 1)]
    return RichResult(
        title="Papadakis neighbour-adjusted analysis",
        summary_lines=[("treatments", t + 1), ("covariates", len(groups)), ("sigma2", s2)],
        payload={
            "levels": levels,
            "treatment_effects": beta[1 : 1 + t],
            "se_treatment": se[1 : 1 + t],
            "beta_neighbour": nb,
            "se_neighbour": se[1 + t :],
            "intercept": beta[0],
            "sigma2": s2,
            "df": df,
            "adjusted_means": adj,
            "covariates": cov,
        },
    )


def cheatsheet():
    return "papadk: Papadakis ANCOVA on neighbouring residuals (E-W and N-S averages)"
