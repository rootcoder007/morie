"""Joint VaR backtest: Kupiec coverage plus Christoffersen independence."""

from ._richresult import hypothesis_test_result
from .volcc import vol_christoffersen_cc

__all__ = ["vol_garch_var_backtest"]


def vol_garch_var_backtest(hits, alpha=0.05):
    r"""Report the full Christoffersen (2003) VaR backtest triple.

    Runs the decomposition once and returns all three statistics --
    unconditional coverage, independence, conditional coverage -- so a
    caller can see *which* half of the null a model fails. Reporting
    only :math:`LR_{cc}` hides that distinction, and the two failure
    modes call for different fixes: a wrong tail probability is a
    calibration problem, clustered breaches are a dynamics problem.

    Parameters
    ----------
    hits : array-like of {0, 1}
        Exceedance indicator in time order.
    alpha : float
        The VaR tail probability the model claims.

    Returns
    -------
    RichResult
        ``statistic``/``pvalue`` carry LR_cc; ``lr_uc``, ``pvalue_uc``,
        ``lr_ind``, ``pvalue_ind`` carry the two components.

    References
    ----------
    Christoffersen, P. F. (2003). *Elements of Financial Risk
    Management*. Academic Press, ch. 8.
    Kupiec, P. H. (1995). *Journal of Derivatives*, 3(2), 73-84.
    Christoffersen, P. F. (1998). *International Economic Review*,
    39(4), 841-862.
    """
    r = vol_christoffersen_cc(hits, alpha=alpha)
    payload = {k: v for k, v in r.items() if k not in ("statistic", "pvalue")}
    payload["lr_cc"] = float(r["statistic"])
    payload["pvalue_cc"] = float(r["pvalue"])
    payload["method"] = "Joint VaR backtest (Kupiec UC + Christoffersen IND/CC)"
    return hypothesis_test_result(
        test_name="VaR backtest (unconditional + conditional coverage)",
        statistic=float(r["statistic"]),
        pvalue=float(r["pvalue"]),
        extra_summary=[("n_obs", r["n_obs"]), ("n_exceedances", r["n_exceedances"])],
        extra_payload=payload,
    )


def cheatsheet():
    return "volgvar: joint VaR backtest -- Kupiec UC, Christoffersen IND and CC"
