"""Tests for irtpc — partial credit model."""

from morie.fn.irtpc import irtpc


def test_irtpc_basic():
    """PCM step parameters and the sufficiency of the total score.

    The generated version fed the 200x20 five-category `mapq_df` fixture in,
    which is minutes of pure-Python EM, and asserted only that the result had
    an attribute. The contract is the same at n=40, k=3.
    """
    rows = []
    for i in range(40):
        lvl = i % 3
        rows.append([lvl, min(2, lvl + i % 2), max(0, lvl - i % 2)])
    rows[0] = [0, 0, 0]
    rows[1] = [2, 2, 2]

    result = irtpc(rows, n_quad=11, max_iter=15)

    assert result.model == "PCM"
    assert sorted(result.item_params) == ["item_0", "item_1", "item_2"]
    for name, par in result.item_params.items():
        # Three categories -> two step parameters, and no discrimination:
        # Masters (1982) fixes a = 1.
        assert list(par) == ["steps"], (name, list(par))
        assert len(par["steps"]) == 2, (name, par["steps"])
        assert all(-6.0 <= d <= 6.0 for d in par["steps"]), (name, par["steps"])

    assert len(result.theta) == 40
    assert result.fit["n"] == 40
    assert result.fit["k"] == 3
    assert result.fit["loglik"] < 0.0
    assert 1 <= result.fit["n_iter"] <= 15

    # Masters (1982): with a fixed at 1 the only theta-dependent term of the
    # log-likelihood is theta * (total score), so the raw total score is a
    # sufficient statistic -- two respondents with the same total must get the
    # same theta, and a higher total must get a higher theta.
    totals = [sum(r) for r in rows]
    by_total = {}
    for i, t in enumerate(totals):
        by_total.setdefault(t, []).append(float(result.theta[i]))
    for t, vals in by_total.items():
        assert max(vals) - min(vals) < 1e-9, (t, vals)
    ordered = [by_total[t][0] for t in sorted(by_total)]
    assert all(b > a for a, b in zip(ordered, ordered[1:])), ordered


def test_cheatsheet():
    from morie.fn.irtpc import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
