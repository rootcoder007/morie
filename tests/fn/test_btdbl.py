"""Tests for btdbl (double/prepivoted bootstrap, Beran 1987).

Replaces the generated stub, which imported ``boot_double``.
"""

from morie.fn.btdbl import btdbl


def _sample(n=50, seed=4):
    st = [seed]

    def r():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return [5.0 + 2.0 * (r() - 0.5) for _ in range(n)]


def test_the_interval_brackets_the_estimate():
    x = _sample()
    res = btdbl(x, alpha=0.05, B_outer=120, B_inner=60, seed=1)
    assert res["lower"] < res["estimate"] < res["upper"]
    assert abs(res["estimate"] - sum(x) / len(x)) < 1e-12


def test_the_calibrated_level_is_reported():
    res = btdbl(_sample(), alpha=0.05, B_outer=150, B_inner=60, seed=1)
    assert 0.0 < res["c_level"] < 1.0
    assert res["critical_root"] == res["critical_root"]   # not NaN


def test_a_tighter_alpha_widens_the_interval():
    x = _sample()
    tight = btdbl(x, alpha=0.01, B_outer=120, B_inner=60, seed=1)
    loose = btdbl(x, alpha=0.25, B_outer=120, B_inner=60, seed=1)
    assert (tight["upper"] - tight["lower"]) > \
        (loose["upper"] - loose["lower"])


def test_a_custom_statistic_is_used():
    x = _sample()
    res = btdbl(x, statistic=lambda v: max(v), alpha=0.1,
                B_outer=80, B_inner=40, seed=1)
    assert abs(res["estimate"] - max(x)) < 1e-12


def test_seed_reproducibility():
    x = _sample()
    a = btdbl(x, B_outer=60, B_inner=30, seed=8)
    b = btdbl(x, B_outer=60, B_inner=30, seed=8)
    assert a["lower"] == b["lower"] and a["upper"] == b["upper"]


def test_validation():
    for call in (lambda: btdbl([1.0, 2.0]),
                 lambda: btdbl(_sample(), alpha=0.0),
                 lambda: btdbl(_sample(), alpha=1.5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
