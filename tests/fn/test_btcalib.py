"""Tests for btcalib (bootstrap-calibrated interval, Loh 1991).

Replaces the generated stub, which imported ``boot_calibrated_ci``.
"""

from morie.fn.btcalib import btcalib


def _sample(n=60, seed=3):
    st = [seed]

    def r():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return [10.0 + 4.0 * (r() - 0.5) for _ in range(n)]


def test_the_interval_brackets_the_mean():
    x = _sample()
    res = btcalib(x, alpha=0.05, B=200, seed=1)
    assert res["lower"] < res["estimate"] < res["upper"]
    assert abs(res["estimate"] - sum(x) / len(x)) < 1e-12


def test_calibration_reports_what_it_changed():
    res = btcalib(_sample(), alpha=0.05, B=300, seed=1)
    assert 0.0 < res["alpha_prime"] < 1.0
    assert res["z_calibrated"] > 0
    # identity_gap says how far the nominal level was from the attained one
    assert abs(res["identity_gap"]) < 0.5


def test_a_tighter_alpha_gives_a_wider_interval():
    x = _sample()
    wide = btcalib(x, alpha=0.01, B=200, seed=1)
    narrow = btcalib(x, alpha=0.20, B=200, seed=1)
    assert (wide["upper"] - wide["lower"]) > \
        (narrow["upper"] - narrow["lower"])


def test_seed_reproducibility():
    x = _sample()
    a = btcalib(x, B=100, seed=11)
    b = btcalib(x, B=100, seed=11)
    assert a["lower"] == b["lower"] and a["upper"] == b["upper"]


def test_validation():
    for call in (lambda: btcalib([1.0, 2.0]),
                 lambda: btcalib(_sample(), alpha=0.0),
                 lambda: btcalib(_sample(), alpha=1.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
