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
    assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])


def test_seed_reproducibility():
    x = _sample()
    a = btcalib(x, B=100, seed=11)
    b = btcalib(x, B=100, seed=11)
    assert a["lower"] == b["lower"] and a["upper"] == b["upper"]


def test_validation():
    for call in (
        lambda: btcalib([1.0, 2.0]),
        lambda: btcalib(_sample(), alpha=0.0),
        lambda: btcalib(_sample(), alpha=1.0),
    ):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_loh_calibration_recomputed():
    """Replay Loh's exact calibration on the same uniform stream: beta_i =
    1 - Phi(|t*_i|), alpha' = the 2a-quantile (order statistic) of beta."""
    import math

    from morie.fn import _array_core as np

    x = _sample(n=12)
    n, B, a = 12, 40, 0.05
    rng = np.random.default_rng(5)
    that = sum(x) / n
    sig = math.sqrt(sum((v - that) ** 2 for v in x) / (n - 1))
    betas = []
    for _ in range(B):
        xb = [x[min(int(float(rng.uniform()) * n), n - 1)] for _ in range(n)]
        mb = sum(xb) / n
        sb = math.sqrt(sum((v - mb) ** 2 for v in xb) / (n - 1)) or 1e-300
        t = math.sqrt(n) * (mb - that) / sb
        betas.append(1 - 0.5 * math.erfc(-abs(t) / math.sqrt(2)))
    idx = max(min(math.ceil(2 * (a / 2) * B) - 1, B - 1), 0)
    res = btcalib(x, alpha=a, B=B, seed=5)
    assert abs(res["alpha_prime"] - sorted(betas)[idx]) < 1e-12
    assert abs(res["estimate"] - that) < 1e-12
    half = res["z_calibrated"] * sig / math.sqrt(n)
    assert abs(res["lower"] - (that - half)) < 1e-12
    assert abs(res["upper"] - (that + half)) < 1e-12
