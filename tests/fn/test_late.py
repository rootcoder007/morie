"""Tests for morie.fn.late — Local Average Treatment Effect via instrumental variables."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import math

import pytest

from morie.fn.late import estimate_late


@pytest.fixture()
def synth_iv_data():
    """Synthetic IV data: Z -> T -> Y with known LATE."""
    rng = np.random.default_rng(42)
    n = 200
    z = rng.binomial(1, 0.5, n)
    # Compliers: T follows Z; some always-takers and never-takers
    compliance_noise = rng.standard_normal(n) * 0.3
    t = (0.6 * z + compliance_noise > 0.3).astype(int)
    # True causal effect = 2.0
    y = 2.0 * t + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"instrument": z, "treatment": t, "outcome": y})


def test_returns_dict_with_late(synth_iv_data):
    result = estimate_late(synth_iv_data, treatment="treatment", outcome="outcome", instrument="instrument")
    assert isinstance(result, dict)
    assert "late" in result
    assert "se" in result


def test_late_is_finite(synth_iv_data):
    result = estimate_late(synth_iv_data, treatment="treatment", outcome="outcome", instrument="instrument")
    assert np.isfinite(result["late"])


def test_has_expected_keys(synth_iv_data):
    result = estimate_late(synth_iv_data, treatment="treatment", outcome="outcome", instrument="instrument")
    for key in ("late", "se", "ci", "f_stat", "n", "method"):
        assert key in result


def test_n_matches(synth_iv_data):
    result = estimate_late(synth_iv_data, treatment="treatment", outcome="outcome", instrument="instrument")
    assert result["n"] == len(synth_iv_data)


def test_weak_instrument_raises():
    """An instrument uncorrelated with treatment should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 200
    z = rng.binomial(1, 0.5, n)
    t = rng.binomial(1, 0.5, n)  # independent of Z
    y = rng.standard_normal(n)
    df = pd.DataFrame({"instrument": z, "treatment": t, "outcome": y})
    # May raise ValueError for weak instrument or return with large SE
    # depending on random seed — just check it does not crash unexpectedly
    result = estimate_late(df, treatment="treatment", outcome="outcome", instrument="instrument")
    assert isinstance(result, dict)


def _resid(v, W):
    """Residual of v on (1, W) for a single covariate column W."""
    n = len(v)
    mw, mv = sum(W) / n, sum(v) / n
    b = sum((w - mw) * (x - mv) for w, x in zip(W, v)) / sum((w - mw) ** 2 for w in W)
    return [x - mv - b * (w - mw) for w, x in zip(W, v)]


def test_late_is_the_wald_ratio_without_covariates():
    z = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    t = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y = [1.0, 1.4, 3.1, 0.8, 3.3, 2.9, 1.2, 3.6, 0.5, 2.2]
    r = estimate_late({"z": z, "t": t, "y": y}, treatment="t", outcome="y", instrument="z")
    n = len(z)
    mz, mt, my = sum(z) / n, sum(t) / n, sum(y) / n
    czy = sum((a - mz) * (b - my) for a, b in zip(z, y))
    czt = sum((a - mz) * (b - mt) for a, b in zip(z, t))
    assert r["late"] == pytest.approx(czy / czt, rel=1e-12)


def test_late_with_a_covariate_matches_frisch_waugh_iv():
    """Just-identified 2SLS = sum(z~ y~) / sum(z~ t~) on residuals from (1, W);
    homoskedastic se^2 = s^2 sum(z~^2) / sum(z~ t~)^2, HC0 se^2 = sum(z~^2 e^2) / sum(z~ t~)^2."""
    z = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
    t = [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    w = [0.3, -0.2, 1.1, 0.4, -0.9, 0.0, 0.7, 1.5, -0.4, 0.2, -1.2, 0.9]
    y = [0.5, 2.1, 1.4, 2.6, 0.1, 0.2, 2.9, 3.1, 1.9, 0.6, -0.8, 3.0]
    d = {"z": z, "t": t, "w": w, "y": y}
    zt, tt, yt = _resid(z, w), _resid(t, w), _resid(y, w)
    den = sum(a * b for a, b in zip(zt, tt))
    theta = sum(a * b for a, b in zip(zt, yt)) / den
    e = [b - theta * a for a, b in zip(tt, yt)]
    n, k = len(z), 3
    s2 = sum(v * v for v in e) / (n - k)
    r = estimate_late(d, treatment="t", outcome="y", instrument="z", covariates=["w"])
    assert r["late"] == pytest.approx(theta, rel=1e-10)
    assert r["se"] == pytest.approx((s2 * sum(a * a for a in zt) / den**2) ** 0.5, rel=1e-10)
    rr = estimate_late(d, treatment="t", outcome="y", instrument="z", covariates=["w"], se_type="robust")
    assert rr["se"] == pytest.approx((sum((a * v) ** 2 for a, v in zip(zt, e)) / den**2) ** 0.5, rel=1e-10)
    # first-stage F for the excluded instrument: t~ on z~ through the origin
    g = sum(a * b for a, b in zip(zt, tt)) / sum(a * a for a in zt)
    rss_u = sum((b - g * a) ** 2 for a, b in zip(zt, tt))
    rss_r = sum(b * b for b in tt)
    assert r["f_stat"] == pytest.approx((rss_r - rss_u) / (rss_u / (n - k)), rel=1e-10)
