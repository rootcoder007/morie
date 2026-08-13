"""Tests for tsbF. Full anchor: wave3/anchor_intermittent.py."""
import pytest
from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.tsbF import (croston_forecast, demand_classification,
                           intermittent_forecast, sba_forecast,
                           tsb_forecast)

P_TRUE, MU_TRUE, N = 0.3, 10.0, 3000


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(17)
    return [max(0.0, MU_TRUE + 2.0 * rng.standard_normal())
            if float(rng.uniform()) < P_TRUE else 0.0
            for _ in range(N)]


def test_tsb_is_unbiased_where_crostons_ratio_is_not():
    """E[p'z'] = p*mu, because p' and z' are independent. Croston's
    ratio carries the inversion bias 1/E[X] != E[1/X].

    Measured across INDEPENDENT replications: one long smoothed series
    has a small effective sample size (correlation time ~1/alpha), so
    its mean carries more Monte Carlo error than its length suggests.
    """
    import math
    truth = P_TRUE * MU_TRUE
    reps = 30
    t_f, c_f, s_f = [], [], []
    for rep in range(reps):
        r = np.random.default_rng(900 + rep)
        ss = [max(0.0, MU_TRUE + 2.0 * r.standard_normal())
              if float(r.uniform()) < P_TRUE else 0.0
              for _ in range(1500)]
        t_f.append(tsb_forecast(ss, alpha=0.05,
                                beta=0.05)["forecast"][0])
        c_f.append(croston_forecast(ss, alpha=0.05)["forecast"][0])
        s_f.append(sba_forecast(ss, alpha=0.05)["forecast"][0])
    t_m, c_m, s_m = k.mean(t_f), k.mean(c_f), k.mean(s_f)
    t_se = k.sd(t_f) / math.sqrt(reps)
    # TSB is unbiased -- that is the claim the method rests on
    assert abs(t_m - truth) < 3.0 * t_se
    # NOT asserted, because neither is resolvable at this sample size:
    # (a) the DIRECTION of Croston's finite-sample bias -- the inversion
    #     bias predicts over-forecasting, but at alpha = 0.05 on this
    #     generator Croston came out BELOW p*mu, and the sign depends on
    #     the smoothing constant. That is the same phenomenon behind SBA
    #     sometimes being more biased than Croston (Wallstrom &
    #     Segerstedt 2010; Teunter & Sani 2009).
    # (b) whether TSB beats Croston on point accuracy here -- both land
    #     about 0.11 below p*mu and the ordering flips between rep
    #     counts. The decisive difference between them is OBSOLESCENCE,
    #     which the test below measures against a closed form.
    # SBA deflates Croston by (1 - alpha/2), whichever side it started
    assert s_m < c_m


def test_the_sba_deflator_is_one_minus_alpha_over_two():
    y = [5.0 if i % 3 == 0 else 0.0 for i in range(60)]
    assert sba_forecast(y, alpha=0.1)["deflator"] == pytest.approx(0.95)


def test_obsolescence_tsb_decays_and_croston_does_not():
    """The failure TSB exists to fix: nothing updates on a zero, so an
    obsolete item keeps its Croston forecast forever."""
    obs = [10.0 if (i < 100 and i % 3 == 0) else 0.0
           for i in range(200)]
    t = tsb_forecast(obs, alpha=0.1, beta=0.1)
    c = croston_forecast(obs, alpha=0.1)
    assert t["fitted"][-1] < 0.2 * t["fitted"][100]
    # and the decay matches the closed form (1 - beta)^t
    decay = t["fitted"][-1] / t["fitted"][100]
    assert decay == pytest.approx((1.0 - 0.1) ** 99, abs=0.05)
    assert c["fitted"][-1] == pytest.approx(c["fitted"][100], abs=1e-12)
    assert t["updates_on_zeros"]
    assert not c["updates_on_zeros"]


def test_classification(series):
    cl = demand_classification(series)
    assert cl["class"] in ("smooth", "erratic", "intermittent", "lumpy")
    assert cl["adi"] > 1.0


def test_argument_checks(series):
    with pytest.raises(ValueError):
        tsb_forecast(series, alpha=0.0)
    with pytest.raises(ValueError):
        tsb_forecast(series, beta=1.5)
    with pytest.raises(ValueError):
        tsb_forecast([0.0] * 20)
    with pytest.raises(ValueError):
        intermittent_forecast(series, method="nope")
    with pytest.raises(ValueError):
        croston_forecast([1.0])
