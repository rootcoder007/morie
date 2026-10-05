"""Fast Hawkes fitting (1.4.0): the shared C++ core (likelihood, analytic gradient, residuals,
projected BFGS), its routes (exact, sum of exponentials, truncated, EM, INAR), and the pieces
the arms share. The reference fits below were computed by this code and are asserted verbatim
by rmoriebricklayer's tests on the same splitmix64 events: a cross-language parity check.
"""

import math

import pytest

from morie import tps_hawkes_advanced as H

core = pytest.importorskip("morie._core")
KERNELS = ("exponential", "weibull", "gamma", "lomax")
BASELINES = ("constant", "sinusoidal")

REF = {
    "exponential/constant": (118.69858088105596, [-0.018591937991204354, 0.3570981557350094, 17.45814437003573]),
    "exponential/sinusoidal": (
        116.36039351077994,
        [
            0.1401679237176167,
            -0.41637456168187165,
            -0.09236260967814879,
            -0.3202571656173522,
            0.35328832763395485,
            17.76625382246286,
        ],
    ),
    "weibull/constant": (
        116.70074310550106,
        [0.005222397486131046, 0.3416039447382219, 1.2237668778195339, 0.05153286699576664],
    ),
    "gamma/constant": (116.00586580075705, [-0.007363356926464604, 0.34983843433834755, 1.3153317779470712, 25.0]),
    "lomax/constant": (118.7462762190834, [-0.021698915606398167, 0.3590927995717976, 30.0, 1.7084589308626537]),
}


def events(n=400):
    """Clustered event times from splitmix64 (rmoriebricklayer's tests build the same vector)."""
    u = H.splitmix_uniforms(n, 7)
    v = H.splitmix_uniforms(n, 8)
    t, x = [], 0.0
    for a, b in zip(u, v):
        x += -math.log(1.0 - a) * (0.05 if b < 0.4 else 1.0)
        t.append(x)
    return t, t[-1] + 0.5


def theta_for(kernel, baseline):
    a = [0.1] if baseline == "constant" else [0.1, -0.3, 0.1, -0.2]
    psi = {"exponential": [8.0], "weibull": [0.9, 0.2], "gamma": [1.4, 6.0], "lomax": [2.5, 0.3]}[kernel]
    return a + [0.35] + psi


def test_splitmix_is_the_shared_stream():
    assert H.splitmix_uniforms(3, 42) == [0.7415648787718233, 0.1599103928769201, 0.27860113025513866]
    t, T = events()
    assert t[:3] == [0.49401725975830246, 0.5109480750755095, 2.82116905344865]
    assert T == 261.98663489850486


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("baseline", BASELINES)
def test_analytic_gradient_matches_finite_differences(kernel, baseline):
    t, T = events(250)
    th = theta_for(kernel, baseline)
    f = H._core_objective(t, T, kernel, baseline, "exact", 1e-12)
    _, g = f(th)
    for i in range(len(th)):
        h = 1e-6 * max(1.0, abs(th[i]))
        up, dn = list(th), list(th)
        up[i] += h
        dn[i] -= h
        fd = (f(up)[0] - f(dn)[0]) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("baseline", BASELINES)
def test_core_likelihood_is_the_definition(kernel, baseline):
    # the O(n^2) Python likelihood of the paper's definition, term by term
    t, T = events(200)
    th = theta_for(kernel, baseline)
    a, eta, psi = th[: len(th) - 1 - (1 if kernel == "exponential" else 2)], None, None
    nb = H._n_baseline_params(baseline)
    a, eta, psi = th[:nb], th[nb], tuple(th[nb + 1 :])
    nu = [float(v) for v in H._baseline(t, baseline, tuple(a), T)]
    ll = 0.0
    for i, ti in enumerate(t):
        s = sum(float(v) for v in H._kernel_density([ti - tj for tj in t[:i]], kernel, psi)) if i else 0.0
        ll += math.log(nu[i] + eta * s)
    ll -= H._baseline_integral(T, baseline, tuple(a)) + eta * sum(
        float(v) for v in H._kernel_cdf([T - tj for tj in t], kernel, psi)
    )
    assert H._core_objective(t, T, kernel, baseline, "exact", 1e-12)(th)[0] == pytest.approx(-ll, rel=1e-10)


@pytest.mark.parametrize("kernel", ("lomax", "gamma"))
def test_sum_of_exponentials_is_within_its_bound(kernel):
    t, T = events(1200)
    th = [0.1, 0.35] + ([0.6, 3.0] if kernel == "gamma" else [1.8, 0.4])
    exact = H._core_objective(t, T, kernel, "constant", "exact", 1e-12)(th)[0]
    for eps in (1e-6, 1e-9):
        soe = H._core_objective(t, T, kernel, "constant", "soe", eps)(th)[0]
        assert abs(soe - exact) <= len(t) * eps


@pytest.mark.parametrize("key", sorted(REF))
def test_fits_reproduce_the_shared_reference(key):
    kernel, baseline = key.split("/")
    t, T = events()
    f = H.fit_hawkes_general(t, T, kernel, baseline)
    nll, theta = REF[key]
    assert f["nll"] == pytest.approx(nll, rel=1e-10)
    assert [float(v) for v in f["theta"]] == pytest.approx(theta, abs=1e-6)
    assert 0.0 <= f["ks_pvalue"] <= 1.0 and f["converged"]


def test_em_reaches_the_direct_maximum_and_inar_is_close():
    t, T = events(250)
    direct = H.fit_hawkes_general(t, T, "exponential", "constant")
    em = H.fit_hawkes_general(t, T, "exponential", "constant", method="em")
    assert em["nll"] == pytest.approx(direct["nll"], abs=1e-6)
    # EM's steps shrink before it reaches the maximum, so it finishes with the direct optimiser
    # from its own point: every kernel and the sinusoidal baseline end at the direct maximum (as
    # rmoriebricklayer's test-hawkes-fit.R, on the same 300 events)
    t3, T3 = events(300)
    for kernel, baseline in (
        ("weibull", "constant"),
        ("gamma", "constant"),
        ("lomax", "constant"),
        ("exponential", "sinusoidal"),
    ):
        em = H.fit_hawkes_general(t3, T3, kernel, baseline, method="em")
        direct = H.fit_hawkes_general(t3, T3, kernel, baseline)
        assert em["method"] == "em"
        assert em["nll"] == pytest.approx(direct["nll"], abs=1e-6), (kernel, baseline)
    inar = H.fit_hawkes_general(t, T, "exponential", "constant", method="inar")
    assert inar["method"] == "inar" and abs(inar["branching_ratio"] - direct["branching_ratio"]) < 0.25
    with pytest.raises(ValueError, match="stationary"):
        H.fit_hawkes_general(t, T, "exponential", "sinusoidal", method="inar")
    with pytest.raises(ValueError, match="completely monotone"):
        H.fit_hawkes_general(t, T, "weibull", "constant", method="soe")
    with pytest.raises(ValueError, match="method must be one of"):
        H.fit_hawkes_general(t, T, "exponential", "constant", method="newton")


def test_truncation_is_close_and_the_routes_report_themselves():
    t, T = events(300)
    ex = H.fit_hawkes_general(t, T, "gamma", "constant", method="exact")
    tr = H.fit_hawkes_general(t, T, "gamma", "constant", method="truncate", eps=1e-10)
    assert tr["method"] == "truncate" and tr["eps"] == 1e-10 and ex["eps"] is None
    assert H._resolve_method("auto", "weibull") == "truncate"
    assert abs(tr["nll"] - ex["nll"]) < 1e-3


@pytest.mark.parametrize("kernel", KERNELS)
def test_residuals_are_the_time_rescaling_definition(kernel):
    t, T = events(150)
    th = theta_for(kernel, "sinusoidal")
    U = H._time_rescaling_residuals(th, t, T, kernel, "sinusoidal")
    nb, eta, psi = 4, th[4], tuple(th[5:])
    m = max(256, int(T) + 1)
    grid = [T * q / (m - 1) for q in range(m)]
    vals = [float(v) for v in H._baseline(grid, "sinusoidal", tuple(th[:nb]), T)]
    cum = [0.0]
    for q in range(1, m):
        cum.append(cum[-1] + 0.5 * (vals[q] + vals[q - 1]) * (grid[q] - grid[q - 1]))

    def lam_int(x):
        pos = x / (grid[1] - grid[0])
        q = min(int(pos), m - 2)
        base = cum[q] + (pos - q) * (cum[q + 1] - cum[q])
        prior = [x - tj for tj in t if tj < x]
        return base + eta * sum(float(v) for v in H._kernel_cdf(prior, kernel, psi)) if prior else base

    prev = 0.0
    for i, ti in enumerate(t):
        cur = lam_int(ti)
        assert float(U[i]) == pytest.approx(1.0 - math.exp(-max(cur - prev, 1e-12)), abs=1e-9)
        prev = cur


def test_exact_kolmogorov_matches_r():
    # 1 - P(D_n < d) from R's stats:::pkolmogorov_two_exact (Marsaglia, Tsang & Wang 2003)
    for n, d, want in (
        (10, 0.274, 0.62847961545650433),
        (599, 0.0357, 0.57975087884804655),
        (10000, 0.01, 0.73178087203705255),
    ):
        assert core.ks_pkolmogorov_exact(n, d) == pytest.approx(want, rel=1e-13)


def test_events_to_days_is_deterministic_and_keeps_every_event():
    from morie.fn import _frame_core as pd

    df = pd.DataFrame({"OCC_DATE": ["1/2/2015 5:00:00 AM"] * 3 + ["3/4/2016 12:00:00 PM", "1/1/2010 1:00:00 AM"]})
    t, T = H._events_to_days(df, None)
    assert len(t) == 4  # the 2010 row is before min_year (2014)
    assert t.tolist()[:3] == sorted(t.tolist()[:3]) and 0 <= t.tolist()[0] < 1
    t2, _ = H._events_to_days(df, None)
    assert t.tolist() == t2.tolist()
    t3, _ = H._events_to_days(df, 2)
    assert len(t3) == 2


def test_minimize_with_a_gradient_meets_bounds():
    from morie.fn._sci_core import minimize

    def f(x):
        return (x[0] - 1) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2, [
            2 * (x[0] - 1) - 40 * x[0] * (x[1] - x[0] ** 2),
            20 * (x[1] - x[0] ** 2),
        ]

    r = minimize(f, [-1.0, 2.0], jac=True, method="L-BFGS-B", bounds=[(-3, 0.5), (None, 5)])
    assert list(r.x) == pytest.approx([0.5, 0.25], abs=1e-6)
    r = minimize(f, [-1.0, 2.0], jac=True)
    assert list(r.x) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_frame_fixes_found_on_the_way():
    from morie.fn import _frame_core as pd

    got = pd.to_datetime(["1/1/2014 5:00:00 AM", "12/31/2019 12:30:15 PM", "3/4/2020 12:00 AM", "1/2/2014"]).tolist()
    assert [(d.month, d.day, d.hour, d.minute) for d in got] == [
        (1, 1, 5, 0),
        (12, 31, 12, 30),
        (3, 4, 0, 0),
        (1, 2, 0, 0),
    ]
    assert pd.to_datetime(["1/2/2014"], dayfirst=True).tolist()[0].month == 2
    assert pd.Series([1, 2])[[]].tolist() == []
    assert list(pd.DataFrame({"a": [1], "b": [2]})) == ["a", "b"]


def test_event_times_and_fits_equal_rmories_on_the_same_records():
    # the values rmorie returns for these records (tests/testthat/test-tps_hawkes_advanced.R)
    from morie.fn import _frame_core as pd

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    ks = range(1, 401)
    df = pd.DataFrame(
        {
            "OCC_YEAR": [2015 + k % 2 for k in ks],
            "OCC_MONTH": [months[(k * 5) % 12] for k in ks],
            "OCC_DAY": [1 + (k * 13) % 28 for k in ks],
        }
    )
    t, horizon = H._events_to_days(df, 300)
    assert t.size == 300
    assert horizon == pytest.approx(727.3948813285101, rel=1e-12)
    assert float(t.sum()) == pytest.approx(108783.93019224967, rel=1e-12)
    assert [float(v) for v in t[:3]] == pytest.approx(
        [0.1599103928769201, 0.27860113025513866, 0.34419071652363753], rel=1e-12
    )
    ex = H.fit_hawkes_general(t, horizon, "exponential", "constant")
    assert ex["method"] == "exact" and ex["n"] == 300  # the last event, at T, is part of the record
    assert ex["theta"] == pytest.approx([-1.9229920843209447, 0.6523636308313808, 4.235992308382929], rel=1e-6)
    assert ex["nll"] == pytest.approx(342.8276118056356, rel=1e-8)
    wb = H.fit_hawkes_general(t, horizon, "weibull", "constant")
    assert wb["method"] == "truncate"
    assert wb["theta"] == pytest.approx(
        [-1.940282186867362, 0.6590628336268858, 1.3760780701481492, 0.26255518901442443], rel=1e-6
    )
    assert wb["nll"] == pytest.approx(334.0862418399089, rel=1e-8)
