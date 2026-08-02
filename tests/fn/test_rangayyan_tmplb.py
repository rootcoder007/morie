"""Rangayyan template-B repairs: basic statistics, FIR filters and the
adaptive-filter quantities.

These modules previously computed `np.median(arg)` and a NaN standard
error while ignoring every real argument. Each test therefore asserts
against the closed form the docstring states."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgmavg import rangayyan_moving_average
from morie.fn.rng007 import rangayyan_ch3_sample_mean
from morie.fn.rng008 import rangayyan_ch3_sample_mean_squared
from morie.fn.rng009 import rangayyan_ch3_sample_rms
from morie.fn.rng010 import rangayyan_ch3_sample_std
from morie.fn.rng039 import rangayyan_ch3_ma_filter_11pt
from morie.fn.rng087 import rangayyan_ch3_ma_filter_general
from morie.fn.rng097 import rangayyan_ch3_ma_8point
from morie.fn.rng137 import rangayyan_ch3_estimation_error
from morie.fn.rng140 import rangayyan_ch3_estimation_error_vector_form
from morie.fn.rng156 import rangayyan_ch3_lms_estimation_error
from morie.fn.rng159 import rangayyan_ch3_lms_gradient_estimate
from morie.fn.rng165 import rangayyan_ch3_rls_phi_matrix
from morie.fn.rng166 import rangayyan_ch3_rls_theta_vector
from morie.fn.rng194 import rangayyan_ch4_heart_rate_from_rr


def test_basic_statistics_match_their_closed_forms():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert rangayyan_ch3_sample_mean(x)["mean"] == pytest.approx(2.5)
    ms = rangayyan_ch3_sample_mean_squared(x)
    assert ms["mean_square"] == pytest.approx(30.0 / 4)
    # mean square is NOT the variance unless the mean is zero
    assert ms["variance"] == pytest.approx(30.0 / 4 - 2.5**2)
    assert ms["variance"] != pytest.approx(ms["mean_square"])
    centred = rangayyan_ch3_sample_mean_squared(x - 2.5)
    assert centred["variance"] == pytest.approx(centred["mean_square"])
    assert rangayyan_ch3_sample_rms(x)["rms"] == pytest.approx(np.sqrt(7.5))
    sd = rangayyan_ch3_sample_std(x)
    # the book's divisor is N, not N - 1
    assert sd["std"] == pytest.approx(np.std(x))
    assert sd["std_unbiased"] == pytest.approx(np.std(x, ddof=1))
    assert sd["std"] < sd["std_unbiased"]
    with pytest.raises(ValueError):
        rangayyan_ch3_sample_mean([])
    with pytest.raises(ValueError):
        rangayyan_ch3_sample_mean(x, N=99)


def test_moving_average_is_a_true_boxcar_with_the_stated_delay():
    x = np.arange(20, dtype=float)
    out = rangayyan_moving_average(x, M=4)
    # by hand: y[3] = (3+2+1+0)/4 = 1.5
    assert out["y"][3] == pytest.approx(1.5)
    assert out["group_delay"] == pytest.approx(1.5)
    # a constant input passes through unchanged once the window fills
    const = rangayyan_moving_average(np.full(20, 7.0), M=5)
    assert const["y"][10] == pytest.approx(7.0)
    # the M-point boxcar nulls a sinusoid at exactly fs/M
    n = np.arange(400)
    fs, M = 100.0, 10
    tone = np.sin(2 * np.pi * (fs / M) * n / fs)  # 10 Hz, a filter zero
    assert np.abs(rangayyan_moving_average(tone, M=M)["y"][50:]).max() < 1e-9
    with pytest.raises(ValueError):
        rangayyan_moving_average(x, M=0)
    with pytest.raises(ValueError):
        rangayyan_moving_average(x[:3], M=8)


def test_named_boxcars_report_their_delays_and_general_fir_phase():
    x = np.arange(30, dtype=float)
    assert rangayyan_ch3_ma_filter_11pt(x)["group_delay"] == 5.0
    assert rangayyan_ch3_ma_8point(x)["group_delay"] == 3.5  # non-integer
    assert rangayyan_ch3_ma_filter_11pt(x, n=15)["y_at_n"] == pytest.approx(10.0)
    # general FIR: symmetric taps are linear phase, asymmetric are not
    sym = rangayyan_ch3_ma_filter_general(x, [0.25, 0.5, 0.25])
    assert sym["linear_phase"] is True
    assert sym["dc_gain"] == pytest.approx(1.0)
    assert rangayyan_ch3_ma_filter_general(x, [0.1, 0.9])["linear_phase"] is False
    # equal taps reproduce the boxcar exactly
    assert rangayyan_ch3_ma_filter_general(x, np.full(4, 0.25))["y"] == pytest.approx(
        rangayyan_moving_average(x, M=4)["y"]
    )
    with pytest.raises(ValueError):
        rangayyan_ch3_ma_filter_general(x, [])


def test_estimation_errors_agree_across_the_scalar_and_vector_forms():
    rng = np.random.default_rng(0)
    N, p = 200, 3
    X = rng.standard_normal((N, p))
    w = np.array([1.0, -0.5, 0.25])
    d = X @ w + rng.standard_normal(N) * 0.1
    vec = rangayyan_ch3_estimation_error_vector_form(d, w, X)
    scal = rangayyan_ch3_estimation_error(d, X @ w)
    assert vec["error"] == pytest.approx(scal["error"])
    assert vec["mse"] == pytest.approx(scal["mse"])
    assert vec["mse"] < 0.02  # the model is nearly right
    # LMS form with time-invariant weights is the same computation
    lms = rangayyan_ch3_lms_estimation_error(d, w, X)
    assert lms["error"] == pytest.approx(vec["error"])
    assert lms["time_varying"] is False
    # ... and accepts genuinely time-varying weights
    W = np.tile(w, (N, 1))
    assert rangayyan_ch3_lms_estimation_error(d, W, X)["time_varying"] is True
    with pytest.raises(ValueError):
        rangayyan_ch3_estimation_error(d, d[:10])
    with pytest.raises(ValueError):
        rangayyan_ch3_estimation_error_vector_form(d, w[:2], X)


def test_lms_gradient_equals_the_numerical_gradient_of_squared_error():
    rng = np.random.default_rng(1)
    r = rng.standard_normal((50, 2))
    x = rng.standard_normal(50)
    w = np.array([0.3, -0.7])
    e = x - r @ w
    grad = rangayyan_ch3_lms_gradient_estimate(r, e)["gradient"]
    # d/dw [e(n)^2] with e = x - w'r is -2 e(n) r(n); check numerically
    h = 1e-7
    for n in (0, 17, 49):
        num = np.array([
            ((x[n] - (w + h * unit) @ r[n]) ** 2 - (x[n] - w @ r[n]) ** 2) / h
            for unit in np.eye(2)
        ])
        assert grad[n] == pytest.approx(num, abs=1e-4)
    with pytest.raises(ValueError):
        rangayyan_ch3_lms_gradient_estimate(r, e[:10])


def test_rls_solves_the_normal_equations_and_forgets_geometrically():
    rng = np.random.default_rng(2)
    N, p = 300, 2
    r = rng.standard_normal((N, p))
    w_true = np.array([2.0, -1.0])
    x = r @ w_true + rng.standard_normal(N) * 0.05
    phi = rangayyan_ch3_rls_phi_matrix(r, lam=0.99)
    th = rangayyan_ch3_rls_theta_vector(r, x, lam=0.99)
    # the RLS weights recover the truth
    assert th["weights"] == pytest.approx(w_true, abs=0.05)
    # and they really do solve Phi w = Theta
    assert phi["Phi"] @ th["weights"] == pytest.approx(th["Theta"], abs=1e-8)
    assert phi["effective_memory"] == pytest.approx(100.0)  # 1/(1 - 0.99)
    # lambda = 1 means infinite memory (ordinary least squares)
    assert np.isinf(rangayyan_ch3_rls_phi_matrix(r, lam=1.0)["effective_memory"])
    # Phi is symmetric positive semi-definite by construction
    P = phi["Phi"]
    assert P == pytest.approx(P.T)
    assert np.linalg.eigvalsh(P).min() > -1e-9
    with pytest.raises(ValueError):
        rangayyan_ch3_rls_phi_matrix(r, lam=1.5)


def test_heart_rate_conversion_and_the_jensen_gap():
    assert rangayyan_ch4_heart_rate_from_rr(1.0)["heart_rate"] == pytest.approx(60.0)
    assert rangayyan_ch4_heart_rate_from_rr(0.75)["heart_rate"] == pytest.approx(80.0)
    out = rangayyan_ch4_heart_rate_from_rr([0.6, 1.0, 1.4])
    # mean of the rates exceeds the rate of the mean interval (Jensen)
    assert out["mean_instantaneous_hr"] > out["hr_from_mean_rr"]
    assert out["hr_from_mean_rr"] == pytest.approx(60.0)
    with pytest.raises(ValueError):
        rangayyan_ch4_heart_rate_from_rr(0.0)
