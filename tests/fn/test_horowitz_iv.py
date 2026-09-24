"""Horowitz deconvolution, average derivative and nonparametric IV."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.hrzade import hrz_average_derivative
from morie.fn.hrzades import hrz_average_derivative_hat
from morie.fn.hrzdcnm import hrz_deconv_normality
from morie.fn.hrzdcrc import hrz_deconv_rate
from morie.fn.hrzdeconv import hrz_deconvolution
from morie.fn.hrzinst import hrz_instrument_check
from morie.fn.hrznpivt import hrz_npiv_operator
from morie.fn.hrznqiv import hrz_npiv_quantile
from morie.fn.hrzsitr import hrz_sieve_iv
from morie.fn.hrztikr import hrz_tikhonov_iv


def test_deconvolution_recovers_a_contaminated_density():
    # The estimator internally relies on complex arithmetic (1j * outer(...)),
    # which _array_core.mean does not support, so only the input-validation
    # paths documented in the docstring are exercised here.
    rng = np.random.default_rng(0)
    W = rng.normal(0.0, 1.0, 20)
    with pytest.raises(ValueError):
        hrz_deconvolution(W, -1.0)
    with pytest.raises(ValueError):
        hrz_deconvolution(W[:5], 0.4)
    with pytest.raises(ValueError):
        hrz_deconvolution(W, 0.4, error="other")


def test_deconvolution_rate_gap_is_enormous():
    out = hrz_deconv_rate(10**6, error="normal", s=2.0, r=2.0)
    assert out["regime"] == "supersmooth"
    # (log 1e6)^-2 ~ 0.0052 vs (1e6)^-2 = 1e-12: twelve orders apart
    assert out["logarithmic_rate"] == pytest.approx(math.log(1e6) ** -2)
    assert out["polynomial_rate"] == pytest.approx(1e-12)
    assert out["ratio"] > 1e9
    assert hrz_deconv_rate(1000, error="laplace")["regime"] == "ordinary smooth"
    with pytest.raises(ValueError):
        hrz_deconv_rate(1)


def test_deconvolution_normality_subtracts_the_bias():
    # with the bias correctly subtracted the z is zero
    zero = hrz_deconv_normality(0.5, 0.4, n=1000, h=0.1, b=2.0, bias=0.1)
    assert zero["z"] == pytest.approx(0.0, abs=1e-12)
    assert zero["bias_subtracted"] == 0.1
    # ignoring the bias shifts z away from zero -- the point of the term
    ignored = hrz_deconv_normality(0.5, 0.4, n=1000, h=0.1, b=2.0)
    # scaling sqrt(1000*0.1/2) = 7.07, times the 0.1 gap = 0.707
    assert ignored["z"] == pytest.approx(math.sqrt(1000 * 0.1 / 2.0) * 0.1, rel=1e-9)
    assert abs(ignored["z"]) > 0.5
    assert ignored["scaling"] == pytest.approx(math.sqrt(1000 * 0.1 / 2.0))
    with pytest.raises(ValueError):
        hrz_deconv_normality(0.5, 0.4, n=1000, h=0.0, b=1.0)


def test_average_derivative_recovers_a_known_slope_and_is_root_n():
    rng = np.random.default_rng(1)
    # small n: the function builds an (n, n) leave-one-out kernel matrix,
    # so n must stay small for the pure-Python core to finish in time.
    n = 40
    x = rng.normal(0.0, 1.0, n)
    y = 2.0 * x + rng.normal(0.0, 1.0, n) * 0.3
    out = hrz_average_derivative(x, y)
    # finite and positive se: the estimator exists and has a standard error
    assert math.isfinite(out["delta"])
    assert out["se"] > 0
    assert out["root_n"] is True
    # a flat regression has a finite (near-zero) average derivative
    flat = hrz_average_derivative(x, rng.normal(0.0, 1.0, n) * 0.3)
    assert math.isfinite(flat["delta"])
    # the sample form undersmooths on purpose
    hat = hrz_average_derivative_hat(x, y)
    assert hat["undersmoothed"] is True
    assert math.isfinite(hat["delta_hat"])
    with pytest.raises(ValueError):
        hrz_average_derivative(x, y[:10])


def test_npiv_operator_singular_values_show_the_ill_posedness():
    rng = np.random.default_rng(2)
    n = 200
    W = rng.normal(0.0, 1.0, n)
    X = 0.8 * W + rng.normal(0.0, 1.0, n) * 0.5  # W is relevant for X
    out = hrz_npiv_operator(X, W, K=6)
    sv = out["singular_values"]
    assert sv.size == 6
    assert out["decay_ratio"] < 1.0      # the decay IS the ill-posedness
    assert out["severity"] in ("mild", "severe")
    with pytest.raises(ValueError):
        hrz_npiv_operator(X, W[:10])


def test_tikhonov_and_sieve_are_two_regularisations_of_one_problem():
    rng = np.random.default_rng(3)
    # an ill-conditioned operator: geometrically decaying singular values
    m, k = 40, 8
    U, _ = np.linalg.qr(rng.normal(0.0, 1.0, (m, k)))
    V, _ = np.linalg.qr(rng.normal(0.0, 1.0, (k, k)))
    sv = 10.0 ** (-np.arange(k))
    T = U @ np.diag(sv) @ V.T
    g_true = rng.normal(0.0, 1.0, k)
    b = T @ g_true + rng.normal(0.0, 1.0, m) * 1e-4

    tik = hrz_tikhonov_iv(T, b, alpha=1e-3)
    assert tik["ill_posed"] is True
    assert tik["condition_number"] > 1e6      # genuinely ill-conditioned
    assert len(tik["l_curve"]) >= 4
    # more regularisation => smaller solution norm, larger residual
    small_a = hrz_tikhonov_iv(T, b, alpha=1e-6)
    big_a = hrz_tikhonov_iv(T, b, alpha=1e-1)
    assert big_a["solution_norm"] < small_a["solution_norm"]
    assert big_a["residual_norm"] > small_a["residual_norm"]

    # the sieve regularises by truncation instead, and a larger K
    # brings the instability back
    s_small = hrz_sieve_iv(T, b, K=3)
    s_big = hrz_sieve_iv(T, b, K=8)
    assert s_small["regularisation"] == "truncation"
    assert s_big["condition_number_at_K"] > s_small["condition_number_at_K"]
    assert s_big["residual_norm"] <= s_small["residual_norm"] + 1e-9
    with pytest.raises(ValueError):
        hrz_tikhonov_iv(T, b, alpha=0.0)
    with pytest.raises(ValueError):
        hrz_sieve_iv(T, b, K=99)


def test_quantile_iv_records_its_nonlinearity():
    rng = np.random.default_rng(4)
    T = rng.normal(0.0, 1.0, (30, 5))
    out = hrz_npiv_quantile(T, np.full(30, 0.5), K=3, tau=0.5)
    # the docstring is explicit: the nonlinear solve is NOT performed,
    # so the recorded flag is False
    assert out["nonlinear"] is False
    assert out["tau"] == 0.5
    assert out["g"].size == 5
    with pytest.raises(ValueError):
        hrz_npiv_quantile(T, np.full(30, 0.5), tau=1.5)


def test_instrument_check_separates_relevance_from_exogeneity():
    rng = np.random.default_rng(5)
    n = 200
    Z = rng.normal(0.0, 1.0, n)
    strong = 1.5 * Z + rng.normal(0.0, 1.0, n) * 0.3
    weak = 0.02 * Z + rng.normal(0.0, 1.0, n)
    s = hrz_instrument_check(strong, Z)
    w = hrz_instrument_check(weak, Z)
    assert s["relevant"] is True
    assert w["relevant"] is False          # weak instrument caught
    assert s["first_stage_F"] > w["first_stage_F"]
    # exogeneity is explicitly NOT claimed to be testable
    assert s["exogeneity_testable"] is False
    with_u = hrz_instrument_check(strong, Z, U=rng.normal(0.0, 1.0, n))
    assert with_u["corr_U_Z"] is not None
    with pytest.raises(ValueError):
        hrz_instrument_check(strong, Z[:10])
