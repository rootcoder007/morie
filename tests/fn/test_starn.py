"""Tests for morie.fn.starn — Spatio-temporal autoregressive model."""

from morie.fn import _array_core as np
import pytest

from morie.fn.starn import starn


class TestStarn:
    def test_output_keys(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))
        W = np.eye(5) * 0 + 0.2
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)
        result = starn(data, W)
        assert result["coefficients"] is not None
        assert result["fitted"].shape == (9, 5)

    def test_residual_shape(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 4))
        W = np.ones((4, 4)) * 0.25
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)
        result = starn(data, W, order=2)
        assert result["residuals"].shape == (7, 4)

    def test_weight_shape_mismatch(self):
        with pytest.raises(ValueError, match="weights must be"):
            starn(np.ones((5, 3)), np.eye(4))


def _ring_w(n):
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        w[i][(i + 1) % n] = w[i][(i - 1) % n] = 0.5
    return w


def _star_series(phi0, phi1, T, n, noise):
    import math
    w = _ring_w(n)
    z = [[math.sin(1.7 * j + 0.3) for j in range(n)]]
    for t in range(1, T):
        prev = z[-1]
        lag = [sum(w[i][j] * prev[j] for j in range(n)) for i in range(n)]
        z.append([phi0 * prev[i] + phi1 * lag[i]
                  + noise * math.cos(2.3 * t * (i + 1)) for i in range(n)])
    return z, w


class TestStarnValues:
    def test_recovers_noise_free_star(self):
        """z_t = 0.5 z_{t-1} + 0.3 W z_{t-1} exactly: both terms, including
        the own lag, come back and the residual variance is round-off."""
        z, w = _star_series(0.5, 0.3, 12, 6, 0.0)
        r = starn(z, w, order=1)
        c = [float(v) for v in r["coefficients"]]
        assert c[0] == pytest.approx(0.5, abs=1e-12)
        assert c[1] == pytest.approx(0.3, abs=1e-12)
        assert r["sigma2"] < 1e-28

    def test_least_squares_and_aic(self):
        """Coefficients solve the normal equations; AIC = N log(RSS/N) + 2k."""
        import math
        z, w = _star_series(0.6, -0.2, 15, 5, 0.1)
        r = starn(z, w, order=1)
        x0, x1, yv = [], [], []
        for t in range(1, 15):
            prev = z[t - 1]
            for i in range(5):
                x0.append(prev[i])
                x1.append(sum(w[i][j] * prev[j] for j in range(5)))
                yv.append(z[t][i])
        a, b, d = sum(v * v for v in x0), sum(p * q for p, q in zip(x0, x1)), sum(v * v for v in x1)
        e, f = sum(p * q for p, q in zip(x0, yv)), sum(p * q for p, q in zip(x1, yv))
        det = a * d - b * b
        want = [(d * e - b * f) / det, (a * f - b * e) / det]
        c = [float(v) for v in r["coefficients"]]
        for got, ref in zip(c, want):
            assert got == pytest.approx(ref, rel=1e-10)
        rss = sum((yv[k] - want[0] * x0[k] - want[1] * x1[k]) ** 2 for k in range(len(yv)))
        assert r["aic"] == pytest.approx(len(yv) * math.log(rss / len(yv)) + 4, rel=1e-10)
