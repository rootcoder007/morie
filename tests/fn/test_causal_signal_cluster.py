"""Causal signal/NN cluster: rng032, rng033, rng036, rng037, rng049,
rng053, rng103, rng196, rgztf, hmc1d, grc1d, kmclm, kmprf, nchunk."""

from morie.fn import _array_core as np
import pytest

from morie.fn.grc1d import geron_causal_1d_cnn
from morie.fn.hmc1d import geron_causal_1d_conv
from morie.fn.kmclm import kamath_causal_lm_loss
from morie.fn.kmprf import kamath_prefix_lm_mask
from morie.fn.nchunk import causal_chunked_attention
from morie.fn.rgztf import rangayyan_z_transform
from morie.fn.rng032 import rangayyan_ch3_causal_convolution
from morie.fn.rng033 import rangayyan_ch3_causal_convolution_alt
from morie.fn.rng036 import rangayyan_ch3_discrete_convolution_causal
from morie.fn.rng037 import rangayyan_ch3_discrete_convolution_causal_alt
from morie.fn.rng049 import rangayyan_ch3_laplace_transform_causal_finite
from morie.fn.rng053 import rangayyan_ch3_z_transform_fir
from morie.fn.rng103 import rangayyan_ch3_integral_causal
from morie.fn.rng196 import rangayyan_ch4_dicrotic_notch_second_derivative


def test_rng036_hand_and_commutativity():
    x = [1.0, 2.0, 3.0]
    h = [1.0, 1.0]
    out = rangayyan_ch3_discrete_convolution_causal(x, h)
    assert out["y"] == pytest.approx([1.0, 3.0, 5.0, 3.0])
    assert out["value"] is None
    assert rangayyan_ch3_discrete_convolution_causal(x, h, n=2)["value"] == pytest.approx(5.0)
    alt = rangayyan_ch3_discrete_convolution_causal_alt(x, h)
    assert alt["y"] == pytest.approx(out["y"])  # commutativity
    with pytest.raises(ValueError):
        rangayyan_ch3_discrete_convolution_causal(x, h, n=9)


def test_rng032_matches_analytic_and_commutes():
    # x(t) = 1, h(t) = 1 on [0, T] -> y(t) = t
    t = np.linspace(0, 5, 501)
    one = np.ones_like(t)
    out = rangayyan_ch3_causal_convolution(one, one, dt=t[1] - t[0])
    assert out["y"] == pytest.approx(t, abs=1e-9)
    alt = rangayyan_ch3_causal_convolution_alt(one, one, dt=t[1] - t[0])
    assert alt["y"] == pytest.approx(out["y"])
    # e^{-t} * e^{-t} = t e^{-t}
    e = np.exp(-t)
    y = rangayyan_ch3_causal_convolution(e, e, dt=t[1] - t[0])["y"]
    assert y == pytest.approx(t * np.exp(-t), abs=2e-4)


def test_rng103_running_integral():
    t = np.linspace(0, 3, 601)
    out = rangayyan_ch3_integral_causal(t, dt=t[1] - t[0])
    assert out["y"] == pytest.approx(t**2 / 2, abs=1e-9)
    assert out["y"][0] == 0.0
    assert out["total"] == pytest.approx(4.5, abs=1e-6)


def test_rng049_laplace_of_unit_pulse():
    # h = 1 on [0, T] -> H(s) = (1 - e^{-sT}) / s
    T, dt = 2.0, 1e-3
    h = np.ones(int(T / dt) + 1)
    for s in (0.5, 2.0):
        H = rangayyan_ch3_laplace_transform_causal_finite(h, s, dt=dt)["H"]
        assert H.real == pytest.approx((1 - np.exp(-s * T)) / s, abs=1e-6)
    # s = jw recovers the Fourier transform of the pulse
    w = 3.0
    Hj = rangayyan_ch3_laplace_transform_causal_finite(h, 1j * w, dt=dt)["H"]
    assert Hj == pytest.approx((1 - np.exp(-1j * w * T)) / (1j * w), abs=1e-6)


def test_rng053_and_rgztf():
    h = [1.0, -0.5]
    out = rangayyan_ch3_z_transform_fir(h, 2.0)
    assert out["H"].real == pytest.approx(1 - 0.5 / 2)
    # frequency response at z = 1 is the tap sum; at z = -1 the alternating sum
    assert rangayyan_ch3_z_transform_fir(h, 1.0)["H"].real == pytest.approx(0.5)
    assert rangayyan_ch3_z_transform_fir(h, -1.0)["H"].real == pytest.approx(1.5)
    with pytest.raises(ValueError):
        rangayyan_ch3_z_transform_fir(h, 0.0)
    z = rangayyan_z_transform(h, z=2.0)
    assert z["degree"] == 1
    assert z["H"] == pytest.approx(out["H"])
    assert rangayyan_z_transform(h)["H"] is None


def test_rng196_book_equation():
    # Eq. (4.22), Rangayyan p. 228; coefficients sum to zero so a
    # constant (and, being a 2nd-derivative estimate, a ramp) map to 0.
    out = rangayyan_ch4_dicrotic_notch_second_derivative(np.ones(9))
    assert out["p"][out["valid"]] == pytest.approx(np.zeros(5), abs=1e-12)
    ramp = np.arange(9.0)
    assert rangayyan_ch4_dicrotic_notch_second_derivative(ramp)["p"][2:7] == pytest.approx(
        np.zeros(5), abs=1e-12
    )
    # a single unit spike reproduces the reversed tap sequence
    spike = np.zeros(9)
    spike[4] = 1.0
    p = rangayyan_ch4_dicrotic_notch_second_derivative(spike)["p"]
    assert p[2:7] == pytest.approx([2.0, -1.0, -2.0, -1.0, 2.0])
    # causal mode is the same sequence delayed by two samples
    pc = rangayyan_ch4_dicrotic_notch_second_derivative(spike, causal=True)["p"]
    assert pc[4:9] == pytest.approx(p[2:7])
    with pytest.raises(ValueError):
        rangayyan_ch4_dicrotic_notch_second_derivative([1.0, 2.0])


def test_hmc1d_causality_and_hand_values():
    x = [1.0, 2.0, 3.0, 4.0]
    out = geron_causal_1d_conv(x, [1.0, 0.5])  # y_t = x_t + 0.5 x_{t-1}
    assert out["y"] == pytest.approx([1.0, 2.5, 4.0, 5.5])
    # no future leakage: perturbing x_3 leaves y_0..y_2 untouched
    x2 = [1.0, 2.0, 3.0, 99.0]
    y2 = geron_causal_1d_conv(x2, [1.0, 0.5])["y"]
    assert y2[:3] == pytest.approx(out["y"][:3])
    with pytest.raises(ValueError):
        geron_causal_1d_conv([1.0], [1.0, 1.0])


def test_grc1d_dilation_and_strict():
    x = np.arange(1.0, 9.0)
    plain = geron_causal_1d_cnn(x, [1.0, 1.0], dilation=1)
    assert plain["y"] == pytest.approx(geron_causal_1d_conv(x, [1.0, 1.0])["y"])
    dil = geron_causal_1d_cnn(x, [1.0, 1.0], dilation=2)  # y_t = x_t + x_{t-2}
    assert dil["y"][2:] == pytest.approx(x[2:] + x[:-2])
    assert dil["receptive_field"] == 3
    strict = geron_causal_1d_cnn(x, [1.0], strict=True)  # y_t = x_{t-1}
    assert strict["y"][1:] == pytest.approx(x[:-1])
    assert strict["y"][0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        geron_causal_1d_cnn(x, [1.0, 1.0], dilation=0)


def test_kmclm_uniform_baseline_and_perfect_model():
    V, T = 8, 5
    uniform = np.zeros((T, V))
    out = kamath_causal_lm_loss(uniform, np.arange(T) % V)
    assert out["loss"] == pytest.approx(np.log(V))
    assert out["perplexity"] == pytest.approx(V)
    # near-deterministic model: loss goes to ~0
    logits = np.full((T, V), -50.0)
    tgt = np.arange(T) % V
    logits[np.arange(T), tgt] = 50.0
    assert kamath_causal_lm_loss(logits, tgt)["loss"] == pytest.approx(0.0, abs=1e-9)
    # padding is skipped
    tgt2 = tgt.copy()
    tgt2[:2] = -100
    assert kamath_causal_lm_loss(uniform, tgt2)["n_tokens"] == T - 2
    with pytest.raises(ValueError):
        kamath_causal_lm_loss(uniform, np.full(T, V))  # id out of range


def test_kmprf_mask_shapes():
    m = kamath_prefix_lm_mask(2, 4)["mask"]
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(m, expected)
    assert np.array_equal(kamath_prefix_lm_mask(0, 4)["mask"], np.tril(np.ones((4, 4), bool)))
    assert kamath_prefix_lm_mask(4, 4)["mask"].all()
    assert np.isneginf(kamath_prefix_lm_mask(2, 4)["additive"][0, 2])
    with pytest.raises(ValueError):
        kamath_prefix_lm_mask(5, 4)


def test_nchunk_matches_full_attention_and_sparsifies():
    rng = np.random.default_rng(0)
    L, d = 12, 4
    Q, K, V = rng.normal(size=(L, d)), rng.normal(size=(L, d)), rng.normal(size=(L, 3))
    full = causal_chunked_attention(Q, K, V, chunk_size=4)  # all chunks visible
    # reference: plain causal softmax attention
    s = Q @ K.T / np.sqrt(d)
    s = np.where(np.tril(np.ones((L, L), bool)), s, -np.inf)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    ref = (e / e.sum(axis=1, keepdims=True)) @ V
    assert full["output"] == pytest.approx(ref)
    assert full["density"] == pytest.approx(np.tril(np.ones((L, L))).mean())
    local = causal_chunked_attention(Q, K, V, chunk_size=4, n_chunks_back=0)
    assert local["density"] < full["density"]
    assert not local["mask"][8, 3]  # chunk 2 cannot see chunk 0
    assert local["attention"].sum(axis=1) == pytest.approx(np.ones(L))
    with pytest.raises(ValueError):
        causal_chunked_attention(Q, K, V, chunk_size=0)
