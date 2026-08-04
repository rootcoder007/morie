"""Rangayyan batch 1, anchored on the book's equations.

Expected values are computed here from the equation as printed in
Biomedical Signal Analysis (2024) -- not by running the implementation.
Where the placeholder docstring disagreed with the book, the book wins,
and the disagreement is asserted so a revert would fail.
"""

import math

import pytest

from morie.fn.bsaar import aicorder
from morie.fn.bsacep import ar2cep
from morie.fn.bsacorr import bartlettpsd


# ---- eq (7.60): I(P) = log(eps_P) + 2P/Ne ---------------------------

def test_aic_matches_equation_7_60():
    eps = [10.0, 4.0, 3.9, 3.85]
    n, n_eff = 100, 40.0            # Hamming: Ne = 0.4 N
    want = [math.log(e) + 2 * (i + 1) / n_eff for i, e in enumerate(eps)]
    r = aicorder(eps, n)
    assert all(abs(a - b) < 1e-15 for a, b in zip(r["criterion"], want))
    assert r["order"] == want.index(min(want)) + 1
    assert r["n_effective"] == n_eff


def test_aic_is_not_the_textbook_n_log_sigma_form():
    # the placeholder docstring said N*log(sigma^2) + 2p; the book's
    # eq (7.60) is log(eps) + 2P/Ne.  They disagree, and this pins which
    # one is implemented.
    eps = [10.0, 4.0]
    r = aicorder(eps, 100)
    textbook = [100 * math.log(e) + 2 * (i + 1) for i, e in enumerate(eps)]
    assert abs(r["criterion"][0] - textbook[0]) > 1.0


def test_aic_window_controls_effective_sample_size():
    eps = [10.0, 4.0, 3.9]
    assert aicorder(eps, 100, window="rectangular")["n_effective"] == 100.0
    assert aicorder(eps, 100, window=0.25)["n_effective"] == 25.0
    with pytest.raises(ValueError):
        aicorder(eps, 100, window="bartlett-hann-nope")
    with pytest.raises(ValueError):
        aicorder([1.0, -2.0], 100)


# ---- eqs (6.14)-(6.16): Bartlett averaged periodogram ---------------

def test_bartlett_locates_a_pure_tone():
    fs, n, f0 = 64.0, 256, 8.0
    x = [math.sin(2 * math.pi * f0 * t / fs) for t in range(n)]
    b = bartlettpsd(x, fs=fs, n_segments=4)
    peak = b["freqs"][max(range(len(b["psd"])),
                          key=lambda i: b["psd"][i])]
    assert peak == pytest.approx(f0)
    assert b["n_segments"] == 4
    assert b["segment_length"] == 64


def test_bartlett_averages_rather_than_sums():
    # K identical segments must give the SAME psd as one segment:
    # eq (6.16) is a mean, so duplicating data cannot inflate power
    seg = [1.0, -2.0, 3.0, -1.0, 0.5, 2.0, -0.5, 1.5]
    one = bartlettpsd(seg, n_segments=1)["psd"]
    four = bartlettpsd(seg * 4, n_segments=4)["psd"]
    assert all(abs(a - b) < 1e-12 for a, b in zip(one, four))


def test_bartlett_segmentation_is_validated():
    x = [float(i) for i in range(16)]
    with pytest.raises(ValueError):
        bartlettpsd(x)                                  # neither given
    with pytest.raises(ValueError):
        bartlettpsd(x, n_segments=2, segment_length=8)  # both given
    with pytest.raises(ValueError):
        bartlettpsd([1.0])                              # too short


# ---- eq (7.65): AR -> cepstrum recursion ----------------------------

def test_cepstrum_matches_equation_7_65():
    a = [0.5, -0.3, 0.2]
    h1 = -a[0]
    h2 = -a[1] - (1 - 1 / 2) * a[0] * h1
    h3 = -a[2] - ((1 - 1 / 3) * a[0] * h2 + (1 - 2 / 3) * a[1] * h1)
    got = ar2cep(a)["cepstrum"]
    assert all(abs(g - w) < 1e-15 for g, w in zip(got, [h1, h2, h3]))


def test_cepstrum_reindexed_form_is_the_same_recursion():
    # sum (1 - k/n) a_k h(n-k) and sum (j/n) h(j) a_{n-j} are the same
    # sum under j = n - k.  Both are computed here and must agree --
    # a claim that they differ would fail this test.
    a = [0.5, -0.3, 0.2, 0.15]
    got = ar2cep(a)["cepstrum"]
    h = [0.0] * (len(a) + 1)
    for n in range(1, len(a) + 1):
        acc = -a[n - 1]
        for j in range(1, n):
            acc -= (j / n) * h[j] * a[n - j - 1]
        h[n] = acc
    assert all(abs(g - w) < 1e-15 for g, w in zip(got, h[1:]))


def test_cepstrum_first_term_and_gain():
    a = [0.7, 0.1]
    assert ar2cep(a)["cepstrum"][0] == pytest.approx(-0.7)
    assert ar2cep(a, gain=math.e)["c0"] == pytest.approx(1.0)
    assert ar2cep(a)["c0"] is None
    with pytest.raises(ValueError):
        ar2cep(a, gain=0.0)
    with pytest.raises(ValueError):
        ar2cep([])
