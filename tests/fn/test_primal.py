"""Tests for primal (Chambolle-Pock primal-dual).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.primal import chambolle_pock, tv_denoise_1d


def test_tv_denoising_removes_a_spike_but_keeps_the_step():
    signal = [0.0] * 20 + [5.0] * 20
    noisy = list(signal)
    noisy[10] += 3.0
    res = tv_denoise_1d(noisy, lam=1.0)
    x = res["x"]
    # the spike is pulled from 3.0 down to about 1.0 while its
    # neighbours stay near 0.1: TV shrinks an isolated jump but does not
    # erase it
    assert x[10] < 1.2
    assert x[10] > x[5]
    assert x[-1] - x[0] > 3.0                # the real step survives


def test_a_large_penalty_flattens_the_signal():
    signal = [0.0] * 10 + [5.0] * 10
    res = tv_denoise_1d(signal, lam=1000.0)
    x = res["x"]
    assert max(x) - min(x) < 0.5
    assert abs(sum(x) / len(x) - sum(signal) / len(signal)) < 0.5


def test_zero_penalty_returns_the_input():
    signal = [1.0, 4.0, 2.0, 8.0]
    res = tv_denoise_1d(signal, lam=0.0)
    for i in range(4):
        assert abs(res["x"][i] - signal[i]) < 1e-6


def test_the_step_size_condition_of_theorem_1_is_reported():
    signal = [0.0, 1.0, 0.0, 1.0]
    res = tv_denoise_1d(signal, lam=0.5)
    # the default steps sit exactly on the boundary: tau = sigma = 0.5
    # and ||K||^2 = 4 for the 1-D difference operator, so the product is
    # 1.0. The check accepts that boundary, and the message now says so.
    assert abs(res["step_condition"] - 1.0) < 1e-9
    assert res["tau"] > 0 and res["sigma"] > 0


def test_chambolle_pock_solves_a_simple_saddle_point():
    # K = I, f*(y) = 0 (prox is 0), g(x) = 1/2||x - b||^2
    b = [3.0, -1.0]

    def prox_g(x, tau):
        return [(x[i] + tau * b[i]) / (1.0 + tau) for i in range(2)]

    res = chambolle_pock(lambda x: list(x), lambda y: list(y),
                         lambda y, s: [0.0, 0.0], prox_g,
                         [0.0, 0.0], [0.0, 0.0], tau=0.5, sigma=0.5)
    assert abs(res["x"][0] - 3.0) < 1e-3
    assert abs(res["x"][1] + 1.0) < 1e-3
    assert res["converged"]


def test_validation():
    for call in (lambda: tv_denoise_1d([1.0]),
                 lambda: tv_denoise_1d([1.0, 2.0], lam=-1.0),
                 lambda: chambolle_pock(lambda x: x, lambda y: y,
                                        lambda y, s: y, lambda x, t: x,
                                        [0.0], [0.0], tau=-1.0,
                                        sigma=0.5),
                 lambda: chambolle_pock(lambda x: x, lambda y: y,
                                        lambda y, s: y, lambda x, t: x,
                                        [0.0], [0.0], tau=10.0,
                                        sigma=10.0, norm_K=1.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
