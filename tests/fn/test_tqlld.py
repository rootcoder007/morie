"""Tests for tqlld (Lloyd-Max scalar quantiser).

Replaces the generated stub, which imported
``turboquant_lloyd_max_codebook``.
"""

from morie.fn.tqlld import lloyd_max_codebook, quantize_with_codebook


def test_one_level_is_the_mean_of_the_source():
    res = lloyd_max_codebook(levels=1, source="gaussian")
    assert abs(res["codebook"][0]) < 1e-6


def test_the_two_level_gaussian_codebook_is_the_printed_pair():
    # the classical Lloyd-Max solution for a standard normal is
    # +/- sqrt(2/pi) = 0.7979, with distortion 1 - 2/pi = 0.3634
    res = lloyd_max_codebook(levels=2, source="gaussian")
    cb = sorted(res["codebook"])
    assert abs(cb[0] + 0.7979) < 2e-3
    assert abs(cb[1] - 0.7979) < 2e-3
    assert abs(res["distortion"] - 0.3634) < 2e-3


def test_distortion_falls_as_levels_rise():
    d = [lloyd_max_codebook(levels=k, source="gaussian")["distortion"]
         for k in (1, 2, 4, 8)]
    assert all(d[i + 1] < d[i] for i in range(len(d) - 1))


def test_the_iteration_never_increases_distortion():
    res = lloyd_max_codebook(levels=4, source="gaussian")
    hist = res["distortion_history"]
    assert all(hist[i + 1] <= hist[i] + 1e-12
               for i in range(len(hist) - 1))
    assert res["converged"]


def test_boundaries_are_the_midpoints_of_neighbouring_codewords():
    res = lloyd_max_codebook(levels=4, source="uniform", lo=-1.0, hi=1.0)
    cb = sorted(res["codebook"])
    for i in range(len(cb) - 1):
        assert abs(res["boundaries"][i] - 0.5 * (cb[i] + cb[i + 1])) < 1e-6


def test_a_uniform_source_gives_an_evenly_spaced_codebook():
    res = lloyd_max_codebook(levels=4, source="uniform", lo=0.0, hi=1.0)
    cb = sorted(res["codebook"])
    gaps = [cb[i + 1] - cb[i] for i in range(3)]
    assert max(gaps) - min(gaps) < 1e-3


def test_quantising_maps_each_sample_to_its_nearest_codeword():
    cb = [-1.0, 0.0, 2.0]
    res = quantize_with_codebook([-0.9, 0.4, 1.6, 2.5], cb)
    assert res["indices"] == [0, 1, 2, 2]
    assert res["values"] == [-1.0, 0.0, 2.0, 2.0]


def test_an_empirical_source_uses_the_data():
    data = [0.0] * 20 + [10.0] * 20
    res = lloyd_max_codebook(levels=2, source="empirical", data=data)
    cb = sorted(res["codebook"])
    assert abs(cb[0]) < 0.2 and abs(cb[1] - 10.0) < 0.2


def test_validation():
    for call in (lambda: lloyd_max_codebook(levels=0),
                 lambda: lloyd_max_codebook(source="cauchy"),
                 lambda: lloyd_max_codebook(source="uniform", lo=1.0,
                                            hi=0.0),
                 lambda: lloyd_max_codebook(source="empirical")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
