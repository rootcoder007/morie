"""The native RNG: Philox4x32-10 and Wichura's AS 241.

Anchored on PUBLISHED numbers, not on whatever the implementation happens to
emit: the Random123 known-answer vectors (Salmon, Moraes, Dror & Shaw, SC'11,
`tests/kat_vectors` of the reference distribution) for the bijection, and the
R arm's own output for the stream, so both languages are pinned to the same
doubles rather than to the same distribution.
"""

import pytest

from morie.fn._rng import (normal_quantile, philox4x32, random_normal,
                           random_uniform)

# Random123 kat_vectors, the three `philox4x32 10` lines, verbatim.
PHILOX_KAT = [
    ((0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000),
     (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF),
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("ctr,key,want", PHILOX_KAT)
def test_philox_matches_the_published_known_answer_tests(ctr, key, want):
    assert philox4x32(ctr, key) == want
    assert philox4x32([ctr], key) == [want]      # (n, 4) form, same answer


def test_philox_is_a_bijection_of_the_counter():
    """Distinct counters, distinct blocks -- the property the whole
    counter-based construction rests on."""
    key = (0xA4093822, 0x299F31D0)
    seen = {philox4x32((j, 0, 0, 0), key) for j in range(500)}
    assert len(seen) == 500


def test_stream_is_bit_identical_to_the_r_arm():
    """`.morie_random_uniform(7, seed = 12345, stream = 3)` and its normal
    twin in R/aaa_rng_native.R print exactly these doubles."""
    u = [float(v) for v in random_uniform(7, seed=12345, stream=3)]
    z = [float(v) for v in random_normal(7, seed=12345, stream=3)]
    assert u == [0.82723027456086129, 0.49637839302886277, 0.11695582850370556,
                 0.56882696587126702, 0.24151097459252924, 0.67883205891121179,
                 0.36555732542183250]
    assert z == [0.94327658191243779, -0.0090781471244313419, -1.1903428714337425,
                 0.17338849442190796, -0.70145044652586208, 0.46443533063516002,
                 -0.34364317514143328]


def test_uniforms_stay_inside_the_open_unit_interval():
    u = [float(v) for v in random_uniform(50000, seed=7)]
    assert 0.0 < min(u) and max(u) < 1.0
    assert abs(sum(u) / len(u) - 0.5) < 0.01


def test_same_seed_reproduces_and_different_seeds_diverge():
    a = list(random_uniform(200, seed=1, stream=0))
    assert a == list(random_uniform(200, seed=1, stream=0))
    assert a != list(random_uniform(200, seed=2, stream=0))
    assert a != list(random_uniform(200, seed=1, stream=1))
    # counter-based: a longer draw contains the shorter one as a prefix.
    assert list(random_uniform(500, seed=1))[:200] == a


def test_normals_have_the_right_first_two_moments():
    z = [float(v) for v in random_normal(50000, seed=42)]
    mean = sum(z) / len(z)
    sd = (sum((v - mean) ** 2 for v in z) / (len(z) - 1)) ** 0.5
    assert abs(mean) < 0.02
    assert abs(sd - 1.0) < 0.02


def test_as241_matches_published_normal_quantiles():
    """Inverse CDF, not Box-Muller -- the same transform R's qnorm uses, so
    one uniform makes one normal and the two arms stay in step."""
    for p, want in {0.975: 1.959963984540054, 0.5: 0.0,
                    0.001: -3.090232306167813, 0.99: 2.3263478740408408}.items():
        assert float(normal_quantile([p])[0]) == pytest.approx(want, abs=1e-13)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        random_uniform(-1)
    with pytest.raises(ValueError):
        normal_quantile([0.0])
    with pytest.raises(ValueError):
        normal_quantile([1.0])
