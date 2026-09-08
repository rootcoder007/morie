"""Tests for tqpack (bit-packing of quantiser indices).

Replaces the generated stub, which imported
``turboquant_bit_pack_indices``.
"""

from morie.fn.tqpack import pack_indices, unpack_indices


def test_pack_then_unpack_is_the_identity():
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 7, 0, 3]
    res = pack_indices(idx, 3)
    assert unpack_indices(res["bytes"], 3, len(idx))["indices"] == idx


def test_the_byte_count_is_the_ceiling_of_the_bit_count():
    idx = [1] * 11
    res = pack_indices(idx, 3)
    assert res["bits_used"] == 33
    assert res["n_bytes"] == 5                 # ceil(33 / 8)
    assert res["padding_bits"] == 7


def test_eight_bit_indices_are_one_byte_each():
    idx = [0, 255, 128, 7]
    res = pack_indices(idx, 8)
    assert res["n_bytes"] == 4
    assert res["padding_bits"] == 0
    assert unpack_indices(res["bytes"], 8, 4)["indices"] == idx


def test_the_compression_ratio_is_reported_against_float64():
    res = pack_indices([1] * 64, 4)
    assert abs(res["compression_vs_float64"] - 64.0 / 4.0) < 1e-9


def test_a_single_bit_packs_eight_values_to_a_byte():
    idx = [1, 0, 1, 1, 0, 0, 1, 0]
    res = pack_indices(idx, 1)
    assert res["n_bytes"] == 1
    assert unpack_indices(res["bytes"], 1, 8)["indices"] == idx


def test_validation():
    for call in (lambda: pack_indices([0], 0),
                 lambda: pack_indices([0], 33),
                 lambda: pack_indices([1.5], 4),
                 lambda: pack_indices([16], 4),
                 lambda: unpack_indices([0], 0, 1)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
