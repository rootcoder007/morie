"""Tests for alfomg.openfold_msa_pair."""

from morie.fn import _array_core as np

from morie.fn.alfomg import openfold_msa_pair


def test_alfomg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # msa is indexed [sequence][position][channel]
    msa = [[[rng.normal() for _ in range(3)] for _ in range(4)] for _ in range(2)]
    # pair is indexed [i][j][channel] and must be square over positions
    pair = [[[rng.normal() for _ in range(2)] for _ in range(4)] for _ in range(4)]
    result = openfold_msa_pair(msa, pair)
    assert isinstance(result, dict)
    # The function returns a RichResult with these documented keys.
    assert "opm" in result
    assert "bias" in result
    assert "attn" in result
    assert "msa_out" in result
    assert "pair_out" in result
    assert "pair_updated" in result
    assert "self_attention" in result
    assert "n_seq" in result
    assert "n_pos" in result
    assert "n_channel" in result
    assert "n_pair_channel" in result
    assert "scale" in result
    assert "gated" in result
    assert "method" in result


def test_alfomg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # Smallest non-trivial case: 1 sequence, 1 position, 1 channel.
    msa = [[[rng.normal()]]]
    pair = [[[rng.normal() for _ in range(2)]]]
    result = openfold_msa_pair(msa, pair)
    assert isinstance(result, dict)
    # Without w_opm the pair representation must be returned unchanged
    # and pair_updated must be False (per docstring).
    assert result["pair_updated"] is False
    assert result["pair_out"] == pair
    assert result["n_seq"] == 1
    assert result["n_pos"] == 1
    assert result["n_channel"] == 1
    assert result["n_pair_channel"] == 2
    # Default scale is 1/sqrt(c) per the docstring.
    assert abs(result["scale"] - 1.0) < 1e-12
    # Unused gate means ungated.
    assert result["gated"] is False

    # Also exercise w_opm supplied as a [n_pair_channel][c*c] matrix.
    s, r, c, cz = 1, 1, 1, 2
    w_opm = [[0.5 for _ in range(c * c)] for _ in range(cz)]
    result2 = openfold_msa_pair(msa, pair, w_opm=w_opm)
    assert result2["pair_updated"] is True
    # self_attention must be a finite scalar in [0, 1].
    assert 0.0 <= result2["self_attention"] <= 1.0
