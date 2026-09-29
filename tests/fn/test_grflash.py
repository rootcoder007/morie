"""Out of chaos, comes order. — Friedrich Nietzsche"""

import doctest as _doctest

import morie.fn.grflash as _doctest_module
from morie.fn.grflash import geron_flash_attention_tile


def test_grflash_basic():
    """Test basic functionality."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0]] * 8
    V = [[1.0, 0.0]] * 8
    result = geron_flash_attention_tile(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grflash_edge():
    """Test edge cases."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0]] * 8
    V = [[1.0, 0.0]] * 8
    result = geron_flash_attention_tile(Q, K, V)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False, optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS
    )
    assert res.attempted > 0
    assert res.failed == 0


def test_tiled_output_equals_direct_softmax_attention():
    import math

    import pytest

    Q = [[0.3, -0.2], [1.0, 0.5]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-0.5, 0.2], [0.3, 0.3]]
    V = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [0.5, 0.5], [2.0, -1.0]]
    out = []
    for q in Q:
        s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(2) for k in K]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = sum(e)
        out.append([sum(e[i] * V[i][c] for i in range(5)) / z for c in range(2)])
    r = geron_flash_attention_tile(Q, K, V, block_size=2)
    for i in range(2):
        assert r["output"][i] == pytest.approx(out[i], rel=1e-13)
    assert r["n_blocks"] == 3
