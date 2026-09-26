"""gelutn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gelutn import gelu_tanh_approx


def test_gelutn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gelu_tanh_approx(y=None)
