"""kgikp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgikp import ik_probability


def test_kgikp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ik_probability(data=None)
