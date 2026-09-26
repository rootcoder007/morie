"""plafp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plafp import plafp


def test_plafp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plafp()
