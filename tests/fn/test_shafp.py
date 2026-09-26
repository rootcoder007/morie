"""shafp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.shafp import shafp


def test_shafp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shafp()
