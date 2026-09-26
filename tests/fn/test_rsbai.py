"""rsbai is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsbai import rsbai


def test_rsbai_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsbai()
