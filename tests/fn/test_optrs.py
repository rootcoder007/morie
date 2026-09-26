"""optrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.optrs import optrs


def test_optrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        optrs()
