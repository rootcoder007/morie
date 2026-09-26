"""csprs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csprs import csprs


def test_csprs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csprs()
