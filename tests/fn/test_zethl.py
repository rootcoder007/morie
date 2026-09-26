"""zethl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zethl import theil_spatial


def test_zethl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        theil_spatial(data=None)
