"""zegin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zegin import gini_spatial


def test_zegin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gini_spatial(data=None)
