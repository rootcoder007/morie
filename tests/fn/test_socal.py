"""socal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.socal import socal


def test_socal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        socal()
