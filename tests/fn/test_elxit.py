"""elxit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elxit import elxit


def test_elxit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elxit()
