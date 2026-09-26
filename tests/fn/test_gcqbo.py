"""gcqbo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcqbo import gcqbo


def test_gcqbo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcqbo()
