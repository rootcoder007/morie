"""mapho is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mapho import mapho


def test_mapho_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mapho()
