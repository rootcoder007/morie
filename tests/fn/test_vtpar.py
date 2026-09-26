"""vtpar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtpar import vtpar


def test_vtpar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtpar()
