"""vtgai is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtgai import vtgai


def test_vtgai_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtgai()
