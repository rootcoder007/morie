"""vtpai is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtpai import vtpai


def test_vtpai_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtpai()
