"""agprx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agprx import agprx


def test_agprx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agprx()
