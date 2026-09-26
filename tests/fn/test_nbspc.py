"""nbspc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbspc import nbspc


def test_nbspc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbspc()
