"""agiso is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agiso import agiso


def test_agiso_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agiso()
