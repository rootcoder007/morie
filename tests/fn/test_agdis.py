"""agdis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agdis import agdis


def test_agdis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agdis()
