"""rnhill is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnhill import rnhill


def test_rnhill_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnhill()
