"""gahill is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gahill import gahill


def test_gahill_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gahill()
