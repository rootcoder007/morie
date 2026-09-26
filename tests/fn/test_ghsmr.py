"""ghsmr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghsmr import ghsmr


def test_ghsmr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghsmr()
