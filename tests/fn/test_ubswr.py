"""ubswr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubswr import ubswr


def test_ubswr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubswr()
