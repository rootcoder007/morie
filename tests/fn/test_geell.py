"""geell is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geell import geell


def test_geell_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geell()
