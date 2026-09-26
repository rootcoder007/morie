"""nmdw2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmdw2 import dwnominate_bridge


def test_nmdw2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dwnominate_bridge(data=None)
