"""hullrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hullrn import hullrn


def test_hullrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hullrn()
