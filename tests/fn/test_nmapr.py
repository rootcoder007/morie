"""nmapr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmapr import apre_stat


def test_nmapr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        apre_stat(data=None)
