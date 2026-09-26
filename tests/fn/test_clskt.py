"""clskt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clskt import clskt


def test_clskt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clskt()
