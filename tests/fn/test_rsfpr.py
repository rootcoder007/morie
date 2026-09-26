"""rsfpr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsfpr import rsfpr


def test_rsfpr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsfpr()
