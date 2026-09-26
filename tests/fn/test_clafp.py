"""clafp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clafp import clafp


def test_clafp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clafp()
