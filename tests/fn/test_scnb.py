"""scnb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.scnb import scnb


def test_scnb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        scnb(y=None, X=None, W=None)
