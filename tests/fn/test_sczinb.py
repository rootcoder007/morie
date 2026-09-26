"""sczinb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sczinb import sczinb


def test_sczinb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sczinb(y=None, X=None, W=None)
