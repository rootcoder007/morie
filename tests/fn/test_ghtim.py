"""ghtim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghtim import ghtim


def test_ghtim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghtim()
