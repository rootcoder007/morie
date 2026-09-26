"""pskde is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pskde import pskde


def test_pskde_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pskde()
