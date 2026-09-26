"""dkgak is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkgak import dkgak


def test_dkgak_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkgak()
