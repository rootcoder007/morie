"""abufc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abufc import abufc


def test_abufc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abufc()
