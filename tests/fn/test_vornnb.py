"""vornnb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vornnb import vornnb


def test_vornnb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vornnb()
