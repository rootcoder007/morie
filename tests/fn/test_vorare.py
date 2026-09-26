"""vorare is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vorare import vorare


def test_vorare_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vorare()
