"""fotrew is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fotrew import fotrew


def test_fotrew_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fotrew()
