"""somol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.somol import somol


def test_somol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        somol()
