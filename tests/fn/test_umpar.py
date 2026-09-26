"""umpar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umpar import umpar


def test_umpar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umpar()
