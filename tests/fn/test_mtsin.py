"""mtsin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtsin import mtsin


def test_mtsin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtsin()
