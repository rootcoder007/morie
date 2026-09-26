"""mtres is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtres import mtres


def test_mtres_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtres()
