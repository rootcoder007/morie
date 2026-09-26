"""mtred is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtred import mtred


def test_mtred_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtred()
