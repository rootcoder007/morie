"""enfrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enfrg import enfrg


def test_enfrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enfrg()
