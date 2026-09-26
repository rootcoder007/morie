"""csfrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csfrg import csfrg


def test_csfrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csfrg()
