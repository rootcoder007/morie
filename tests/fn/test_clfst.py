"""clfst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clfst import clfst


def test_clfst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clfst()
