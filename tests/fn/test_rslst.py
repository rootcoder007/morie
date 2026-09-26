"""rslst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rslst import rslst


def test_rslst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rslst()
