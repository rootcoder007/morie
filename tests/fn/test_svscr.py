"""svscr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svscr import scree_spatial


def test_svscr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        scree_spatial(data=None)
