"""diaspath is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.diaspath import diameter


def test_diaspath_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        diameter(G=None)
