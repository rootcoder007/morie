"""sprtobt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sprtobt import sprtobt


def test_sprtobt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sprtobt(y=None, X=None, W=None)
