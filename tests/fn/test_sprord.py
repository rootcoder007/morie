"""sprord is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sprord import sprord


def test_sprord_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sprord(y=None, X=None, W=None)
