"""iscrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.iscrs import iscrs


def test_iscrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        iscrs()
