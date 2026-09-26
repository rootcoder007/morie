"""afevi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afevi import afevi


def test_afevi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afevi()
