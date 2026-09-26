"""getrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.getrs import getrs


def test_getrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        getrs()
