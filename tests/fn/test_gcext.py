"""gcext is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcext import gcext


def test_gcext_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcext()
