"""gcjet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcjet import gcjet


def test_gcjet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcjet()
