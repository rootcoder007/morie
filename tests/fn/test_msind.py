"""msind is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msind import indscal


def test_msind_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        indscal(data=None)
