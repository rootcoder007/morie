"""gplin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gplin import gplin


def test_gplin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gplin()
