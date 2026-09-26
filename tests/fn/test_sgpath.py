"""sgpath is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgpath import sgpath


def test_sgpath_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgpath()
