"""dtlpl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtlpl import dtlpl


def test_dtlpl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtlpl()
