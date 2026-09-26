"""dtlmd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtlmd import dtlmd


def test_dtlmd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtlmd()
