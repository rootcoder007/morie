"""ppsal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppsal import ppsal


def test_ppsal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppsal()
