"""svucs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svucs import uncovered_set


def test_svucs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        uncovered_set(data=None)
