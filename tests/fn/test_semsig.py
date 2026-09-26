"""semsig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semsig import semsig


def test_semsig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semsig(resid=None, n=None)
