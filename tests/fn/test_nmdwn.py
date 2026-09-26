"""nmdwn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmdwn import dwnominate


def test_nmdwn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dwnominate(data=None)
