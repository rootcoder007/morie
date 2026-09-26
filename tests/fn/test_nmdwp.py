"""nmdwp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmdwp import dwnominate_polar


def test_nmdwp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dwnominate_polar(data=None)
