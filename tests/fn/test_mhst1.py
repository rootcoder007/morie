"""mhst1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mhst1 import mantel_haenszel_or


def test_mhst1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mantel_haenszel_or(strata=None)
