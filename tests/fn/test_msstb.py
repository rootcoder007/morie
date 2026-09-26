"""msstb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msstb import stress_s2


def test_msstb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stress_s2(X=None)
