"""cyclE is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cyclE import cyclone_intensity


def test_cyclE_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cyclone_intensity(v_max=None)
