"""sysmpl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sysmpl import systematic_sample


def test_sysmpl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        systematic_sample(frame=None, n=None)
