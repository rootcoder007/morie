"""rsmnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsmnd import rsmnd


def test_rsmnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsmnd()
