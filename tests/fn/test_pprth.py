"""pprth is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pprth import pprth


def test_pprth_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pprth()
