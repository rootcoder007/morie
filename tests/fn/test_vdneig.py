"""vdneig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdneig import vdneig


def test_vdneig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdneig()
