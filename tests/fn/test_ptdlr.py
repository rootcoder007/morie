"""ptdlr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptdlr import pp_delaunay_resid


def test_ptdlr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pp_delaunay_resid(data=None)
