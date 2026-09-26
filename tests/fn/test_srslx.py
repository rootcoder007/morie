"""srslx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srslx import srslx


def test_srslx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srslx()
