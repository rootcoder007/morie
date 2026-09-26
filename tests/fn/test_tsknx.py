"""tsknx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsknx import tsknx


def test_tsknx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsknx()
