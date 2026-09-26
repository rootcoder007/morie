"""tsarx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsarx import tsarx


def test_tsarx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsarx()
