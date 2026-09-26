"""cscld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cscld import cscld


def test_cscld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cscld()
