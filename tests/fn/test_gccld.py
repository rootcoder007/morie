"""gccld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gccld import gccld


def test_gccld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gccld()
